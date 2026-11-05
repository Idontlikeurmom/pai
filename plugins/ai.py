# GENERATED from ../plugins/ai.py by build_package.py — edit the original, not this copy.
"""PyAI plugin: train, sample, chat with and serve a character-level GPT."""
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["train", "train_until", "generate", "save_model", "load_model",
           "set_name", "set_act", "reply", "serve", "use_ollama",
           "train_job", "run_job", "teach_job", "set_creator"]

# Size of a fresh model. Checkpoints remember their own size, so old ones still load.
CFG = dict(block=256, dim=256, heads=8, layers=6)
OLD_CFG = dict(block=128, dim=128, heads=4, layers=4)  # checkpoints saved before sizes were stored
SIZES = {"small": dict(block=128, dim=128, heads=4, layers=4),
         "base": dict(block=256, dim=256, heads=8, layers=6),
         "large": dict(block=256, dim=384, heads=8, layers=8)}
BATCH, LR = 32, 6e-4
END = "<END>"
device = "cuda" if torch.cuda.is_available() else "cpu"
_amp = device == "cuda"  # half precision on GPU is several times faster
_scaler = torch.amp.GradScaler("cuda", enabled=_amp)
_state = {}


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        self.block = cfg["block"]
        self.tok = nn.Embedding(vocab, cfg["dim"])
        self.pos = nn.Embedding(cfg["block"], cfg["dim"])
        self.blocks = nn.Sequential(*[Block(cfg["dim"], cfg["heads"]) for _ in range(cfg["layers"])])
        self.ln = nn.LayerNorm(cfg["dim"])
        self.head = nn.Linear(cfg["dim"], vocab)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


class BPE:
    """A tiny byte-level BPE tokenizer in pure Python, in the style of GPT's."""

    def __init__(self, merges, vocab):
        self.merges = merges          # {(id, id): new_id} in the order they were learned
        self.vocab = vocab            # {id: bytes}
        self.size = len(vocab)

    @classmethod
    def train(cls, text, vocab_size=2000):
        vocab_size = max(256, int(vocab_size))
        ids = list(text.encode("utf-8"))
        merges, vocab, nxt = {}, {i: bytes([i]) for i in range(256)}, 256
        while nxt < vocab_size and len(ids) >= 2:
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break
            merges[pair] = nxt
            vocab[nxt] = vocab[pair[0]] + vocab[pair[1]]
            ids = _merge_ids(ids, pair, nxt)
            nxt += 1
        return cls(merges, vocab)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            ids = _merge_ids(ids, best, best_rank)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab.get(i, b"") for i in ids).decode("utf-8", errors="replace")

    def state(self):
        return {"merges": [[a, b, n] for (a, b), n in self.merges.items()],
                "vocab": {str(i): list(bs) for i, bs in self.vocab.items()}}

    @classmethod
    def load(cls, s):
        merges = {(a, b): n for a, b, n in s["merges"]}
        vocab = {int(i): bytes(bs) for i, bs in s["vocab"].items()}
        return cls(merges, vocab)


def _merge_ids(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _new_model(vocab, cfg):
    _state["cfg"] = cfg
    _state["step"] = 0
    _state["model"] = GPT(vocab, cfg).to(device)
    _state["opt"] = torch.optim.AdamW(_state["model"].parameters(), lr=LR, weight_decay=0.1)


def _setup_tokenizer(text, tokenizer, vocab_size):
    """Pick the tokenizer for a fresh model: character-level (default) or byte-level BPE."""
    if tokenizer and tokenizer.lower() == "bpe":
        _state["tok"] = BPE.train(text, vocab_size)
        _state["chars"] = None
    else:
        _state["tok"] = None
        _state["chars"] = sorted(set(text))


def _vocab():
    return _state["tok"].size if _state.get("tok") else len(_state["chars"])


def _ids(text):
    if _state.get("tok"):
        return _state["tok"].encode(text)
    stoi = {c: i for i, c in enumerate(_state["chars"])}
    return [stoi[c] for c in text if c in stoi]


def _text(ids):
    if _state.get("tok"):
        return _state["tok"].decode(list(ids))
    chars = _state["chars"]
    return "".join(chars[i] for i in ids)


def _encode(path):
    text = open(path, encoding="utf-8").read()
    if "model" not in _state:  # train() with no setup: default to a character model
        _setup_tokenizer(text, None, 0)
        _new_model(_vocab(), dict(CFG))
    return torch.tensor(_ids(text), dtype=torch.long)


def _batch(data):
    block = _state["cfg"]["block"]
    ix = torch.randint(len(data) - block - 1, (BATCH,))
    x = torch.stack([data[i:i + block] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block + 1] for i in ix]).to(device)
    return x, y


def _step(data, lr):
    model, opt = _state["model"], _state["opt"]
    for g in opt.param_groups:
        g["lr"] = lr
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=_amp):
        _, loss = model(*_batch(data))
    opt.zero_grad()
    _scaler.scale(loss).backward()
    _scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    _scaler.step(opt)
    _scaler.update()
    _state["step"] += 1
    return loss.item()


def train(path, steps=500):
    """Train on a text file; calling it again keeps training the same model."""
    data = _encode(path)
    _state["model"].train()
    for step in range(steps):
        loss = _step(data, LR)
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step}: loss {loss:.3f}")


@torch.no_grad()
def _val_loss(data, batches=20):
    model = _state["model"]
    model.eval()
    loss = sum(model(*_batch(data))[1].item() for _ in range(batches)) / batches
    model.train()
    return loss


def train_until(path, until, save_path="big_model.pt", save_every_min=20, lr=None, keep_best=False):
    """Train until a clock time like "2026-10-28 16:50", saving a checkpoint regularly.

    If save_path already exists, training resumes from it (e.g. after a restart).
    lr sets the peak learning rate (default LR). With keep_best, a checkpoint is only
    written when validation loss improves, so overtraining can never make the saved model worse.
    """
    peak = lr or LR
    deadline = datetime.strptime(until, "%Y-%m-%d %H:%M").timestamp()
    if os.path.exists(save_path):
        load_model(save_path)
        print(f"resumed from {save_path} at step {_state['step']}")
    data = _encode(path)
    split = int(0.99 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) < _state["cfg"]["block"] + 2:  # tiny datasets: score on the training text
        val_data = train_data
    params = sum(p.numel() for p in _state["model"].parameters())
    print(f"{params / 1e6:.1f}M parameters, {len(data):,} chars, training until {until}", flush=True)

    best = _val_loss(val_data, batches=50) if keep_best else None
    if keep_best:
        print(f"starting val loss {best:.3f}; only improvements are saved", flush=True)

    def checkpoint():
        nonlocal best
        if not keep_best:
            save_model(save_path)
            print(f"saved {save_path}  val loss {_val_loss(val_data):.3f}", flush=True)
            return
        val = _val_loss(val_data, batches=50)
        if val < best:
            best = val
            save_model(save_path)
            print(f"saved {save_path}  val loss {val:.3f} (new best)", flush=True)
        else:
            print(f"not saved: val loss {val:.3f} is not better than {best:.3f}", flush=True)

    start, last_save = time.time(), time.time()
    _state["model"].train()
    while time.time() < deadline:
        progress = (time.time() - start) / max(deadline - start, 1)
        warm = min(1.0, (_state["step"] + 1) / 300)
        lr = warm * (peak * 0.1 + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * progress)))
        loss = _step(train_data, lr)
        if _state["step"] % 50 == 0:
            print(f"{datetime.now():%H:%M} step {_state['step']}: loss {loss:.3f} lr {lr:.1e}", flush=True)
        if time.time() - last_save > save_every_min * 60:
            checkpoint()
            last_save = time.time()
            print("sample:", reply("Write a Python function that reverses a string.", n=200), flush=True)
    checkpoint()
    print(f"done at step {_state['step']}", flush=True)


def _train_steps(path, steps, peak, save_path):
    """Train for a fixed number of steps (the default when no stop time is given)."""
    data = _encode(path)
    split = int(0.99 * len(data))
    train_data, val = data[:split], data[split:]
    if len(val) < _state["cfg"]["block"] + 2:
        val = train_data
    params = sum(p.numel() for p in _state["model"].parameters())
    print(f"{params / 1e6:.1f}M parameters, vocab {_vocab()}, {len(data):,} tokens, {steps} steps", flush=True)
    _state["model"].train()
    for i in range(steps):
        warm = min(1.0, (_state["step"] + 1) / 200)
        rate = warm * (peak * 0.1 + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * i / max(steps, 1))))
        loss = _step(train_data, rate)
        if i % 100 == 0 or i == steps - 1:
            print(f"step {_state['step']}: loss {loss:.3f}", flush=True)
    save_model(save_path)
    print(f"saved {save_path}  val loss {_val_loss(val):.3f}", flush=True)


@torch.no_grad()
def generate(n=300, prompt="", stop=None):
    """Sample n tokens, optionally continuing from a prompt and stopping at a stop string."""
    model = _state["model"]
    start = _ids(prompt) or [0]
    idx = torch.tensor([start], device=device)
    was_training = model.training
    model.eval()
    for _ in range(n):
        logits, _ = model(idx[:, -model.block:])
        nxt = torch.multinomial(F.softmax(logits[:, -1] / 0.8, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if stop and _text(idx[0, len(start):].tolist()).endswith(stop):
            break
    model.train(was_training)
    return prompt + _text(idx[0, len(start):].tolist())


@torch.no_grad()
def _generate_stream(n, prompt, stop=None):
    """Like generate, but yield the reply text piece by piece as it is produced."""
    model = _state["model"]
    start = _ids(prompt) or [0]
    idx = torch.tensor([start], device=device)
    was_training = model.training
    model.eval()
    emitted = ""
    for _ in range(n):
        logits, _ = model(idx[:, -model.block:])
        nxt = torch.multinomial(F.softmax(logits[:, -1] / 0.8, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        full = _text(idx[0, len(start):].tolist())
        done = next((m for m in (stop, "\nUser:") if m and full.endswith(m)), None)
        if done:
            tail = full[len(emitted):len(full) - len(done)]
            if tail:
                yield tail
            break
        delta = full[len(emitted):]
        if delta:
            yield delta
            emitted = full
    model.train(was_training)


def save_model(path):
    tmp = path + ".tmp"
    torch.save({"chars": _state.get("chars"),
                "tok": _state["tok"].state() if _state.get("tok") else None,
                "cfg": _state["cfg"], "step": _state["step"],
                "weights": _state["model"].state_dict(), "opt": _state["opt"].state_dict()}, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint


def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("tok"):
        _state["tok"] = BPE.load(ckpt["tok"])
        _state["chars"] = None
    else:
        _state["tok"] = None
        _state["chars"] = ckpt["chars"]
    _new_model(_vocab(), ckpt.get("cfg", OLD_CFG))
    _state["model"].load_state_dict(ckpt["weights"])
    _state["step"] = ckpt.get("step", 0)
    if "opt" in ckpt:
        _state["opt"].load_state_dict(ckpt["opt"])


_persona = {"name": "AI", "act": "", "creator": ""}
_lock = threading.Lock()


def set_name(name):
    _persona["name"] = name


def set_act(act):
    _persona["act"] = act


def set_creator(creator):
    _persona["creator"] = creator


_ollama = {"model": None, "history": []}


def use_ollama(model="qwen2.5-coder:1.5b"):
    """Answer with a pretrained model running in Ollama instead of our own trained one."""
    _ollama["model"] = model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_BLOCK = re.compile(r"```[ \t]*(\w*)[ \t]*\n(.*?)```", re.S)
PAI_EXAMPLES = [
    {"role": "user", "content": "Write a function that returns the bigger of two numbers."},
    {"role": "assistant", "content": "```pai\nfunc bigger(a, b):\n    if a > b:\n        return a\n"
                                     "    return b\n\nsay bigger(3, 7)\n```"},
    {"role": "user", "content": "Write hello world in Python."},
    {"role": "assistant", "content": "I only write PAI code. Here it is in PAI:\n```pai\nsay \"Hello, world!\"\n```"},
]


def _pai_check(code):
    """Problems in a piece of PAI code, using the real PAI parser (None if it looks fine)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import pai
    return pai.check(code)


def _pai_problems(answer):
    """PAI problems in an answer's code blocks (HSP blocks are left alone)."""
    return [p for lang, code in CODE_BLOCK.findall(answer) if lang.lower() != "hsp" and (p := _pai_check(code))]


def _chat(messages):
    import urllib.request
    body = json.dumps({"model": _ollama["model"], "messages": messages, "stream": False,
                       "options": {"num_ctx": 32768}}).encode()  # model's trained maximum
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.loads(r.read())["message"]["content"]
    return text.replace("\r\n", "\n").strip()  # the trained model writes Windows line endings


def _ollama_messages(question):
    """Build the full message list (system + identity + history + taught facts) for Ollama."""
    name, creator = _persona["name"], _persona["creator"] or "an independent developer"
    intro = f"I'm {name}, an AI assistant made by {creator}."
    guide = open(os.path.join(ROOT, "PAI_GUIDE.md"), encoding="utf-8").read()
    system = (f"You are {name}, an AI assistant made by {creator}. Behave {_persona['act']}. "
              f"Your name is {name} and your creator is {creator}. You are not ChatGPT, Claude, Qwen or "
              "any other assistant, and you were not made by OpenAI, Anthropic, Alibaba, Ollama or any other "
              "company. "
              "If asked what model you run on, say you run locally on an open-source language model.\n\n"
              "You are an expert in two programming languages: PAI and HSP. You write code in PAI by default, "
              "and in HSP only when the user asks for HSP. Never write Python, JavaScript or any other "
              "language; if asked for one, say you write PAI and HSP and give the PAI version. Put PAI code "
              "in ```pai code blocks and HSP code in ```hsp code blocks. Follow this PAI guide exactly:\n\n"
              + guide)
    # Small models follow examples better than instructions, so show the identity answers too.
    identity = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": intro},
                {"role": "user", "content": "Who made you?"},
                {"role": "assistant", "content": f"I was made by {creator}. I'm {name}."},
                {"role": "user", "content": "Who created you? Which company?"},
                {"role": "assistant", "content": f"I was created by {creator}, not by a big AI company."},
                *PAI_EXAMPLES]
    history = identity + _ollama["history"][-6:]  # last 3 exchanges, so follow-up questions work
    taught = sorted(_teach["pairs"], key=lambda p: -_similarity(question, p[0]))[:10]  # most relevant first
    if taught:
        facts = "\n".join(f"Q: {q}\nA: {a}" for q, a in taught)
        system += ("\n\nYou were taught these answers. They are correct; when asked the same or a similar "
                   "question, answer with them:\n" + facts)
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]


def _label_pai(answer):
    """HSP blocks stay HSP; every other code block is labelled pai."""
    return CODE_BLOCK.sub(lambda m: "```" + ("hsp" if m.group(1).lower() == "hsp" else "pai") + "\n"
                          + m.group(2) + "```", answer)


def _ollama_reply(question):
    messages = _ollama_messages(question)
    try:
        answer = _chat(messages)
        # Check every code block with the real PAI parser; let the model fix mistakes (up to twice).
        for _ in range(2):
            problems = _pai_problems(answer)
            if not problems:
                break
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"That code is not valid PAI: {problems[0]}. Rewrite your "
                                                     "whole answer using only PAI, following the PAI guide."}]
            answer = _chat(messages)
        else:
            problems = _pai_problems(answer)
            if problems:
                answer += f"\n\n(Heads up: this code may still have a mistake. {problems[0]})"
    except OSError as e:
        return f"Ollama is not reachable ({e}). Start the Ollama app and try again."
    answer = _label_pai(answer)
    _ollama["history"] += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return answer


def _strip_think_stream(pieces):
    """Yield text with <think>...</think> removed, from a stream of partial pieces."""
    hold, in_think = "", False
    for piece in pieces:
        hold += piece
        while hold:
            if in_think:
                i = hold.find("</think>")
                if i < 0:
                    hold = hold[-8:] if len(hold) > 8 else hold
                    break
                hold, in_think = hold[i + 8:], False
            else:
                i = hold.find("<think>")
                if i < 0:
                    keep = max((k for k in range(1, min(7, len(hold)) + 1)
                                if "<think>".startswith(hold[-k:])), default=0)
                    out = hold[:len(hold) - keep] if keep else hold
                    if out:
                        yield out
                    hold = hold[len(hold) - keep:] if keep else ""
                    break
                if hold[:i]:
                    yield hold[:i]
                hold, in_think = hold[i + 7:], True
    if hold and not in_think:
        yield hold


def _ollama_stream(question):
    """Stream an Ollama reply token by token (and remember it in history)."""
    import urllib.request
    messages = _ollama_messages(question)
    body = json.dumps({"model": _ollama["model"], "messages": messages, "stream": True,
                       "options": {"num_ctx": 32768}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body, {"Content-Type": "application/json"})

    def pieces():
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                for raw in r:
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if obj.get("done"):
                        break
        except OSError as e:
            yield f"Ollama is not reachable ({e}). Start the Ollama app and try again."

    collected = []
    for visible in _strip_think_stream(pieces()):
        collected.append(visible)
        yield visible
    answer = _label_pai("".join(collected).strip())
    _ollama["history"] += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def reply(question, n=600):
    """Answer a question as the named persona. The act text is put in front of the chat."""
    recalled = _recall(question)
    if recalled is not None:  # a taught answer for this question: use it word for word
        return recalled
    if _ollama["model"]:
        return _ollama_reply(question)
    if "model" not in _state:
        return "I have no trained model yet. Use train or load_model first."
    prompt = f"{_persona['act']}\n\nUser: {question}\n{_persona['name']}:"
    has_end = True if _state.get("tok") else set(END) <= set(_state["chars"])
    with _lock:
        text = generate(n, prompt, stop=END if has_end else "\n\n")[len(prompt):]
    text = text.split(END)[0].split("\nUser:")[0]
    if not has_end:
        text = text.split("\n\n")[0]
    return text.strip() or "..."


def _reply_stream(question):
    """Yield a reply piece by piece (taught answer, Ollama, or own trained model)."""
    recalled = _recall(question)
    if recalled is not None:
        yield recalled
        return
    if _ollama["model"]:
        yield from _ollama_stream(question)
        return
    if "model" not in _state:
        yield "I have no trained model yet. Use train or load_model first."
        return
    prompt = f"{_persona['act']}\n\nUser: {question}\n{_persona['name']}:"
    has_end = True if _state.get("tok") else set(END) <= set(_state["chars"])
    with _lock:
        yield from _generate_stream(600, prompt, stop=END if has_end else "\n\n")


@torch.no_grad()
def _eval_loss(data):
    """Average next-token loss over a dataset, scanning it in order (not random)."""
    model, block = _state["model"], _state["cfg"]["block"]
    model.eval()
    losses = []
    for i in range(0, len(data) - block - 1, block):
        x = data[i:i + block].unsqueeze(0).to(device)
        y = data[i + 1:i + block + 1].unsqueeze(0).to(device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def eval_job(model, on=None, prompt=None):
    """The eval block: score a trained model on a test set (loss + perplexity) and show a sample."""
    if model.startswith("ollama:"):
        raise ValueError("eval scores your own trained .pt models, not ollama models")
    load_model(model)
    if not on:
        raise ValueError('eval needs a test set, e.g.  on "test.jsonl"')
    data = torch.tensor(_ids(_dataset_text(on)), dtype=torch.long)
    block = _state["cfg"]["block"]
    if len(data) < block + 2:
        print(f"{on}: too small to score (needs more than {block} tokens)")
    else:
        loss = _eval_loss(data)
        print(f"{model} on {on}:  loss {loss:.3f}   perplexity {math.exp(min(loss, 20)):.1f}   "
              f"({len(data):,} tokens)")
    q = prompt or "Write a function that adds two numbers."
    print(f"\nsample - {q}\n{reply(q, n=200)}")


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>__NAME__</title>
<style>
body{font:16px system-ui;margin:0;display:flex;flex-direction:column;height:100vh;background:#111;color:#eee}
h1{font-size:18px;margin:0;padding:14px 18px;border-bottom:1px solid #333}
#log{flex:1;overflow:auto;padding:18px}
.m{margin:8px 0;padding:10px 14px;border-radius:12px;max-width:75%;white-space:pre-wrap}
.u{background:#2b5cff;margin-left:auto}.a{background:#262626}
form{display:flex;gap:8px;padding:14px;border-top:1px solid #333}
input{flex:1;padding:12px;border-radius:10px;border:1px solid #444;background:#1c1c1c;color:#eee;font-size:16px}
button{padding:0 16px;border-radius:10px;border:0;background:#2b5cff;color:#fff;font-size:16px;cursor:pointer}
#mic.on{background:#d33}
.code{font-family:Consolas,monospace;font-size:14px}
.copy{display:block;margin-top:8px;padding:4px 10px;font-size:12px;background:#444}
.teach{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.teach button{padding:4px 10px;font-size:12px;background:#3a3a3a}
.teach textarea{width:100%;box-sizing:border-box;background:#1c1c1c;color:#eee;border:1px solid #555;
border-radius:8px;padding:8px;font:13px Consolas,monospace}
</style>
<h1>__NAME__</h1><div id=log></div>
<form id=f><input id=q placeholder="Ask __NAME__ anything..." autofocus autocomplete=off>
<button type=button id=mic title="Voice input">&#127908;</button><button>Send</button></form>
<script>
const TEACH=__TEACH__;
const log=document.getElementById('log'),q=document.getElementById('q');
function add(c,t){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function ask(text){
  add('u',text);const w=add('a','');
  let a='';
  try{
    const r=await fetch('/stream',{method:'POST',body:JSON.stringify({q:text})});
    const reader=r.body.getReader(),dec=new TextDecoder();
    for(;;){const {done,value}=await reader.read();if(done)break;
      a+=dec.decode(value,{stream:true});w.textContent=a;log.scrollTop=log.scrollHeight;}
  }catch(e){a='(error: '+e+')';w.textContent=a;}
  const isCode=a.includes('\\n');
  if(isCode)w.classList.add('code');
  const b=document.createElement('button');b.className='copy';b.textContent='Copy';
  b.onclick=()=>{navigator.clipboard.writeText(a);b.textContent='Copied'};
  w.appendChild(b);
  if(TEACH){
    const bar=document.createElement('div');bar.className='teach';
    const good=document.createElement('button');good.textContent=String.fromCodePoint(0x1F44D)+' Good';
    const fix=document.createElement('button');fix.textContent=String.fromCodePoint(0x270F)+' Teach';
    const send=async(answer,btn)=>{
      btn.textContent='Learning...';btn.disabled=true;
      const lr=await fetch('/teach',{method:'POST',body:JSON.stringify({q:text,a:answer})});
      btn.textContent=(await lr.json()).status;
    };
    good.onclick=()=>send(a,good);
    fix.onclick=()=>{
      const box=document.createElement('textarea');
      box.value=a;box.rows=Math.min(14,a.split(String.fromCharCode(10)).length+2);
      const save=document.createElement('button');save.textContent='Save correct answer';
      save.onclick=async()=>{if(box.value.trim()){await send(box.value,save);box.disabled=true;}};
      bar.append(box,save);box.focus();fix.remove();good.remove();
    };
    bar.append(good,fix);w.appendChild(bar);
  }
  if(!isCode&&window.speechSynthesis)speechSynthesis.speak(new SpeechSynthesisUtterance(a));
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();if(q.value.trim()){ask(q.value);q.value=''}};
const SR=window.SpeechRecognition||window.webkitSpeechRecognition,mic=document.getElementById('mic');
if(!SR){mic.disabled=true;mic.title='Voice input needs Chrome or Edge'}
else{const rec=new SR();rec.lang='en-US';
 rec.onresult=e=>{q.value=e.results[0][0].transcript;ask(q.value);q.value=''};
 rec.onend=()=>mic.classList.remove('on');
 mic.onclick=()=>{mic.classList.add('on');rec.start()}}
</script>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/api":
            info = {"name": _persona["name"], "endpoints": {
                "POST /api/chat": '{"message": "..."} -> {"reply": "..."}',
                "POST /stream": '{"q": "..."} -> reply streamed as plain text'}}
            self._send(json.dumps(info, indent=1).encode(), "application/json")
            return
        page = PAGE.replace("__NAME__", _persona["name"]).replace("__TEACH__", json.dumps(_teach["on"]))
        self._send(page.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/teach":
            self._send(json.dumps({"status": _learn(msg["q"], msg["a"])}).encode(), "application/json")
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for piece in _reply_stream(msg.get("q", "")):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        text = reply(msg.get("message") or msg.get("q", ""))  # /ask and /api/chat
        self._send(json.dumps({"reply": text, "a": text, "name": _persona["name"]}).encode(),
                   "application/json")


def _parse_host(host):
    """'8080', 'localhost:8080', '0.0.0.0:8080' or 'lan:8080' -> (bind_address, port)."""
    h = str(host).strip()
    addr, _, port = h.rpartition(":") if ":" in h else ("", "", h)
    bind = "0.0.0.0" if addr.strip().lower() in ("lan", "0.0.0.0", "*", "all") else "127.0.0.1"
    return bind, int(port or 8082)


def serve(port=8082, bind="127.0.0.1"):
    """Run the chat page at http://<host>:<port>. bind 0.0.0.0 to share on the LAN."""
    server = ThreadingHTTPServer((bind, port), _Handler)
    where = "this machine only" if bind == "127.0.0.1" else "your whole network"
    print(f"{_persona['name']} is live at http://localhost:{port}  ({where})  (Ctrl+C to stop)")
    if bind != "127.0.0.1":
        print("  serving on the LAN with no password - only do this on a network you trust")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


# ---- Block statements: train / run / teach ------------------------------------------

TIMEZONES = {"EST": "America/New_York", "EDT": "America/New_York", "ET": "America/New_York",
             "CST": "America/Chicago", "CDT": "America/Chicago", "MST": "America/Denver",
             "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles", "UTC": "UTC",
             "GMT": "Europe/London", "BST": "Europe/London", "CET": "Europe/Brussels",
             "CEST": "Europe/Brussels", "BRUSSELS": "Europe/Brussels"}
Q_KEYS = ("question", "instruction", "prompt", "input", "q", "user")
A_KEYS = ("answer", "output", "response", "completion", "a", "assistant")
_teach = {"on": False, "model": None, "path": None, "pairs": []}


def _end_time(end):
    """'15:00, EST' or '2026-10-28 15:00, CET' -> an aware datetime (today, or tomorrow if passed)."""
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    clock, _, zone = (part.strip() for part in end.partition(","))
    now = datetime.now(ZoneInfo(TIMEZONES.get(zone.upper(), zone))) if zone else datetime.now().astimezone()
    if " " in clock:
        return datetime.strptime(clock, "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
    hour, minute = map(int, clock.split(":"))
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return stop if stop > now else stop + timedelta(days=1)


def _example(q, a):
    return f"{_persona['act']}\n\nUser: {q.strip()}\n{_persona['name']}: {a.strip()}\n{END}\n"


def _pair(row):
    row = {str(k).strip().lower(): v for k, v in row.items()}
    q = next((str(row[k]) for k in Q_KEYS if row.get(k)), None)
    a = next((str(row[k]) for k in A_KEYS if row.get(k)), None)
    if row.get("instruction") and row.get("input"):
        q = f"{row['instruction']}\n{row['input']}"
    return (q, a) if q and a else None


def _pairs_from_row(row):
    """One dataset row -> a list of (question, answer) pairs.

    Supports the chat format {"messages": [{"role": "user"/"assistant", "content": ...}, ...]}
    - the same schema the fine-tuning datasets use - as well as flat question/answer rows.
    """
    msgs = row.get("messages")
    if isinstance(msgs, list):
        pairs, pending = [], None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role == "user" and content:
                pending = str(content)
            elif role == "assistant" and content and pending is not None:
                pairs.append((pending, str(content)))
                pending = None
        return pairs
    one = _pair(row)
    return [one] if one else []


def _dataset_pairs(path):
    """Read a CSV/JSON/JSONL dataset into a list of (question, answer) pairs."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        header = [h.strip().lower() for h in rows[0]] if rows else []
        if any(h in Q_KEYS for h in header) and any(h in A_KEYS for h in header):
            pairs = [_pair(dict(zip(header, r))) for r in rows[1:]]
        else:  # no header: first column is the question, second the answer
            pairs = [(r[0], r[1]) for r in rows if len(r) >= 2 and r[0] and r[1]]
    elif ext in (".json", ".jsonl"):
        raw = open(path, encoding="utf-8").read().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        pairs = [p for r in data if isinstance(r, dict) for p in _pairs_from_row(r)]
    else:
        raise ValueError(f"{path}: dataset must be .json, .jsonl, .csv or .txt")
    pairs = [p for p in pairs if p]
    if not pairs:
        raise ValueError(f"{path}: no question/answer pairs found (use keys like question/answer, "
                         "instruction/output, or a chat 'messages' list)")
    return pairs


def _dataset_text(path):
    """Turn a dataset into character-GPT training text (plain .txt is used as-is)."""
    if os.path.splitext(path)[1].lower() == ".txt":
        return open(path, encoding="utf-8").read()
    pairs = _dataset_pairs(path)
    print(f"{path}: {len(pairs)} question/answer pairs")
    return "".join(_example(q, a) for q, a in pairs)


def _training_file(model, dataset):
    text = _dataset_text(dataset)
    while len(text) < 20000:  # tiny datasets: repeat so there is enough to sample from
        text += text
    path = os.path.splitext(model)[0] + ".train.txt"
    open(path, "w", encoding="utf-8").write(text)
    return path


def train_job(model, dataset, env="Local", end=None, base=None, method="lora", epochs=None,
              name=None, creator=None, act=None, steps=None, lr=None, batch=None, size=None,
              tokenizer=None, vocab=None):
    """The train block.

    With `base` (a pretrained model), fine-tune it on `dataset` with LoRA and export a GGUF
    for Ollama - this needs a GPU, so it writes a Colab notebook. Without `base`, train a
    character-level model from scratch, on this PC or in Colab. `name`/`creator`/`act` set
    the assistant's identity.
    """
    if name:
        set_name(name)
    if act:
        set_act(act)
    if creator:
        set_creator(creator)
    if base:
        return _finetune_job(model, dataset, base, method, epochs)
    global BATCH
    cfg = dict(SIZES.get((size or "base").lower(), SIZES["base"]))
    if batch:
        BATCH = int(batch)
    peak = float(lr) if lr else LR
    data_file = _training_file(model, dataset)
    if "colab" in env.lower():
        from datetime import timedelta
        stop = _end_time(end) if end else datetime.now().astimezone() + timedelta(hours=2)
        notebook = _colab_notebook(model, data_file, stop)
        print(f"wrote {notebook}\nUpload it at colab.research.google.com, choose T4 GPU, Run all, "
              f"then download {os.path.basename(model)} into this folder when it's done.")
        return
    if not os.path.exists(model):
        _state.clear()  # a brand-new model
        text = open(data_file, encoding="utf-8").read()
        _setup_tokenizer(text, tokenizer, int(vocab) if vocab else 2000)
        _new_model(_vocab(), cfg)
    else:
        load_model(model)
    print(f"training {model} on {dataset} ({device.upper()}, vocab {_vocab()})")
    if end:
        local = _end_time(end).astimezone().strftime("%Y-%m-%d %H:%M")
        train_until(data_file, local, save_path=model, save_every_min=10, keep_best=True, lr=peak)
    else:
        _train_steps(data_file, int(steps) if steps else 2000, peak, model)


def _colab_notebook(model, data_file, stop):
    from datetime import timezone
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    name = os.path.basename(model)
    until_utc = stop.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")  # Colab's clock is UTC
    add("markdown", f"# Train {name}\nRuntime > Change runtime type > T4 GPU, then Runtime > Run all. "
                    "Checkpoints go to Google Drive, so re-running after a disconnect resumes.")
    add("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv")
    add("code", f'from google.colab import drive\ndrive.mount("/content/drive")\nSAVE = "/content/drive/MyDrive/{name}"')
    add("code", "!mkdir -p plugins")
    add("code", "%%writefile plugins/ai.py\n" + open(__file__, encoding="utf-8").read())
    add("code", "%%writefile train.txt\n" + open(data_file, encoding="utf-8").read())
    add("code", f"""
import sys
sys.path.insert(0, "plugins")
import ai
ai.set_name({json.dumps(_persona['name'])})
ai.set_act({json.dumps(_persona['act'])})
ai.CFG.update(dict(block=256, dim=384, heads=8, layers=8))
ai.BATCH = 64
ai.train_until("train.txt", "{until_utc}", save_path=SAVE, save_every_min=10, keep_best=True)
""")
    add("code", "from google.colab import files\nfiles.download(SAVE)")
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
          "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
    path = os.path.splitext(model)[0] + "_colab.ipynb"
    open(path, "w", encoding="utf-8").write(json.dumps(nb, indent=1))
    return path


# ---- Fine-tuning a pretrained model (LoRA -> GGUF -> Ollama) -------------------------

FINETUNE_BASES = {  # friendly names -> the matching Unsloth model on Hugging Face
    "qwen2.5-coder:1.5b": "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder:7b": "unsloth/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5:1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "unsloth/Qwen2.5-7B-Instruct",
    "llama3.2:1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct",
}

_FT_TRAIN = '''
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "__BASE__", max_seq_length=4096, load_in_4bit=__4BIT__)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, random_state=0, use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

data = load_dataset("json", data_files="train.jsonl", split="train")
data = data.map(lambda row: {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)})

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=SFTConfig(
    dataset_text_field="text", max_seq_length=4096, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=__EPOCHS__, learning_rate=2e-4, warmup_steps=5,
    lr_scheduler_type="linear", optim="adamw_8bit", logging_steps=5, seed=0, output_dir="outputs",
    report_to="none", fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported()))
trainer.train()
'''

_FT_TRY = '''
FastLanguageModel.for_inference(model)
system = __SYSTEM__
for question in ["Write a function that checks if a number is even.", "Who made you?"]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=300, temperature=0.3, do_sample=True)
    print("Q:", question)
    print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), "\\n" + "-" * 40)
'''

_FT_EXPORT = '''
import glob, shutil
model.save_pretrained_gguf("gguf_out", tokenizer, quantization_method="q4_k_m")
cands = [f for f in glob.glob("gguf_out*/**/*.gguf", recursive=True) + glob.glob("*.gguf") if "q4_k_m" in f.lower()]
shutil.copy(cands[0], "__NAME__.gguf")
print(cands[0], "->", "__NAME__.gguf")
from google.colab import files
files.download("__NAME__.gguf")
'''


def _resolve_base(base):
    """A friendly base name -> (Hugging Face id, note). A full id (with '/') is used as-is."""
    key = base.strip().lower()
    if key in FINETUNE_BASES:
        return FINETUNE_BASES[key], None
    if "/" in base:
        return base.strip(), None
    default = FINETUNE_BASES["qwen2.5-coder:1.5b"]
    return default, (f"'{base}' isn't a known base; using {default}. Pass a full Hugging Face id "
                     "like base \"unsloth/Qwen2.5-3B-Instruct\" to choose another.")


def _finetune_system():
    name = _persona["name"]
    creator = _persona["creator"] or "an independent developer"
    system = f"You are {name}, an AI assistant made by {creator}."
    if _persona["act"]:
        system += " " + _persona["act"]
    return system


def _finetune_jsonl(dataset):
    """Chat-format JSONL (one {"messages": [...]} per line) for fine-tuning."""
    ext = os.path.splitext(dataset)[1].lower()
    if ext in (".json", ".jsonl"):
        raw = open(dataset, encoding="utf-8").read().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        chat = [r for r in data if isinstance(r, dict) and isinstance(r.get("messages"), list)]
        if chat:  # already chat format - keep it exactly
            return "\n".join(json.dumps(r, ensure_ascii=False) for r in chat)
    system = _finetune_system()
    lines = []
    for q, a in _dataset_pairs(dataset):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": q}, {"role": "assistant", "content": a}]
        lines.append(json.dumps({"messages": msgs}, ensure_ascii=False))
    return "\n".join(lines)


def _write_modelfile(name):
    system = _finetune_system().replace('"', "'")
    path = name + ".Modelfile"
    open(path, "w", encoding="utf-8").write(f'FROM ./{name}.gguf\nSYSTEM "{system}"\n')
    return os.path.abspath(path)


def _finetune_notebook(name, jsonl, base_id, base_label, epochs, four_bit):
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"# Fine-tune {name}\nBase: `{base_label}` -> `{base_id}`\n\n"
                    "Runtime > Change runtime type > T4 GPU, then Runtime > Run all (~20-30 min).\n"
                    f"The last cell downloads `{name}.gguf`. Put it in this folder, then run "
                    f"`ollama create {name} -f {name}.Modelfile`.")
    add("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv")
    add("code", "%%capture\n!pip install unsloth")
    add("code", "%%writefile train.jsonl\n" + jsonl)
    add("markdown", "## Train (LoRA)")
    add("code", _FT_TRAIN.replace("__BASE__", base_id).replace("__EPOCHS__", str(epochs))
        .replace("__4BIT__", "True" if four_bit else "False"))
    add("markdown", "## Try it")
    add("code", _FT_TRY.replace("__SYSTEM__", json.dumps(_finetune_system())))
    add("markdown", "## Export for Ollama and download")
    add("code", _FT_EXPORT.replace("__NAME__", name))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
          "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
    path = name + "_finetune.ipynb"
    open(path, "w", encoding="utf-8").write(json.dumps(nb, indent=1))
    return os.path.abspath(path)


def _finetune_job(model, dataset, base, method="lora", epochs=None):
    """Fine-tune `base` on `dataset`: write a Colab notebook and a matching Ollama Modelfile."""
    method = (method or "lora").lower()
    if method not in ("lora", "qlora"):
        raise ValueError(f"train: method must be 'lora' or 'qlora' (got {method!r})")
    name = re.sub(r"\s+", "-", os.path.splitext(os.path.basename(model))[0].strip().lower()) or "model"
    base_id, note = _resolve_base(base)
    jsonl = _finetune_jsonl(dataset)
    examples = jsonl.count("\n") + 1 if jsonl else 0
    n = int(float(epochs)) if epochs else 3
    nb = _finetune_notebook(name, jsonl, base_id, base, n, four_bit=method == "qlora")
    modelfile = _write_modelfile(name)
    if note:
        print(note)
    print(f"{dataset}: {examples} training examples -> {name}")
    print(f"wrote {nb}")
    print(f"wrote {modelfile}")
    print("Fine-tuning needs a GPU, so it runs on Colab's free T4:")
    print("  1) upload the notebook at colab.research.google.com")
    print("  2) Runtime > Change runtime type > T4 GPU, then Runtime > Run all (~20-30 min)")
    print(f"  3) it downloads {name}.gguf - put it in this folder")
    print(f"  4) ollama create {name} -f {os.path.basename(modelfile)}")
    print(f'  5) use it in PAI:  run "ollama:{name}"')


def run_job(model, name=None, host="localhost:8082", act=None, creator=None, teach=False):
    """The run block: load a model file (or "ollama:<model>") and serve its chat page."""
    if name:
        set_name(name)
    if act:
        set_act(act)
    if creator:
        set_creator(creator)
    if model.startswith("ollama:"):
        use_ollama(model.split(":", 1)[1])
    else:
        load_model(model)
    _teach.update(on=teach, model=model, path=_persona["name"].replace(" ", "_") + ".taught.json")
    if teach and os.path.exists(_teach["path"]):
        _teach["pairs"] = json.load(open(_teach["path"], encoding="utf-8"))
        print(f"loaded {len(_teach['pairs'])} taught answers from {_teach['path']}")
    bind, port = _parse_host(host)
    serve(port, bind)


def teach_job(model, name=None, host="localhost:8082", act=None, creator=None):
    """The teach block: like run, plus a Teach button that makes the model learn your answer."""
    run_job(model, name=name, host=host, act=act, creator=creator, teach=True)


def _normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _similarity(a, b):
    """0..1: how alike two questions are (spelling and shared words)."""
    from difflib import SequenceMatcher
    a, b = _normalize(a), _normalize(b)
    words_a, words_b = set(a.split()), set(b.split())
    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
    return max(SequenceMatcher(None, a, b).ratio(), overlap)


def _recall(question, threshold=0.8):
    """The taught answer for this question (or a very similar one), if there is one."""
    if not _teach["on"] or not _teach["pairs"]:
        return None
    score, answer = max((_similarity(question, q), a) for q, a in _teach["pairs"])
    return answer if score >= threshold else None


def _learn(question, answer):
    """Teach mode: remember a corrected answer, and for our own models also train on it."""
    global BATCH
    key = _normalize(question)
    _teach["pairs"] = [p for p in _teach["pairs"] if _normalize(p[0]) != key] + [[question, answer]]
    json.dump(_teach["pairs"], open(_teach["path"], "w", encoding="utf-8"), indent=1)
    problems = _pai_problems(answer)
    warning = f" (but its code isn't valid PAI: {problems[0]})" if problems else ""
    if _ollama["model"]:
        return f"Learned!{warning}"
    text = "".join(_example(q, a) for q, a in _teach["pairs"][-20:])
    while len(text) < 4 * _state["cfg"]["block"]:
        text += text
    data = torch.tensor(_ids(text), dtype=torch.long)
    with _lock:
        saved_batch, BATCH = BATCH, 8  # small batches keep a lesson to well under a minute on CPU
        try:
            _state["model"].train()
            for _ in range(20):
                loss = _step(data, 1e-4)
        finally:
            BATCH = saved_batch
            _state["model"].eval()
        if os.path.exists(_teach["model"]):
            shutil.copy(_teach["model"], _teach["model"] + ".bak")  # so a bad lesson can be undone
        save_model(_teach["model"])
    return f"Learned! (loss {loss:.2f}){warning}  (backup: {os.path.basename(_teach['model'])}.bak)"
