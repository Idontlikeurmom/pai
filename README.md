# PAI

**PAI** is a small, friendly programming language with AI built into the syntax.
It reads like plain English, its **core runs on nothing but the Python standard
library**, and it has first-class commands for **training, running, and teaching
AI models** — so it's a language made for building AI.

```pai
let name = "world"
say f"Hello, {name}!"

train "my_model.pt"
  dataset "qa.json"
  env "Local"
```

## Why PAI

- **Reads like English** — `let`, `say`, `repeat 3 times`, `func`, `use`.
- **Zero dependencies** for the core language — just Python 3.9+ and the stdlib.
- **AI is a keyword, not a library** — `train`, `run`, `teach`, and an `ai()`
  function are part of the language.
- **Beginner-friendly errors** — mistakes are explained in plain words, with the
  line number and a hint (it even translates Python habits like `print` or `def`).

## Install & run

From PyPI (the core language):

```bash
pip install pailang
pai examples/hello.pai
pai --version
```

Or clone the repo — no install needed, and this is how you also get the AI
training commands (`train` / `run` / `teach`) and the examples:

```bash
git clone https://github.com/Idontlikeurmom/pai.git
cd pai
python cli.py                 # interactive mode
python cli.py examples/hello.pai
python cli.py --version
```

Optional, to get a `pai` command on your PATH:

```bash
pip install .          # core language only (zero dependencies)
pip install ".[ai]"    # also installs PyTorch for the training commands
pai examples/hello.pai
```

> The **core language** has zero dependencies (pure standard library). The
> training commands (`train`, `run`, `teach`) need **PyTorch** (`pip install
> ".[ai]"`). The `ai()` function and serving a pretrained model use a local
> [Ollama](https://ollama.com) server. Run AI commands from the cloned repo so
> the `plugins/` folder is present.

## Learn the language

See **[PAI_GUIDE.md](PAI_GUIDE.md)** for the full tour: variables, loops, functions,
lists, maps, slices, built-ins, and the AI commands. Examples live in
[`examples/`](examples/).

## A taste

```pai
func greet(who, greeting="Hi"):
    return f"{greeting}, {who}!"

for n in ["Ada", "Linus"]:
    say greet(n, greeting="Hello")

let answer = ai("Give me one fun fact about octopuses.")
say answer
```

## License

MIT — see [LICENSE](LICENSE).
