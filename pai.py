# GENERATED from ../pai.py by build_package.py — edit the original, not this copy.
"""PAI: a small Python-like programming language with built-in AI commands.

Usage:
    python pai.py program.pai     run a program
    python pai.py                 interactive mode (type code, see results)

See PAI_GUIDE.md for the language.
"""
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

KEYWORDS = {"let", "if", "elif", "else", "while", "for", "in", "repeat", "times", "func", "return",
            "break", "continue", "pass", "and", "or", "not", "true", "false", "none", "say", "use"}
AI_BLOCKS = {"train": "train_job", "run": "run_job", "teach": "teach_job", "eval": "eval_job"}
OPERATORS = ["**", "//", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=",
             "+", "-", "*", "/", "%", "<", ">", "=", "(", ")", "[", "]", "{", "}", ",", ":", "."]
PYTHON_HINTS = {"def": "PAI functions start with 'func', like: func add(a, b):",
                "print": "PAI prints with 'say', like: say \"hello\"",
                "import": "PAI loads plugins and files with 'use', like: use ai",
                "True": "PAI writes true in lowercase", "False": "PAI writes false in lowercase",
                "None": "PAI writes none in lowercase", "null": "PAI uses none", "nil": "PAI uses none",
                "function": "PAI functions start with 'func'", "var": "PAI creates variables with 'let'",
                "const": "PAI creates variables with 'let'", "elseif": "PAI uses 'elif'"}
TYPE_NAMES = {"str": "text", "int": "number", "float": "number", "list": "list", "dict": "map",
              "bool": "true/false", "NoneType": "none"}


class PaiError(Exception):
    def __init__(self, message, line=None):
        super().__init__(message)
        self.message, self.line = message, line

    def __str__(self):
        return f"Line {self.line}: {self.message}" if self.line else self.message


# ---- Tokenizer ------------------------------------------------------------------------

class Token:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind, value, line):
        self.kind, self.value, self.line = kind, value, line

    def __repr__(self):
        return f"{self.kind}:{self.value!r}"


def read_string(line, i, lineno):
    quote, j, out = line[i], i + 1, []
    escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
    while j < len(line):
        ch = line[j]
        if ch == "\\" and j + 1 < len(line):
            out.append(escapes.get(line[j + 1], "\\" + line[j + 1]))
            j += 2
        elif ch == quote:
            return "".join(out), j + 1
        else:
            out.append(ch)
            j += 1
    raise PaiError("text is missing its closing quote", lineno)


def normalize_source(source):
    """Windows line endings and tabs -> plain newlines and 4 spaces."""
    return source.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")


def tokenize(source):
    tokens, indents, depth = [], [0], 0  # inside brackets, line breaks and indentation don't count
    lines = normalize_source(source).split("\n")
    for lineno, line in enumerate(lines, 1):
        i, n = 0, len(line)
        if depth == 0:
            stripped = line.lstrip(" ")
            if not stripped or stripped.startswith(("--", "#")):
                continue
            indent = n - len(stripped)
            if indent > indents[-1]:
                indents.append(indent)
                tokens.append(Token("INDENT", None, lineno))
            while indent < indents[-1]:
                indents.pop()
                tokens.append(Token("DEDENT", None, lineno))
            if indent != indents[-1]:
                raise PaiError("this line's indentation doesn't line up with the lines above", lineno)
            i = indent
        while i < n:
            ch = line[i]
            if ch == " ":
                i += 1
            elif line.startswith("--", i) or ch == "#":
                break
            elif ch.isdigit():
                j = i
                while j < n and (line[j].isdigit() or line[j] == "_"):
                    j += 1
                if j + 1 < n and line[j] == "." and line[j + 1].isdigit():
                    j += 1
                    while j < n and line[j].isdigit():
                        j += 1
                text = line[i:j].replace("_", "")
                tokens.append(Token("NUMBER", float(text) if "." in text else int(text), lineno))
                i = j
            elif ch.isalpha() or ch == "_":
                j = i
                while j < n and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                word = line[i:j]
                if word in ("f", "F") and j < n and line[j] in "\"'":
                    text, i = read_string(line, j, lineno)
                    tokens.append(Token("FSTRING", text, lineno))
                else:
                    tokens.append(Token("KEYWORD" if word in KEYWORDS else "NAME", word, lineno))
                    i = j
            elif ch in "\"'":
                text, i = read_string(line, i, lineno)
                tokens.append(Token("STRING", text, lineno))
            else:
                for op in OPERATORS:
                    if line.startswith(op, i):
                        if op in "([{":
                            depth += 1
                        elif op in ")]}":
                            depth = max(0, depth - 1)
                        tokens.append(Token("OP", op, lineno))
                        i += len(op)
                        break
                else:
                    raise PaiError(f"PAI doesn't understand the character {ch!r}", lineno)
        if depth == 0 and tokens and tokens[-1].kind not in ("NEWLINE", "INDENT", "DEDENT"):
            tokens.append(Token("NEWLINE", None, lineno))
    if depth:
        raise PaiError("a bracket ( [ or { is never closed", len(lines))
    last = len(lines)
    tokens += [Token("DEDENT", None, last) for _ in indents[1:]]
    tokens.append(Token("EOF", None, last))
    return tokens


# ---- Parser ---------------------------------------------------------------------------

class Node:
    def __init__(self, kind, line, **fields):
        self.kind, self.line = kind, line
        self.__dict__.update(fields)

    def __repr__(self):
        fields = {k: v for k, v in self.__dict__.items() if k not in ("kind", "line")}
        return f"{self.kind}{fields}"


def describe(tok):
    if tok.kind == "NEWLINE":
        return "the end of the line"
    if tok.kind in ("INDENT", "DEDENT"):
        return "a change in indentation"
    if tok.kind == "EOF":
        return "the end of the program"
    return repr(tok.value)


class Parser:
    def __init__(self, source):
        self.lines = normalize_source(source).split("\n")
        self.tokens = tokenize(source)
        self.pos = 0

    # helpers
    def peek(self, ahead=0):
        return self.tokens[min(self.pos + ahead, len(self.tokens) - 1)]

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def check(self, kind, value=None, ahead=0):
        tok = self.peek(ahead)
        return tok.kind == kind and (value is None or tok.value == value)

    def accept(self, kind, value=None):
        if self.check(kind, value):
            return self.next()
        return None

    def expect(self, kind, value=None, what=None):
        if self.check(kind, value):
            return self.next()
        tok = self.peek()
        raise PaiError(f"expected {what or repr(value) or kind.lower()} but found {describe(tok)}", tok.line)

    # program structure
    def program(self):
        body = []
        while not self.check("EOF"):
            body.append(self.statement())
        return body

    def block(self):
        self.expect("OP", ":", "':' at the end of the line")
        if not self.accept("NEWLINE"):  # one-line block: if x > 1: say x
            return [self.simple_statement()]
        if not self.check("INDENT"):
            raise PaiError("expected an indented block on the next line", self.peek().line)
        self.next()
        body = []
        while not self.accept("DEDENT"):
            if self.check("EOF"):
                break
            body.append(self.statement())
        return body

    def statement(self):
        tok = self.peek()
        if tok.kind == "KEYWORD":
            word = tok.value
            if word == "if":
                return self.if_statement()
            if word == "while":
                self.next()
                return Node("while", tok.line, cond=self.expression(), body=self.block())
            if word == "for":
                self.next()
                var = self.expect("NAME", what="a variable name").value
                self.expect("KEYWORD", "in", "'in'")
                return Node("for", tok.line, var=var, items=self.expression(), body=self.block())
            if word == "repeat":
                self.next()
                count = self.expression()
                self.expect("KEYWORD", "times", "'times'")
                return Node("repeat", tok.line, count=count, body=self.block())
            if word == "func":
                return self.func_statement()
        if tok.kind == "NAME" and tok.value in ("def", "import", "function", "var", "const") \
                and self.peek(1).kind in ("NAME", "STRING"):
            raise PaiError(PYTHON_HINTS[tok.value], tok.line)
        if tok.kind == "NAME" and tok.value in AI_BLOCKS and self.check("STRING", ahead=1) \
                and self.check("NEWLINE", ahead=2):
            return self.ai_block()
        return self.simple_statement()

    def simple_statement(self):
        tok = self.peek()
        node = self.simple_statement_body()
        if not self.accept("NEWLINE") and not self.check("EOF") and not self.check("DEDENT"):
            raise PaiError(f"expected the end of the line but found {describe(self.peek())}", self.peek().line)
        return node

    def simple_statement_body(self):
        tok = self.peek()
        if tok.kind == "KEYWORD":
            word = tok.value
            if word == "let":
                self.next()
                name = self.expect("NAME", what="a variable name after 'let'").value
                self.expect("OP", "=", "'='")
                return Node("let", tok.line, name=name, value=self.expression())
            if word == "say":
                self.next()
                values = []
                if not self.check("NEWLINE") and not self.check("EOF"):
                    values.append(self.expression())
                    while self.accept("OP", ","):
                        values.append(self.expression())
                return Node("say", tok.line, values=values)
            if word == "return":
                self.next()
                value = None if self.check("NEWLINE") or self.check("EOF") else self.expression()
                return Node("return", tok.line, value=value)
            if word in ("break", "continue", "pass"):
                self.next()
                return Node(word, tok.line)
            if word == "use":
                self.next()
                target = self.next()
                if target.kind not in ("NAME", "STRING"):
                    raise PaiError("expected a plugin name or a \"file.pai\" after 'use'", tok.line)
                return Node("use", tok.line, target=target.value, is_file=target.kind == "STRING")
        expr = self.expression()
        op = self.peek()
        if op.kind == "OP" and op.value in ("=", "+=", "-=", "*=", "/="):
            if expr.kind not in ("name", "index", "attr"):
                raise PaiError("you can only assign to a variable, a list item or a map key", op.line)
            self.next()
            return Node("assign", op.line, target=expr, op=op.value, value=self.expression())
        return Node("expr", tok.line, expr=expr)

    def if_statement(self):
        tok = self.next()
        branches = [(self.expression(), self.block())]
        otherwise = None
        while True:
            if self.accept("KEYWORD", "elif"):
                branches.append((self.expression(), self.block()))
            elif self.accept("KEYWORD", "else"):
                otherwise = self.block()
                break
            else:
                break
        return Node("if", tok.line, branches=branches, otherwise=otherwise)

    def func_statement(self):
        tok = self.next()
        name = self.expect("NAME", what="a function name").value
        self.expect("OP", "(", "'('")
        params = []
        while not self.accept("OP", ")"):
            pname = self.expect("NAME", what="a parameter name").value
            default = self.expression() if self.accept("OP", "=") else None
            params.append((pname, default))
            if not self.check("OP", ")"):
                self.expect("OP", ",", "',' or ')'")
        return Node("func", tok.line, name=name, params=params, body=self.block())

    def ai_block(self):
        kind_tok = self.next()
        target = self.next().value
        self.expect("NEWLINE")
        options = {}
        if self.accept("INDENT"):
            while not self.accept("DEDENT"):
                key_tok = self.expect("NAME", what="an option name")
                raw = strip_comment(self.lines[key_tok.line - 1].strip()[len(key_tok.value):].strip())
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                    raw = raw[1:-1]
                options[key_tok.value] = raw
                while not self.accept("NEWLINE"):
                    if self.check("EOF"):
                        break
                    self.next()
        return Node("ai_block", kind_tok.line, command=kind_tok.value, target=target, options=options)

    # expressions, lowest precedence first
    def expression(self):
        return self.or_expr()

    def or_expr(self):
        left = self.and_expr()
        while self.check("KEYWORD", "or"):
            line = self.next().line
            left = Node("logic", line, op="or", left=left, right=self.and_expr())
        return left

    def and_expr(self):
        left = self.not_expr()
        while self.check("KEYWORD", "and"):
            line = self.next().line
            left = Node("logic", line, op="and", left=left, right=self.not_expr())
        return left

    def not_expr(self):
        if self.check("KEYWORD", "not"):
            line = self.next().line
            return Node("unary", line, op="not", value=self.not_expr())
        return self.comparison()

    def comparison(self):
        left = self.additive()
        while True:
            tok = self.peek()
            if tok.kind == "OP" and tok.value in ("==", "!=", "<", ">", "<=", ">="):
                self.next()
                left = Node("binary", tok.line, op=tok.value, left=left, right=self.additive())
            elif tok.kind == "KEYWORD" and tok.value == "in":
                self.next()
                left = Node("binary", tok.line, op="in", left=left, right=self.additive())
            elif tok.kind == "KEYWORD" and tok.value == "not" and self.check("KEYWORD", "in", ahead=1):
                self.next()
                self.next()
                left = Node("binary", tok.line, op="not in", left=left, right=self.additive())
            else:
                return left

    def additive(self):
        left = self.term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            tok = self.next()
            left = Node("binary", tok.line, op=tok.value, left=left, right=self.term())
        return left

    def term(self):
        left = self.unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "//", "%"):
            tok = self.next()
            left = Node("binary", tok.line, op=tok.value, left=left, right=self.unary())
        return left

    def unary(self):
        if self.check("OP", "-"):
            line = self.next().line
            return Node("unary", line, op="-", value=self.unary())
        return self.power()

    def power(self):
        base = self.postfix()
        if self.check("OP", "**"):
            line = self.next().line
            return Node("binary", line, op="**", left=base, right=self.unary())
        return base

    def postfix(self):
        node = self.atom()
        while True:
            if self.check("OP", "("):
                line = self.next().line
                args, kwargs = [], {}
                while not self.accept("OP", ")"):
                    if self.check("NAME") and self.check("OP", "=", ahead=1):
                        key = self.next().value
                        self.next()
                        kwargs[key] = self.expression()
                    else:
                        args.append(self.expression())
                    if not self.check("OP", ")"):
                        self.expect("OP", ",", "',' or ')'")
                node = Node("call", line, func=node, args=args, kwargs=kwargs)
            elif self.check("OP", "["):
                line = self.next().line
                index = self.slice_or_index(line)
                self.expect("OP", "]", "']'")
                node = Node("index", line, obj=node, index=index)
            elif self.check("OP", "."):
                line = self.next().line
                name = self.expect("NAME", what="a name after '.'").value
                node = Node("attr", line, obj=node, name=name)
            else:
                return node

    def slice_or_index(self, line):
        """items[i], or a slice items[start:end:step] where every part is optional."""
        parts = [None]
        while True:
            if self.check("OP", ":"):
                self.next()
                parts.append(None)
                if len(parts) > 3:
                    raise PaiError("a slice has at most two ':' (start:end:step)", line)
            elif self.check("OP", "]"):
                break
            elif parts[-1] is None:
                parts[-1] = self.expression()
            else:
                self.expect("OP", "]", "']' or ':'")
        if len(parts) == 1:
            if parts[0] is None:
                raise PaiError("expected a position inside [ ]", line)
            return parts[0]
        parts += [None] * (3 - len(parts))
        return Node("slice", line, start=parts[0], stop=parts[1], step=parts[2])

    def atom(self):
        tok = self.next()
        if tok.kind == "NUMBER" or tok.kind == "STRING":
            return Node("const", tok.line, value=tok.value)
        if tok.kind == "FSTRING":
            return Node("fstring", tok.line, parts=parse_fstring(tok.value, tok.line))
        if tok.kind == "NAME":
            return Node("name", tok.line, name=tok.value)
        if tok.kind == "KEYWORD" and tok.value in ("true", "false", "none"):
            return Node("const", tok.line, value={"true": True, "false": False, "none": None}[tok.value])
        if tok.kind == "OP" and tok.value == "(":
            node = self.expression()
            self.expect("OP", ")", "')'")
            return node
        if tok.kind == "OP" and tok.value == "[":
            items = []
            while not self.accept("OP", "]"):
                items.append(self.expression())
                if not self.check("OP", "]"):
                    self.expect("OP", ",", "',' or ']'")
            return Node("list", tok.line, items=items)
        if tok.kind == "OP" and tok.value == "{":
            pairs = []
            while not self.accept("OP", "}"):
                key = self.expression()
                self.expect("OP", ":", "':' between a key and its value")
                pairs.append((key, self.expression()))
                if not self.check("OP", "}"):
                    self.expect("OP", ",", "',' or '}'")
            return Node("map", tok.line, pairs=pairs)
        if tok.kind == "KEYWORD":
            raise PaiError(f"'{tok.value}' can't be used here", tok.line)
        raise PaiError(f"expected a value but found {describe(tok)}", tok.line)


def strip_comment(text):
    quoted = None
    for i, ch in enumerate(text):
        if quoted:
            if ch == quoted:
                quoted = None
        elif ch in "\"'":
            quoted = ch
        elif text.startswith("--", i) or ch == "#":
            return text[:i].rstrip()
    return text


def parse_fstring(text, line):
    """f"Hi {name}!" -> ["Hi ", <expr name>, "!"]"""
    parts, i, buf = [], 0, []
    while i < len(text):
        ch = text[i]
        if text.startswith("{{", i) or text.startswith("}}", i):
            buf.append(ch)
            i += 2
        elif ch == "{":
            end = text.find("}", i)
            if end < 0:
                raise PaiError("a '{' in this f-text is never closed", line)
            if buf:
                parts.append("".join(buf))
                buf = []
            sub = Parser(text[i + 1:end])
            expr = sub.expression()
            parts.append(expr)
            i = end + 1
        else:
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def parse(source):
    return Parser(source).program()


# ---- Values and output ----------------------------------------------------------------

class Function:
    def __init__(self, name, params, body, closure, interp):
        self.name, self.params, self.body, self.closure, self.interp = name, params, body, closure, interp

    def __call__(self, *args, **kwargs):
        return self.interp.call_function(self, list(args), kwargs)


def show(value, nested=False):
    """How PAI prints a value."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e16 else repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False) if nested else value
    if isinstance(value, list):
        return "[" + ", ".join(show(v, True) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{show(k, True)}: {show(v, True)}" for k, v in value.items()) + "}"
    if isinstance(value, Function):
        return f"<func {value.name}>"
    if callable(value):
        return f"<built-in {getattr(value, '__name__', 'function')}>"
    return str(value)


def type_name(value):
    if isinstance(value, Function) or callable(value):
        return "function"
    return TYPE_NAMES.get(type(value).__name__, type(value).__name__)


def to_number(text):
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return text
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            return float(text)
        except (TypeError, ValueError):
            raise PaiError(f"can't turn {show(text, True)} into a number") from None


def ask_ai(question, model="qwen2.5-coder:1.5b"):
    """Ask the local Ollama model a question and return its answer as text."""
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": str(question)}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except OSError as e:
        raise PaiError(f"ai() needs Ollama running ({e})") from None


def split_text(text, sep=None):
    """split(text, sep); an empty separator splits text into its letters."""
    return list(text) if sep == "" else text.split(sep)


def read_file(path):
    return Path(path).read_text(encoding="utf-8")


def write_file(path, text):
    Path(path).write_text(show(text), encoding="utf-8")


BUILTINS = {
    "len": len, "range": lambda *a: list(range(*a)), "str": show, "text": show, "num": to_number,
    "int": lambda x: int(to_number(x)), "round": round, "abs": abs, "min": min, "max": max, "sum": sum,
    "sorted": lambda items: sorted(items), "reversed": lambda items: list(reversed(items)),
    "input": lambda prompt="": input(prompt), "ask": lambda prompt="": input(prompt),
    "random": random.randint, "choice": random.choice, "type": type_name,
    "join": lambda items, sep="": sep.join(show(i) for i in items),
    "split": lambda text, sep=None: split_text(text, sep), "upper": str.upper, "lower": str.lower,
    "keys": lambda m: list(m.keys()), "values": lambda m: list(m.values()),
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil, "pi": math.pi,
    "wait": time.sleep, "now": lambda: time.strftime("%H:%M:%S"), "today": lambda: time.strftime("%Y-%m-%d"),
    "read": read_file, "write": write_file, "exists": os.path.exists, "ai": ask_ai,
}
METHODS = {"push": "append", "add": "append", "pop": "pop", "insert": "insert", "remove": "remove",
           "sort": "sort", "reverse": "reverse", "count": "count", "index": "index", "upper": "upper",
           "lower": "lower", "strip": "strip", "replace": "replace", "split": "split",
           "startswith": "startswith", "endswith": "endswith", "find": "find", "get": "get",
           "join": "join", "has": "__contains__", "contains": "__contains__"}
KNOWN_METHODS = set(METHODS) | {"keys", "values", "items", "length"}


# ---- Interpreter ----------------------------------------------------------------------

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class Scope:
    def __init__(self, parent=None):
        self.vars, self.parent = {}, parent

    def find(self, name):
        scope = self
        while scope is not None:
            if name in scope.vars:
                return scope
            scope = scope.parent
        return None


class Interpreter:
    def __init__(self, out=None):
        self.out = out or (lambda text: print(text, flush=True))
        self.builtins = Scope()
        self.builtins.vars.update(BUILTINS)
        self.globals = Scope(self.builtins)
        self.plugins = {}
        self.repl = False

    def run(self, source):
        self.exec_block(parse(source), self.globals)

    # statements
    def exec_block(self, body, scope):
        for node in body:
            self.exec(node, scope)

    def exec(self, node, scope):
        try:
            getattr(self, "s_" + node.kind)(node, scope)
        except (PaiError, ReturnSignal, BreakSignal, ContinueSignal):
            raise
        except RecursionError:
            raise PaiError("too much recursion (a function keeps calling itself)", node.line) from None
        except Exception as e:  # turn Python errors into PAI errors with a line number
            raise PaiError(friendly(e), node.line) from None

    def s_let(self, node, scope):
        scope.vars[node.name] = self.eval(node.value, scope)

    def s_assign(self, node, scope):
        value = self.eval(node.value, scope)
        target = node.target
        if node.op != "=":
            value = binary(node.op[0], self.eval(target, scope), value)
        if target.kind == "name":
            owner = scope.find(target.name)
            (owner if owner is not None and owner is not self.builtins else scope).vars[target.name] = value
        elif target.kind == "index":
            obj = self.eval(target.obj, scope)
            if target.index.kind == "slice":
                part = lambda p: None if p is None else self.eval(p, scope)
                obj[slice(part(target.index.start), part(target.index.stop), part(target.index.step))] = value
            else:
                obj[self.eval(target.index, scope)] = value
        else:
            obj = self.eval(target.obj, scope)
            if not isinstance(obj, dict):
                raise PaiError(f"can't set .{target.name} on a {type_name(obj)}")
            obj[target.name] = value

    def s_say(self, node, scope):
        self.out(" ".join(show(self.eval(v, scope)) for v in node.values))

    def s_expr(self, node, scope):
        value = self.eval(node.expr, scope)
        if self.repl and value is not None:
            self.out(show(value, nested=True))

    def s_if(self, node, scope):
        for cond, body in node.branches:
            if self.eval(cond, scope):
                return self.exec_block(body, scope)
        if node.otherwise:
            self.exec_block(node.otherwise, scope)

    def loop(self, body, scope):
        """Run a loop body once; returns False when the loop should stop."""
        try:
            self.exec_block(body, scope)
        except BreakSignal:
            return False
        except ContinueSignal:
            pass
        return True

    def s_while(self, node, scope):
        while self.eval(node.cond, scope):
            if not self.loop(node.body, scope):
                break

    def s_for(self, node, scope):
        items = self.eval(node.items, scope)
        if isinstance(items, (int, float)):
            raise PaiError("can't loop over a number; use range(n) or 'repeat n times'")
        for item in list(items):
            scope.vars[node.var] = item
            if not self.loop(node.body, scope):
                break

    def s_repeat(self, node, scope):
        count = self.eval(node.count, scope)
        if not isinstance(count, int):
            raise PaiError(f"repeat needs a whole number, not {show(count, True)}")
        for _ in range(count):
            if not self.loop(node.body, scope):
                break

    def s_func(self, node, scope):
        scope.vars[node.name] = Function(node.name, node.params, node.body, scope, self)

    def s_return(self, node, scope):
        raise ReturnSignal(None if node.value is None else self.eval(node.value, scope))

    def s_break(self, node, scope):
        raise BreakSignal()

    def s_continue(self, node, scope):
        raise ContinueSignal()

    def s_pass(self, node, scope):
        pass

    def s_use(self, node, scope):
        if node.is_file:
            other = Interpreter(self.out)
            other.run(read_file(node.target))
            self.globals.vars.update(other.globals.vars)
        else:
            plugin = self.load_plugin(node.target)
            for name in getattr(plugin, "__all__", []):
                self.builtins.vars[name] = getattr(plugin, name)

    def s_ai_block(self, node, scope):
        plugin = self.load_plugin("ai")
        func = getattr(plugin, AI_BLOCKS[node.command])
        try:
            func(node.target, **node.options)
        except TypeError as e:
            if "unexpected keyword" in str(e):
                bad = re.search(r"'(\w+)'", str(e))
                raise PaiError(f"{node.command} has no option '{bad.group(1) if bad else '?'}'") from None
            raise

    def load_plugin(self, name):
        if name not in self.plugins:
            path = HERE / "plugins" / f"{name}.py"
            if not path.exists():
                raise PaiError(f"no plugin named '{name}' (looked for {path})")
            spec = importlib.util.spec_from_file_location(f"pai_plugin_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.plugins[name] = module
        return self.plugins[name]

    # expressions
    def eval(self, node, scope):
        return getattr(self, "e_" + node.kind)(node, scope)

    def e_const(self, node, scope):
        return node.value

    def e_fstring(self, node, scope):
        return "".join(p if isinstance(p, str) else show(self.eval(p, scope)) for p in node.parts)

    def e_name(self, node, scope):
        owner = scope.find(node.name)
        if owner is None:
            hint = PYTHON_HINTS.get(node.name, f"create it with: let {node.name} = ...")
            raise PaiError(f"'{node.name}' doesn't exist ({hint})", node.line)
        return owner.vars[node.name]

    def e_list(self, node, scope):
        return [self.eval(item, scope) for item in node.items]

    def e_map(self, node, scope):
        return {self.eval(k, scope): self.eval(v, scope) for k, v in node.pairs}

    def e_logic(self, node, scope):
        left = self.eval(node.left, scope)
        if node.op == "and":
            return self.eval(node.right, scope) if left else left
        return left if left else self.eval(node.right, scope)

    def e_unary(self, node, scope):
        value = self.eval(node.value, scope)
        return (not value) if node.op == "not" else -value

    def e_binary(self, node, scope):
        return binary(node.op, self.eval(node.left, scope), self.eval(node.right, scope), node.line)

    def e_index(self, node, scope):
        obj = self.eval(node.obj, scope)
        if node.index.kind == "slice":
            part = lambda p: None if p is None else self.eval(p, scope)
            step = part(node.index.step)
            if step == 0:
                raise PaiError("a slice step can't be 0", node.line)
            return obj[slice(part(node.index.start), part(node.index.stop), step)]
        index = self.eval(node.index, scope)
        try:
            return obj[index]
        except IndexError:
            raise PaiError(f"position {index} doesn't exist (this list has {len(obj)} items, "
                           f"numbered 0 to {len(obj) - 1})", node.line) from None
        except KeyError:
            raise PaiError(f"this map has no key {show(index, True)}", node.line) from None

    def e_attr(self, node, scope):
        obj = self.eval(node.obj, scope)
        if isinstance(obj, dict) and node.name in obj:
            return obj[node.name]
        if node.name in ("keys", "values", "items", "length"):
            return {"keys": lambda: list(obj.keys()), "values": lambda: list(obj.values()),
                    "items": lambda: [list(p) for p in obj.items()], "length": lambda: len(obj)}[node.name]
        if node.name == "split" and isinstance(obj, str):
            return lambda sep=None: split_text(obj, sep)
        if node.name in METHODS and hasattr(obj, METHODS[node.name]):
            return getattr(obj, METHODS[node.name])
        hint = " (use .push(x) to add to a list)" if node.name == "append" else ""
        raise PaiError(f"a {type_name(obj)} has no '{node.name}'{hint}", node.line)

    def e_call(self, node, scope):
        func = self.eval(node.func, scope)
        args = [self.eval(a, scope) for a in node.args]
        kwargs = {k: self.eval(v, scope) for k, v in node.kwargs.items()}
        if isinstance(func, Function):
            return self.call_function(func, args, kwargs, node.line)
        if not callable(func):
            raise PaiError(f"{show(func, True)} is a {type_name(func)}, not a function", node.line)
        return func(*args, **kwargs)

    def call_function(self, func, args, kwargs, line=None):
        if len(args) > len(func.params):
            raise PaiError(f"{func.name}() takes {len(func.params)} values but got {len(args)}", line)
        local = Scope(func.closure)
        for i, (pname, default) in enumerate(func.params):
            if i < len(args):
                local.vars[pname] = args[i]
            elif pname in kwargs:
                local.vars[pname] = kwargs.pop(pname)
            elif default is not None:
                local.vars[pname] = self.eval(default, func.closure)
            else:
                raise PaiError(f"{func.name}() is missing a value for '{pname}'", line)
        if kwargs:
            raise PaiError(f"{func.name}() has no parameter '{next(iter(kwargs))}'", line)
        try:
            self.exec_block(func.body, local)
        except ReturnSignal as r:
            return r.value
        return None


def binary(op, a, b, line=None):
    if op == "+":
        if isinstance(a, str) != isinstance(b, str):
            raise PaiError(f"can't add {type_name(a)} and {type_name(b)}; use str(...) to turn a number into text",
                           line)
        return a + b
    if op == "/":
        if b == 0:
            raise PaiError("can't divide by zero", line)
        result = a / b
        return int(result) if isinstance(a, int) and isinstance(b, int) and result.is_integer() else result
    if op in ("//", "%") and b == 0:
        raise PaiError("can't divide by zero", line)
    ops = {"-": lambda: a - b, "*": lambda: a * b, "//": lambda: a // b, "%": lambda: a % b,
           "**": lambda: a ** b, "==": lambda: a == b, "!=": lambda: a != b, "<": lambda: a < b,
           ">": lambda: a > b, "<=": lambda: a <= b, ">=": lambda: a >= b, "in": lambda: a in b,
           "not in": lambda: a not in b}
    return ops[op]()


def friendly(error):
    """Python's error messages, in PAI words."""
    text = str(error)
    for py, pai in TYPE_NAMES.items():
        text = re.sub(rf"\b'?{py}'?\b", pai, text)
    return text


def check(source):
    """Look for problems in PAI code without running it. Returns a message, or None if it looks fine."""
    try:
        body = parse(source)
    except PaiError as e:
        return str(e)
    defined = set(BUILTINS)
    calls, methods = [], []

    def walk(node):
        if isinstance(node, Node):
            if node.kind in ("func", "let"):
                defined.add(node.name)
            if node.kind == "for":
                defined.add(node.var)
            if node.kind == "func":
                defined.update(p for p, _ in node.params)
            if node.kind == "assign" and node.target.kind == "name":
                defined.add(node.target.name)
            if node.kind == "use":
                defined.add("*")
            if node.kind == "call" and node.func.kind == "name":
                calls.append(node.func)
            elif node.kind == "call" and node.func.kind == "attr" and node.func.name not in KNOWN_METHODS:
                methods.append(node.func)
            elif node.kind == "name" and node.name in PYTHON_HINTS:
                calls.append(node)
            for value in node.__dict__.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(body)
    for call in calls:
        if call.name not in defined and "*" not in defined:
            hint = f" ({PYTHON_HINTS[call.name]})" if call.name in PYTHON_HINTS else ""
            return f"Line {call.line}: '{call.name}' is not PAI{hint}"
    for method in methods:
        hint = " (use .push(x) to add to a list)" if method.name in ("append", "unshift") else ""
        return (f"Line {method.line}: .{method.name}() is not a PAI method{hint}. "
                f"PAI methods: {', '.join(sorted(KNOWN_METHODS))}")
    return None


# ---- Running programs -----------------------------------------------------------------

def run_file(path):
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    os.chdir(path.parent)  # file names in a program are relative to the program itself
    interp = Interpreter()
    try:
        interp.run(source)
    except PaiError as e:
        print(f"PAI error in {path.name}, {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


def repl():
    print("PAI interactive mode. Type code and press Enter; blocks end with an empty line. Ctrl+C to quit.")
    interp = Interpreter()
    interp.repl = True
    while True:
        try:
            lines = [input("pai> ")]
            while lines[-1].rstrip().endswith(":") or (len(lines) > 1 and lines[-1].strip()):
                lines.append(input("...  "))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            interp.run("\n".join(lines))
        except PaiError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    run_file(sys.argv[1]) if len(sys.argv) == 2 else repl()
