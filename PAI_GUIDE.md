# PAI language guide

PAI is a small Python-like language with built-in AI commands.
Run a program with `python pai.py file.pai`, or `python pai.py` for interactive mode.
Blocks end with `:` and are indented with 4 spaces. Comments start with `--`.

## Basics
```pai
-- variables are created with let, then changed with =
let name = "Ada"
let count = 3
count = count + 1
count += 1

say "Hello", name                 -- say prints values separated by spaces
say f"{name} has {count} apples"  -- f-text puts values inside {}
let answer = ask("Your name? ")   -- read text typed by the user
let n = num("42")                 -- text to number
let t = str(42)                   -- number to text
```
Values: numbers (`3`, `2.5`), text (`"hi"`), `true`, `false`, `none`, lists (`[1, 2, 3]`), maps (`{"a": 1}`).
Operators: `+ - * / // % **`, `== != < > <= >=`, `and or not`, `in`, `not in`.
Text and numbers can't be added directly: use `"Age: " + str(age)` or f-text.

## Decisions and loops
```pai
if score >= 90:
    say "A"
elif score >= 80:
    say "B"
else:
    say "try again"

repeat 3 times:
    say "hi"

for i in range(1, 6):      -- 1, 2, 3, 4, 5
    say i

for fruit in ["apple", "pear"]:
    say fruit

while count > 0:
    count -= 1
    if count == 2: continue
    if count == 1: break
```

## Functions
```pai
func add(a, b):
    return a + b

func greet(who, greeting="Hi"):
    return f"{greeting}, {who}!"

say add(2, 3)
say greet("world", greeting="Hello")
```

## Lists and maps
```pai
let items = [3, 1, 2]
items.push(4)              -- add to the end
items.pop()                -- remove the last item
say items[0], len(items), sorted(items), sum(items)

let person = {"name": "Ada", "age": 36}
person["city"] = "London"
say person["name"], person.age, person.keys()
if "age" in person:
    say "has an age"
```
Slices take part of a list or text: `items[start:end:step]`, every part optional, `end` not included.
```pai
let nums = [10, 20, 30, 40, 50]
say nums[1:3]              -- [20, 30]
say nums[:2], nums[3:]     -- [10, 20] [40, 50]
say nums[-1], nums[-2:]    -- 50 [40, 50]  (negative counts from the end)
say nums[::-1]             -- [50, 40, 30, 20, 10]  (reversed)
say "hello"[::-1]          -- olleh
```
Text methods: `.upper() .lower() .strip() .replace(a, b) .split(sep) .startswith(x) .endswith(x)`.

## Built-in functions
`say` (statement), `ask(prompt)`, `len(x)`, `range(a, b)`, `str(x)`, `num(x)`, `int(x)`, `round(x)`,
`abs(x)`, `min(...)`, `max(...)`, `sum(list)`, `sorted(list)`, `reversed(list)`, `random(a, b)`,
`choice(list)`, `type(x)`, `join(list, sep)`, `split(text, sep)`, `sqrt(x)`, `floor(x)`, `ceil(x)`,
`now()`, `today()`, `wait(seconds)`, `read(file)`, `write(file, text)`, `exists(file)`,
`ai(question)` (asks the local AI and returns its answer as text).

## AI commands
```pai
train "my_model.pt"        -- train a NEW model from scratch
  dataset "data.jsonl"     -- JSONL/JSON/CSV/TXT: Q&A rows or chat "messages"
  steps "2000"             -- how many training steps (or use end for a stop time)
  size "base"              -- "small", "base" or "large"
  tokenizer "bpe"          -- "char" (default) or "bpe"
  vocab "2000"             -- BPE vocabulary size
  lr "0.0006"              -- learning rate (optional)
  env "Local"              -- "Local" or "Colab, GPU"

train "mentor"             -- FINE-TUNE a pretrained model (much smarter, recommended)
  base "qwen2.5-coder:1.5b"  -- the model to improve (or a full Hugging Face id)
  dataset "data.jsonl"       -- chat "messages" data
  method "lora"              -- "lora" (default) or "qlora"
  epochs "3"                 -- passes over the data
  name "Mentor"              -- the assistant's identity, baked in
  creator "Your Name"

run "my_model.pt"          -- chat page; "ollama:qwen2.5-coder:1.5b" uses Ollama
  name "My Model"
  creator "Your Name"
  host "localhost:8080"      -- "localhost:PORT" (this PC) or "lan:PORT" (whole network)

teach "my_model.pt"        -- like run, with a Teach button to correct answers
  name "My Model"
  host localhost:8080

eval "my_model.pt"         -- score a trained model on a test set + show a sample
  on "test.jsonl"
  prompt "Write a function that adds two numbers."
```

### Two kinds of train
- **With `base`** (recommended): `train` fine-tunes that pretrained model with LoRA and
  writes a Colab notebook plus a matching Ollama `Modelfile`. Run the notebook on a free
  T4 GPU, download the `.gguf` it produces, then `ollama create <name> -f <name>.Modelfile`
  and chat with it using `run "ollama:<name>"`. This makes a genuinely capable assistant.
- **Without `base`**: `train` builds a small character-level model from scratch. Good for
  learning and tiny demos, but far weaker than fine-tuning a real base model.

### Datasets
`train` reads one shared format for all data. Any of these work:
- **Chat** (`.jsonl`, recommended): one object per line with a `messages` list ---
  `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
  This is the same format used for fine-tuning, so one dataset serves both.
- **Q&A rows** (`.json`/`.jsonl`/`.csv`): objects or columns with keys like
  `question`/`answer`, `instruction`/`output`, or `prompt`/`completion`.
- **Plain text** (`.txt`): used as-is.

Training needs PyTorch (`pip install torch`); the core language needs nothing but Python.

### Serving a model
`run` and `teach` open a chat page that **streams** replies as they're written. Use
`host "localhost:8080"` for this PC only, or `host "lan:8080"` so others on your network
can use it (no password - only on a network you trust). There's also a JSON API:
`POST /api/chat` with `{"message": "..."}`, and `GET /api` lists the endpoints. In `teach`
mode, each correction backs up the model to `<name>.bak` first, so a bad lesson can be undone.

## Example: number guessing game
```pai
let secret = random(1, 10)
let guess = 0
while guess != secret:
    guess = num(ask("Guess 1-10: "))
    if guess < secret:
        say "Higher!"
    elif guess > secret:
        say "Lower!"
say "You got it!"
```

## Not in PAI (use these instead)
`print(x)` -> `say x` | `def f():` -> `func f():` | `import x` -> `use x` |
`True/False/None` -> `true/false/none` | `x = 5` for a new variable -> `let x = 5` |
`list.append(x)` -> `list.push(x)` | `input()` -> `ask()` | `int("5")` works, `num("5.5")` for decimals
