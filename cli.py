"""PAI command-line entry point.

    pai              start interactive mode
    pai program.pai  run a program
    pai --version    print the version
"""
import sys

from pai import run_file, repl

__version__ = "1.0.0"


def main():
    sys.setrecursionlimit(10000)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = sys.argv[1:]
    if args and args[0] in ("-v", "--version"):
        print(f"PAI {__version__}")
        return
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if len(args) > 1:
        print("usage: pai [program.pai]   (no file = interactive)", file=sys.stderr)
        sys.exit(1)
    run_file(args[0]) if args else repl()


if __name__ == "__main__":
    main()
