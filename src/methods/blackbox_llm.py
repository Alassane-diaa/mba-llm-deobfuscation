import re

from src.expr import compile_expr

SYSTEM_BASE = """You are an expert in binary code deobfuscation and mixed Boolean-arithmetic (MBA) expressions.
You receive input/output examples of an unknown function on unsigned bit-vectors.
Find a short expression that computes the same function.
Rules:
- Use only the variables given, written v0, v1, ...
- Allowed operators: + - * & | ^ ~ , integer literals (decimal), parentheses.
- Inputs and outputs are shown as signed decimal integers, but they represent unsigned bit-vectors of the width given in the prompt.
- Every arithmetic operation is modulo 2^width.
- Bitwise operators act on the width-bit values.
- The expression must match all examples.
- OUTPUT ONLY THE EXPRESSION ON ONE LINE, WITHOUT LABELS, QUOTES, MARKDOWN, OR EXPLANATION."""

ONE_SHOT = """

Worked example (a different function):
Width: 32 bits
Variables: v0, v1
Examples (inputs and outputs as signed decimal):
v0=3, v1=5 -> 8
v0=-7, v1=2 -> -5
Expected output:
v0 + v1"""

SYSTEM = SYSTEM_BASE + ONE_SHOT


def signed(value, bits):
    """Affiche un entier masqué sur `bits` bits en décimal signé."""
    mask = (1 << bits) - 1
    v = value & mask
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def build_prompt(n_vars, bits, pairs):
    lines = [f"Width: {bits} bits",
             "Variables: " + ", ".join(f"v{i}" for i in range(n_vars)),
             "Examples (inputs and outputs as signed decimal):"]
    for row, out in pairs:
        ins = ", ".join(f"v{i}={x}" for i, x in enumerate(row))
        lines.append(f"{ins} -> {signed(out, bits)}")
    lines.append("Expected output:")
    return "\n".join(lines)


FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z _-]*?)\s*[:=]\s*(.+)$")
VAR = re.compile(r"^v\d+$")


def _valid(expr, n_vars, bits):
    """Compile, respecte la grammaire, et s'évalue sur des entrées nulles."""
    try:
        f = compile_expr(expr, n_vars, bits)
        f({f"v{i}": 0 for i in range(n_vars)})
        return True
    except Exception:
        return False


def _candidates(text):
    """Lignes candidates : contenu des blocs de code si présents, sinon texte brut."""
    blocks = FENCE.findall(text)
    source = "\n".join(blocks) if blocks else text
    for raw in source.splitlines():
        line = raw.strip().strip("`").strip().rstrip(";").strip()
        if not line:
            continue
        if line.startswith("return "):
            line = line[len("return "):].strip()
        m = PREFIX.match(line)
        if m and not VAR.match(m.group(1).strip()):
            line = m.group(2).strip()
        yield line


def parse_candidate(text, n_vars, bits):
    """Retourne (expression, raison). La raison vaut 'ok' si une seule expression valide est trouvée."""
    if not text or not text.strip():
        return "", "réponse vide"
    found = [line for line in _candidates(text) if _valid(line, n_vars, bits)]
    unique = list(dict.fromkeys(found))
    if not unique:
        return "", "aucune expression valide trouvée"
    if len(unique) > 1:
        return "", f"{len(unique)} expressions valides différentes : rejet"
    return unique[0], "ok"


def clean(text, n_vars, bits):
    """Compatibilité : retourne seulement l'expression."""
    return parse_candidate(text, n_vars, bits)[0]