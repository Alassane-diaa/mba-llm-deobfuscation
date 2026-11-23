import re
import json
import hashlib
from pathlib import Path

from src.expr import compile_expr, nvars
from src.sampling import train_inputs, to_env

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

SYSTEM = SYSTEM_BASE

JSON_OUTPUT_RULE = """
Reply with exactly one JSON object and nothing else:
{"reasoning": "...", "result": "..."}
The result field must contain only the final expression."""

EXPERT_PERSONA = "You are an expert reverse engineer, knowing how to deobfuscate Mixed-Boolean Arithmetic (MBA) expressions."


def build_system(persona="none", output_mode="plain"):
    if persona == "expert":
        first, rest = SYSTEM_BASE.split("\n", 1)
        system = EXPERT_PERSONA + "\n" + rest
    else:
        system = SYSTEM
    if output_mode == "json":
        system += JSON_OUTPUT_RULE
    return system


def signed(value, bits):
    """Affiche un entier masqué sur `bits` bits en décimal signé."""
    mask = (1 << bits) - 1
    v = value & mask
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def demo_examples(n_vars, bits, count, excluded, demo_file="experiments/demo_functions.txt"):
    if count <= 0:
        return []
    examples = []
    excluded_normalized = {x.strip() for x in excluded}
    for line in Path(demo_file).read_text(encoding="utf-8").splitlines():
        expression = line.strip()
        if not expression or expression in excluded_normalized or nvars(expression) != n_vars:
            continue
        f = compile_expr(expression, n_vars, bits)
        seed = int(hashlib.sha256(f"demo|{expression}".encode("utf-8")).hexdigest()[:8], 16)
        demo_pairs = []
        for row in train_inputs(n_vars, bits, total=3, seed=seed):
            demo_pairs.append((row, f(to_env(row, bits))))
        examples.append(demo_pairs)
        if len(examples) == count:
            break
    return examples


def build_prompt(n_vars, bits, pairs, demo_pairs=None):
    lines = [f"Width: {bits} bits",
             "Variables: " + ", ".join(f"v{i}" for i in range(n_vars)),
             "Examples (inputs and outputs as signed decimal):"]
    for row, out in pairs:
        ins = ", ".join(f"v{i}={x}" for i, x in enumerate(row))
        lines.append(f"{ins} -> {signed(out, bits)}")
    for index, examples in enumerate(demo_pairs or [], 1):
        lines.extend(["", f"Demonstration example {index} (a different function):",
                      f"Width: {bits} bits",
                      "Variables: " + ", ".join(f"v{i}" for i in range(n_vars)),
                      "Examples (inputs and outputs as signed decimal):"])
        for row, out in examples:
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


def parse_candidate(text, n_vars, bits, output_mode="plain"):
    """Retourne (expression, raison). La raison vaut 'ok' si une seule expression valide est trouvée."""
    if not text or not text.strip():
        return "", "réponse vide"
    if output_mode == "json":
        try:
            obj = json.loads(text.strip())
        except json.JSONDecodeError as e:
            return "", f"JSON invalide: {e.msg}"
        if not isinstance(obj, dict):
            return "", "JSON invalide: la racine doit être un objet"
        if "result" not in obj or not isinstance(obj["result"], str):
            return "", "JSON invalide: champ 'result' manquant ou non textuel"
        text = obj["result"]
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