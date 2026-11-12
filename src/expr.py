import ast
import re

ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
           ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor,
           ast.Invert, ast.USub)


def nvars(src):
    idx = [int(m) for m in re.findall(r"\bv(\d+)\b", src)]
    return max(idx) + 1 if idx else 0


def nodes(src):
    """Nombre de noeuds de l'AST (opérateurs, variables, constantes), comme dans les papiers."""
    tree = ast.parse(src, mode="eval")
    return sum(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant))
               for n in ast.walk(tree))


def compile_expr(src, n_vars, bits):
    """Retourne f(env) -> entier non signé sur `bits` bits. Lève ValueError si l'expression est invalide."""
    tree = ast.parse(src.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"noeud interdit : {type(node).__name__}")
        if isinstance(node, ast.Name):
            if not re.fullmatch(r"v\d+", node.id):
                raise ValueError(f"nom interdit : {node.id}")
            if int(node.id[1:]) >= n_vars:
                raise ValueError(f"variable hors plage : {node.id}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            raise ValueError("constante non entière")
    code = compile(tree, "<expr>", "eval")
    mask = (1 << bits) - 1

    def f(env):
        return eval(code, {"__builtins__": {}}, env) & mask

    return f