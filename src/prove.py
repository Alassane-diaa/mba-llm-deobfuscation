import ast

import z3


def _to_z3(node, V, bits):
    if isinstance(node, ast.Expression):
        return _to_z3(node.body, V, bits)
    if isinstance(node, ast.Constant):
        return z3.BitVecVal(node.value & ((1 << bits) - 1), bits)
    if isinstance(node, ast.Name):
        return V[node.id]
    if isinstance(node, ast.UnaryOp):
        x = _to_z3(node.operand, V, bits)
        return ~x if isinstance(node.op, ast.Invert) else -x
    a = _to_z3(node.left, V, bits)
    b = _to_z3(node.right, V, bits)
    op = type(node.op)
    if op is ast.Add:
        return a + b
    if op is ast.Sub:
        return a - b
    if op is ast.Mult:
        return a * b
    if op is ast.BitAnd:
        return a & b
    if op is ast.BitOr:
        return a | b
    if op is ast.BitXor:
        return a ^ b
    if op is ast.LShift:
        return a << b
    if op is ast.RShift:
        return z3.LShR(a, b)  # décalage logique, comme dans notre évaluation
    raise ValueError(f"opérateur non géré : {op.__name__}")


def prove(cand, target, n_vars, bits=32, timeout_ms=10_000):
    """Retourne un dict avec 'status' : proven | counterexample | timeout | error."""
    try:
        tc = ast.parse(cand, mode="eval")
        tt = ast.parse(target, mode="eval")
        V = {f"v{i}": z3.BitVec(f"v{i}", bits) for i in range(n_vars)}
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(_to_z3(tc, V, bits) != _to_z3(tt, V, bits))
        r = s.check()
        if r == z3.unsat:
            return {"status": "proven"}
        if r == z3.sat:
            m = s.model()
            return {"status": "counterexample",
                    "model": {k: str(m.eval(v, model_completion=True))
                              for k, v in V.items()}}
        return {"status": "timeout", "reason": str(s.reason_unknown())}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}