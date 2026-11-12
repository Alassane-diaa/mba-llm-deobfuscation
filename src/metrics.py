from src.expr import compile_expr, nodes
from src.prove import prove
from src.sampling import to_env


def evaluate(cand, target_src, n_vars, bits, test_rows, reason=None):
    """Mesure un candidat sur les entrées de test, puis prouve l'équivalence avec Z3 si le test passe."""
    out = {"valid": False, "correct": 0, "total": len(test_rows), "success": False,
           "nodes_cand": None, "nodes_target": nodes(target_src), "ratio": None,
           "proof": None, "proven": False, "error": None}

    if not cand:
        out["error"] = reason or "candidat vide"
        return out

    try:
        f_cand = compile_expr(cand, n_vars, bits)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    f_target = compile_expr(target_src, n_vars, bits)
    correct = 0
    for row in test_rows:
        env = to_env(row, bits)
        if f_cand(env) == f_target(env):
            correct += 1

    out.update(valid=True, correct=correct,
               success=correct == len(test_rows), nodes_cand=nodes(cand))
    if out["success"]:
        out["ratio"] = out["nodes_cand"] / out["nodes_target"]
        out["proof"] = prove(cand, target_src, n_vars, bits)
        out["proven"] = out["proof"]["status"] == "proven"
    return out