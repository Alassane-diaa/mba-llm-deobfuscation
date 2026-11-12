from src.expr import compile_expr
from src.sampling import to_env
from src.methods.blackbox_llm import SYSTEM, build_prompt, parse_candidate, signed


def run_single(ask, n, bits, pairs):
    """Protocole de base : un seul prompt, sans feedback.
    Retourne (candidat, tours, historique, raison)."""
    raw = ask(build_prompt(n, bits, pairs), system=SYSTEM)
    cand, reason = parse_candidate(raw, n, bits)
    history = [{"round": 1, "raw": raw, "cand": cand, "reason": reason}]
    return cand, 1, history, reason


def run_agent(ask, n, bits, pairs, max_rounds=3):
    """Boucle : le LLM propose, Python vérifie sur les exemples d'entraînement et renvoie les erreurs.
    Retourne (candidat, tours, historique, raison)."""
    feedback = ""
    history = []
    cand = ""
    reason = "non exécuté"

    for rnd in range(1, max_rounds + 1):
        prompt = build_prompt(n, bits, pairs)
        if feedback:
            prompt += ("\n\nYour previous attempt was rejected.\n" + feedback +
                       "\nPropose a corrected expression.")

        raw = ask(prompt, system=SYSTEM)
        cand, reason = parse_candidate(raw, n, bits)
        entry = {"round": rnd, "raw": raw, "cand": cand, "reason": reason}
        history.append(entry)

        if not cand:
            feedback = (f"Your answer was rejected: {reason}. "
                        "Reply with ONE valid expression on one line, nothing else.")
            continue

        f = compile_expr(cand, n, bits)
        wrong = []
        for row, exp in pairs:
            got = f(to_env(row, bits))
            if got != exp:
                wrong.append((row, exp, got))
        entry["wrong"] = len(wrong)

        if not wrong:
            return cand, rnd, history, "ok"

        examples = "; ".join(
            f"inputs {row} should give {signed(exp, bits)}, it gives {signed(got, bits)}"
            for row, exp, got in wrong[:3])
        feedback = (f"'{cand}' is wrong on {len(wrong)}/{len(pairs)} examples. "
                    f"Examples: {examples}.")

    return cand, max_rounds, history, reason