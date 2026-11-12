import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src.expr import nvars, compile_expr
from src.sampling import train_inputs, test_inputs, to_env
from src.metrics import evaluate
from src.providers import get_provider
from src.methods.blackbox_methods import run_single, run_agent


def summary(recs, path):
    n = len(recs)
    if not n:
        print("Aucun résultat.")
        return

    succ = [r for r in recs if r.get("success")]
    proven = [r for r in recs if r.get("proven")]
    timeouts = [r for r in recs if (r.get("proof") or {}).get("status") == "timeout"]
    cex = [r for r in recs if (r.get("proof") or {}).get("status") == "counterexample"]
    tot_c = sum(r.get("correct", 0) for r in recs)
    tot_t = sum(r.get("total", 0) for r in recs)
    ratios = [r["ratio"] for r in proven if r.get("ratio") is not None]

    print(f"\nRésultats : {path}")
    print(f"Réponses valides (parsing) : {sum(bool(r.get('valid')) for r in recs)}/{n}")
    print(f"Réussite par échantillons : {len(succ)}/{n} = {len(succ)/n:.1%}")
    print(f"Prouvées équivalentes (Z3) : {len(proven)}/{n} = {len(proven)/n:.1%}")
    print(f"Contre-exemples Z3 : {len(cex)} | timeouts Z3 : {len(timeouts)}")
    print(f"Correspondance moyenne sur les tests : {tot_c/tot_t:.1%}" if tot_t else "")
    print(f"Tours moyens : {sum(r.get('rounds', 0) for r in recs)/n:.2f}")
    if ratios:
        print(f"Qualité (taille candidat / cible, sur prouvées) : {sum(ratios)/len(ratios):.2f}")

    reasons = Counter(r.get("reason") or r.get("error") or "-" for r in recs)
    print("Raisons :")
    for reason, count in reasons.most_common():
        print(f"  {count:3d}  {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="fichier d'expressions, une par ligne")
    ap.add_argument("--provider", default="gemini",
                    help="gemini | groq | openrouter | ollama | custom")
    ap.add_argument("--model", default=None, help="nom du modèle (sauf gemini : voir .env)")
    ap.add_argument("--method", default="single", choices=["single", "agent"])
    ap.add_argument("--rounds", type=int, default=3, help="tours max pour agent")
    ap.add_argument("--train", type=int, default=100, help="exemples visibles par le LLM")
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--max-vars", type=int, default=6,
                    help="ne garder que les expressions à au plus N variables")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    ask, tag = get_provider(a.provider, a.model)

    lines = [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    selected = [(i, l) for i, l in enumerate(lines) if nvars(l) <= a.max_vars]
    selected = selected[a.start:a.start + a.limit]

    out = Path("results/raw") / (
        f"{Path(a.file).stem}_{a.provider}_{a.method}_v{a.max_vars}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    recs = []
    with out.open("w", encoding="utf-8") as fh:
        for i, target in selected:
            rec = {"id": i, "target": target, "provider": tag, "method": a.method}
            try:
                n = nvars(target)
                f = compile_expr(target, n, a.bits)
                pairs = [(row, f(to_env(row, a.bits)))
                         for row in train_inputs(n, a.bits, total=a.train)]

                if a.method == "single":
                    cand, rounds, hist, reason = run_single(ask, n, a.bits, pairs)
                else:
                    cand, rounds, hist, reason = run_agent(ask, n, a.bits, pairs, a.rounds)

                res = evaluate(cand, target, n, a.bits, test_inputs(n, a.bits), reason=reason)
                rec.update(res, cand=cand, rounds=rounds, history=hist, reason=reason)
            except Exception as e:
                rec.update(valid=False, success=False, proven=False, correct=0, total=0,
                           ratio=None, proof=None, cand="", rounds=0, history=[],
                           reason="exception", error=f"{type(e).__name__}: {e}")

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            recs.append(rec)
            print(f"[{i}] valide={rec.get('valid')} succès={rec.get('success')} "
                  f"prouvé={rec.get('proven')} score={rec.get('correct')}/{rec.get('total')} "
                  f"tours={rec.get('rounds')} raison={rec.get('reason')} -> {rec.get('cand')}")

    summary(recs, out)


if __name__ == "__main__":
    main()