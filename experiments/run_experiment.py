import argparse
import hashlib
import json
import time
import uuid
from collections import Counter
from pathlib import Path

from src.expr import nvars, compile_expr
from src.sampling import train_inputs, test_inputs, to_env
from src.metrics import evaluate
from src.providers import get_provider
from src.methods.blackbox_methods import run_single, run_agent
from src.methods.blackbox_llm import build_system, demo_examples


def target_seed(target, salt):
    """Graine propre à chaque cible : deux expressions n'ont pas les mêmes entrées."""
    return int(hashlib.sha256(f"{salt}|{target}".encode("utf-8")).hexdigest()[:8], 16)


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
    if tot_t:
        print(f"Correspondance moyenne sur les tests : {tot_c/tot_t:.1%}")
    print(f"Tours moyens : {sum(r.get('rounds', 0) for r in recs)/n:.2f}")
    print(f"Temps moyen par expression : {sum(r.get('seconds', 0) for r in recs)/n:.2f}s")
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
    ap.add_argument("--output", choices=["plain", "json"], default="plain")
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument("--persona", choices=["none", "expert"], default="none")
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args()
    if a.shots < 0:
        ap.error("--shots doit être positif ou nul")

    cache_namespace = uuid.uuid4().hex
    max_output_tokens = a.max_output_tokens if a.output == "json" else None
    ask, tag = get_provider(a.provider, a.model, output_mode=a.output,
                            max_output_tokens=max_output_tokens, persona=a.persona,
                            shots=a.shots, cache_namespace=cache_namespace)
    system = build_system(persona=a.persona, output_mode=a.output)

    lines = [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    selected = [(i, l) for i, l in enumerate(lines) if nvars(l) <= a.max_vars]
    selected = selected[a.start:a.start + a.limit]

    variant = f"_out-{a.output}_persona-{a.persona}_shots{a.shots}"
    variant += f"_max{a.max_output_tokens}" if a.output == "json" else ""
    out = Path("results/raw") / (
        f"{Path(a.file).stem}_{a.provider}_{a.method}_v{a.max_vars}{variant}_"
        f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    recs = []
    with out.open("w", encoding="utf-8") as fh:
        for i, target in selected:
            rec = {"id": i, "target": target, "provider": tag, "method": a.method}
            seconds = 0
            try:
                n = nvars(target)
                f = compile_expr(target, n, a.bits)
                pairs = [(row, f(to_env(row, a.bits)))
                         for row in train_inputs(n, a.bits, total=a.train,
                                                 seed=target_seed(target, "train"))]
                demos = demo_examples(n, a.bits, a.shots, set(lines))

                if a.method == "single":
                    started = time.time()
                    try:
                        cand, rounds, hist, reason = run_single(
                            ask, n, a.bits, pairs, system=system, output_mode=a.output,
                            demo_pairs=demos)
                    finally:
                        seconds = time.time() - started
                else:
                    started = time.time()
                    try:
                        cand, rounds, hist, reason = run_agent(
                            ask, n, a.bits, pairs, a.rounds,
                            system=system, output_mode=a.output, demo_pairs=demos)
                    finally:
                        seconds = time.time() - started

                test_rows = test_inputs(n, a.bits, seed=target_seed(target, "test"))
                res = evaluate(cand, target, n, a.bits, test_rows, reason=reason)
                rec.update(res, cand=cand, rounds=rounds, history=hist, reason=reason,
                           seconds=seconds)
            except Exception as e:
                rec.update(valid=False, success=False, proven=False, correct=0, total=0,
                           ratio=None, proof=None, cand="", rounds=0, history=[],
                           reason="exception", error=f"{type(e).__name__}: {e}",
                           seconds=seconds)

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            recs.append(rec)
            print(f"[{i}] valide={rec.get('valid')} succès={rec.get('success')} "
                  f"prouvé={rec.get('proven')} score={rec.get('correct')}/{rec.get('total')} "
                  f"tours={rec.get('rounds')} raison={rec.get('reason')} -> {rec.get('cand')}")

    summary(recs, out)


if __name__ == "__main__":
    main()