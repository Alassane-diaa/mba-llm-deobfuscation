import argparse
import hashlib
import json
import re
import time
import uuid
import statistics
from collections import Counter
from collections import defaultdict
from pathlib import Path

from src.expr import nvars, compile_expr
from src.sampling import train_inputs, test_inputs, to_env
from src.metrics import evaluate
from src.providers import get_provider
from src.methods.blackbox_methods import run_single, run_agent
from src.methods.blackbox_llm import (
    build_system,
    demo_examples,
    load_demo_functions,
    select_demo_functions,
)


def parse_vars(text):
    if not text or not text.strip():
        raise argparse.ArgumentTypeError("--vars ne peut pas être vide")

    values = []
    for item in text.split(","):
        item = item.strip()
        if not re.fullmatch(r"[+-]?\d+", item):
            raise argparse.ArgumentTypeError(
                f"valeur invalide dans --vars : {item!r}; entrez des entiers séparés par des virgules"
            )
        try:
            value = int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"valeur invalide dans --vars : {item!r}; entrez des entiers séparés par des virgules"
            )
        if not 1 <= value <= 6:
            raise argparse.ArgumentTypeError(
                f"valeur hors intervalle dans --vars : {value}; les valeurs permises vont de 1 à 6"
            )
        values.append(value)
    return sorted(set(values))


def select_targets(lines, vars_list, start=0, limit=None):
    selected = [(i, target) for i, target in enumerate(lines)
                if nvars(target) in vars_list]
    if limit is None:
        return selected[start:]
    return selected[start:start + limit]


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
    quality_ops = [r["quality_ops"] for r in proven
                   if r.get("quality_ops") is not None]

    print(f"\nRésultats : {path}")
    print(f"Réponses valides (parsing) : {sum(bool(r.get('valid')) for r in recs)}/{n}")
    print(f"Réussite par échantillons : {len(succ)}/{n} = {len(succ)/n:.1%}")
    print(f"Prouvées équivalentes (Z3) : {len(proven)}/{n} = {len(proven)/n:.1%}")
    print(f"Contre-exemples Z3 : {len(cex)} | timeouts Z3 : {len(timeouts)}")
    if tot_t:
        print(f"Correspondance moyenne sur les tests : {tot_c/tot_t:.1%}")
    print(f"Tours moyens : {sum(r.get('rounds', 0) for r in recs)/n:.2f}")
    print(f"Temps moyen par expression : {sum(r.get('seconds', 0) for r in recs)/n:.2f}s")
    if quality_ops:
        mean_quality = sum(quality_ops) / len(quality_ops)
        median_quality = statistics.median(quality_ops)
        print(f"Qualité opérateurs (moyenne, sur {len(quality_ops)} prouvées) : "
              f"{mean_quality:.2f}")
        print(f"Qualité opérateurs (médiane, sur {len(quality_ops)} prouvées) : "
              f"{median_quality:.2f}")

    reasons = Counter(r.get("reason") or r.get("error") or "-" for r in recs)
    print("Raisons :")
    for reason, count in reasons.most_common():
        print(f"  {count:3d}  {reason}")

    grouped = defaultdict(list)
    for rec in recs:
        grouped[nvars(rec["target"])].append(rec)

    headers = ("n", "expressions", "valides", "réussites échantillons",
               "prouvées Z3", "qualité moyenne", "qualité médiane", "temps moyen (s)")
    rows = []
    for var_count in sorted(grouped):
        group = grouped[var_count]
        proven_group = [rec for rec in group if rec.get("proven")]
        qualities = [rec["quality_ops"] for rec in proven_group
                     if rec.get("quality_ops") is not None]
        rows.append((
            str(var_count),
            str(len(group)),
            str(sum(bool(rec.get("valid")) for rec in group)),
            str(sum(bool(rec.get("success")) for rec in group)),
            str(len(proven_group)),
            f"{sum(qualities) / len(qualities):.2f}" if qualities else "n/a",
            f"{statistics.median(qualities):.2f}" if qualities else "n/a",
            f"{sum(rec.get('seconds', 0) for rec in group) / len(group):.2f}",
        ))

    widths = [max(len(header), *(len(row[i]) for row in rows))
              for i, header in enumerate(headers)]

    def format_row(row):
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    print("\nRésumé par nombre de variables")
    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))


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
    ap.add_argument("--vars", type=parse_vars, default=parse_vars("2,3,4,5,6"),
                    help="liste de nombres exacts de variables séparés par des virgules (1 à 6)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--output", choices=["plain", "json"], default="plain")
    ap.add_argument("--max-output-tokens", type=int, default=512)
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args()
    if a.shots < 0:
        ap.error("--shots doit être positif ou nul")

    cache_namespace = uuid.uuid4().hex
    max_output_tokens = a.max_output_tokens if a.output == "json" else None
    ask, tag = get_provider(a.provider, a.model, output_mode=a.output,
                            max_output_tokens=max_output_tokens,
                            shots=a.shots, cache_namespace=cache_namespace)
    system = build_system(output_mode=a.output)

    lines = [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    selected = select_targets(lines, a.vars, start=a.start, limit=a.limit)
    demo_functions = []
    if a.shots:
        demo_functions = load_demo_functions()

    out = Path("results/raw") / (
        f"{Path(a.file).stem}_{a.provider}_{a.method}_vars"
        f"{'-'.join(str(value) for value in a.vars)}_"
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
                demo_functions = select_demo_functions(
                    demo_functions, n, a.shots, target)
                demos = demo_examples(demo_functions, n, a.bits)
                rec["demo_functions"] = demo_functions

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