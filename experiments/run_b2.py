import argparse
import json
import time
from pathlib import Path

from src.expr import nvars, compile_expr
from src.sampling import train_inputs, test_inputs, to_env
from src.methods.blackbox_llm import deobfuscate
from src.metrics import evaluate


def summary(recs, path):
    n = len(recs)
    if not n:
        print("Aucun résultat.")
        return
    succ = [r for r in recs if r["success"]]
    valid = sum(r["valid"] for r in recs)
    ratios = [r["ratio"] for r in succ if r["ratio"] is not None]
    print(f"\nRésultats : {path}")
    print(f"Réussite (exacte sur les entrées de test) : {len(succ)}/{n} = {len(succ)/n:.1%}")
    print(f"Réponses syntaxiquement valides : {valid}/{n}")
    if ratios:
        print(f"Qualité (taille candidat / cible, moyenne sur les réussites) : {sum(ratios)/len(ratios):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="fichier d'expressions, une par ligne (ex : datasets/b2)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--bits", type=int, default=32)
    a = ap.parse_args()

    lines = [l.strip() for l in open(a.file, encoding="utf-8") if l.strip()]
    lines = lines[a.start:a.start + a.limit]

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(a.file).stem}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    recs = []
    with out.open("w", encoding="utf-8") as fh:
        for i, target in enumerate(lines, start=a.start):
            n = nvars(target)
            f = compile_expr(target, n, a.bits)
            pairs = [(row, f(to_env(row, a.bits))) for row in train_inputs(n, a.bits)]

            cand, raw = deobfuscate(n, a.bits, pairs)
            res = evaluate(cand, target, n, a.bits, test_inputs(n, a.bits))

            rec = {"id": i, "target": target, "raw": raw, "cand": cand, **res}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            recs.append(rec)
            print(f"[{i}] succès={res['success']} valide={res['valid']} "
                  f"score={res['correct']}/{res['total']} -> {cand}")

    summary(recs, out)


if __name__ == "__main__":
    main()