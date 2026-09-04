#!/usr/bin/env python3
"""
Score the predicted counts (runs/notebook_test/SUMMARY.csv) against
manually-provided per-video ground truth (runs/notebook_test/GROUND_TRUTH.csv).

Fill in gt_herring / gt_non_herring in GROUND_TRUTH.csv first, then run:
    ./yolo-eval/bin/python hpc/score_counts.py

Outputs a per-(category, model) error table and writes SCORED.csv.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "runs", "notebook_test")
PRED = os.path.join(OUT_DIR, "SUMMARY.csv")
GT = os.path.join(OUT_DIR, "GROUND_TRUTH.csv")
SCORED = os.path.join(OUT_DIR, "SCORED.csv")


def load_gt():
    gt = {}
    with open(GT) as f:
        for r in csv.DictReader(f):
            h, n = r["gt_herring"].strip(), r["gt_non_herring"].strip()
            if h == "" or n == "":
                continue  # not filled in yet
            gt[r["category"]] = (int(h), int(n))
    return gt


def main():
    gt = load_gt()
    if not gt:
        print("No ground truth filled in yet.")
        print(f"Edit {GT} — set gt_herring and gt_non_herring for each video, then re-run.")
        return

    rows = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            cat = r["category"]
            if cat not in gt:
                continue
            try:
                ph, pn = int(r["herring"]), int(r["non_herring"])
            except ValueError:
                continue  # ERROR/MISSING rows
            gh, gn = gt[cat]
            rows.append({
                "category": cat, "model": r["model"],
                "pred_herring": ph, "gt_herring": gh, "err_herring": ph - gh,
                "pred_non_herring": pn, "gt_non_herring": gn, "err_non_herring": pn - gn,
                "abs_total_err": abs(ph - gh) + abs(pn - gn),
            })

    if not rows:
        print("Ground truth found but no matching predictions to score.")
        return

    # print table
    print("=" * 92)
    print("COUNT SCORING  (err = predicted - ground_truth; negative = under-count)")
    print("=" * 92)
    hdr = (f"{'category':<18}{'model':<10}"
           f"{'H pred/gt':>12}{'H err':>7}{'NH pred/gt':>13}{'NH err':>8}{'|err|':>7}")
    print(hdr)
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (x["category"], x["abs_total_err"])):
        h_ratio = f"{r['pred_herring']}/{r['gt_herring']}"
        nh_ratio = f"{r['pred_non_herring']}/{r['gt_non_herring']}"
        print(f"{r['category']:<18}{r['model']:<10}"
              f"{h_ratio:>12}{r['err_herring']:>+7}"
              f"{nh_ratio:>13}{r['err_non_herring']:>+8}"
              f"{r['abs_total_err']:>7}")

    # best model per category
    print("-" * 92)
    print("Best model per category (lowest total abs error):")
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in by_cat.items():
        best = min(rs, key=lambda x: x["abs_total_err"])
        print(f"  {cat:<18} -> {best['model']}  (|err|={best['abs_total_err']})")

    with open(SCORED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {SCORED}")


if __name__ == "__main__":
    main()
