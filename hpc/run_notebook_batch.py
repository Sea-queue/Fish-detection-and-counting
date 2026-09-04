#!/usr/bin/env python3
"""
Batch-run the parameterized yolo11_prediction_final notebook over the test
dataset: 4 categories (herring easy/hard, non-herring easy/hard) x 3 models.

- Reuses the notebook's exact counting logic (extracted verbatim from the cell).
- Only models that output the 'Herring'/'Non-Herring' class names are used
  (train110/111/113); train49 & train95 use different names and would count 0.
- Writes an annotated .mp4 + per-detection .csv per run under runs/notebook_test/,
  and prints a summary table of final Herring / Non-Herring counts.

Run from the project root with the project venv:
    ./yolo-eval/bin/python hpc/run_notebook_batch.py
"""
import csv
import io
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

NOTEBOOK = "hpc/yolo11_prediction_final.ipynb"
OUT_DIR = "runs/notebook_test"
os.makedirs(OUT_DIR, exist_ok=True)

# category -> test video (per user-confirmed mapping)
VIDEOS = {
    "herring_easy":     "test_scripts/test-videos/nemasket_normal 2.mp4",
    "herring_hard":     "test_scripts/test-videos/nemasket_exta_huge 4.mp4",
    "non_herring_easy": "test_scripts/test-videos/non-herring-easy.mp4",
    "non_herring_hard": "test_scripts/test-videos/count5.mp4",
}

# only models that natively output Herring / Non-Herring names
MODELS = {
    "train110": "model/train110.pt",
    "train111": "model/train111.pt",
    "train113": "model/train113.pt",
}


def load_notebook_code():
    """Extract the notebook's single code cell as an executable string,
    with the trailing parameter/driver block stripped so we only get the
    function definitions (get_zone, classify_track, fish_counting_driver...)."""
    nb = json.load(open(NOTEBOOK))
    src = "".join(nb["cells"][0]["source"])
    marker = "# --- Parameters"
    if marker in src:
        src = src[: src.index(marker)]
    return src


def build_namespace():
    """Exec the notebook's function definitions into a fresh namespace."""
    ns = {}
    exec(load_notebook_code(), ns)
    assert "fish_counting_driver" in ns, "fish_counting_driver not found in notebook"
    return ns


def summarize_csv(csv_path):
    """Final Herring / Non-Herring counts = distinct track_ids whose status
    became 'confirmed', voted by majority class over the track (mirrors the
    notebook's classwise_track_ids logic)."""
    if not os.path.exists(csv_path):
        return {"Herring": 0, "Non-Herring": 0, "confirmed_tracks": 0, "total_tracks": 0}
    class_hist = {}
    confirmed = set()
    all_tracks = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tid = row["track_id"]
            all_tracks.add(tid)
            class_hist.setdefault(tid, []).append(row["class_name"])
            if row["status"] == "confirmed":
                confirmed.add(tid)
    counts = Counter()
    for tid in confirmed:
        voted = Counter(class_hist[tid]).most_common(1)[0][0]
        counts[voted] += 1
    return {
        "Herring": counts.get("Herring", 0),
        "Non-Herring": counts.get("Non-Herring", 0),
        "confirmed_tracks": len(confirmed),
        "total_tracks": len(all_tracks),
    }


def main():
    from ultralytics import YOLO

    ns = build_namespace()
    driver = ns["fish_counting_driver"]

    summary = []
    combos = [(c, v, m, mp) for c, v in VIDEOS.items() for m, mp in MODELS.items()]
    print(f"Running {len(combos)} combinations "
          f"({len(VIDEOS)} videos x {len(MODELS)} models)\n")

    for i, (cat, video, model_name, model_path) in enumerate(combos, 1):
        prefix = os.path.join(OUT_DIR, f"{cat}__{model_name}")
        out_video = f"{prefix}.mp4"
        out_csv = f"{prefix}.csv"
        print(f"[{i}/{len(combos)}] {cat} | {model_name}")
        print(f"    video: {video}")

        if not os.path.exists(video):
            print(f"    !! MISSING VIDEO, skipping")
            summary.append((cat, model_name, "MISSING_VIDEO", "-", "-", "-"))
            continue

        try:
            model = YOLO(model_path)
            driver(model, video, out_video, out_csv)
            s = summarize_csv(out_csv)
            print(f"    -> Herring={s['Herring']}  Non-Herring={s['Non-Herring']}  "
                  f"(confirmed {s['confirmed_tracks']}/{s['total_tracks']} tracks)\n")
            summary.append((cat, model_name, s["Herring"], s["Non-Herring"],
                            s["confirmed_tracks"], s["total_tracks"]))
        except Exception as e:
            print(f"    !! ERROR: {e}\n")
            summary.append((cat, model_name, "ERROR", str(e)[:40], "-", "-"))

    # summary table
    print("\n" + "=" * 78)
    print("SUMMARY  (final counts = distinct confirmed tracks, majority-vote class)")
    print("=" * 78)
    hdr = f"{'category':<18}{'model':<10}{'Herring':>9}{'Non-Herring':>13}{'confirmed':>11}{'tracks':>9}"
    print(hdr)
    print("-" * 78)
    for row in summary:
        cat, model_name, herr, nonherr, conf, tot = row
        print(f"{cat:<18}{model_name:<10}{str(herr):>9}{str(nonherr):>13}"
              f"{str(conf):>11}{str(tot):>9}")

    # also dump summary csv
    summ_csv = os.path.join(OUT_DIR, "SUMMARY.csv")
    with open(summ_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "model", "herring", "non_herring",
                    "confirmed_tracks", "total_tracks"])
        w.writerows(summary)
    print(f"\nSummary written to {summ_csv}")
    print(f"Per-run videos + CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
