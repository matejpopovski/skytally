#!/usr/bin/env bash
# Regenerate results/ and docs/ from trained checkpoints. Expects runs/p2 and runs/p3 from
#   python -m skytally.supervise --config configs/p2.json   (and configs/p3.json)
# and data/visdrone from scripts/download_visdrone.py. Takes about 30 minutes on an M3 Max.
set -euo pipefail

python scripts/dataset_stats.py > /dev/null
cp runs/p2/metrics.jsonl results/train_p2.jsonl
cp runs/p3/metrics.jsonl results/train_p3.jsonl

for run in p2 p3; do
    python -m skytally.evaluate --checkpoint "runs/$run/last.pt" --long-side 1536 \
        --candidates "runs/cand_${run}_1536.pkl" --output "results/eval_$run.json"
done

python scripts/crosscheck_map.py --candidates runs/cand_p2_1536.pkl
python scripts/sweep_nms.py --candidates runs/cand_p2_1536.pkl
python scripts/operating_point.py --candidates runs/cand_p2_1536.pkl

for run in p2 p3; do
    for side in 960 1280 1536 1920 2400; do
        python -m skytally.evaluate --checkpoint "runs/$run/last.pt" --long-side "$side" \
            --output "results/resolution_${run}_$side.json"
    done
    python scripts/bench_latency.py --checkpoint "runs/$run/last.pt" \
        --output "results/latency_$run.json"
done

python scripts/bench_norm_mps.py

python scripts/eval_tracker_sim.py --operating-point results/operating_point.json \
    --output results/tracker_sim_measured.json
python scripts/eval_tracker_sim.py --output results/tracker_sim_degraded.json
python scripts/run_e2e_sim.py --checkpoint runs/p2/last.pt
python scripts/make_figures.py
