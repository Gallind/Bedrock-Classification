"""Full experiment round at maximum hardware utilization.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.run_all \
      --configs training/config/experiment_3band.yaml training/config/experiment_2band.yaml

Schedules every job of a round — per experiment: blocks training + eval (+
polygon4 prediction map) and all LOPO folds + summary — onto parallel "lanes":
one lane owns the GPU (mps), the others run on disjoint CPU-core budgets
(OMP/MKL thread caps), so the GPU never idles while CPU cores sit unused.
Lanes pull jobs greedily from a dependency-aware queue: long training jobs are
queued first, evals/summaries unlock as their parents finish.

All jobs are independent OS processes; per-job console output goes to
<runs_dir>/_runner/<job>.log (each training also keeps its own train.log).
A failed job skips its dependents, the rest of the round continues.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_config
from .crossval import FOLD_ORDER, threads_per_job
from .logging_utils import add_file_handler, setup_logging

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


@dataclass
class Job:
    name: str
    argv: list[str]
    deps: list[str] = field(default_factory=list)
    takes_device: bool = False   # append --device <lane device> at launch
    status: str = "pending"      # pending -> running -> done | failed | skipped


def build_jobs(
    configs: list[str], polygons: list[str],
    epochs: int | None, limit: int | None, predict_polygon: str | None,
) -> list[Job]:
    """The round's job graph, longest jobs first (greedy lanes finish sooner)."""
    py = [sys.executable, "-m"]
    smoke = ([] if epochs is None else ["--epochs", str(epochs)]) + (
        [] if limit is None else ["--limit", str(limit)]
    )
    trains, folds, posts = [], [], []
    for config in configs:
        tag = Path(config).stem
        trains.append(Job(
            f"{tag}:train", py + ["seabed_unet.train", "--config", config] + smoke,
            takes_device=True,
        ))
        posts.append(Job(
            f"{tag}:eval", py + ["seabed_unet.evaluate", "--config", config],
            deps=[f"{tag}:train"], takes_device=True,
        ))
        if predict_polygon:
            posts.append(Job(
                f"{tag}:predict",
                py + ["seabed_unet.predict", "--config", config,
                      "--polygon", predict_polygon],
                deps=[f"{tag}:train"], takes_device=True,
            ))
        fold_names = []
        for poly in polygons:
            name = f"{tag}:fold_{poly}"
            fold_names.append(name)
            folds.append(Job(
                name,
                py + ["seabed_unet.crossval", "--config", config, "--fold", poly] + smoke,
                takes_device=True,
            ))
        posts.append(Job(
            f"{tag}:lopo_summary",
            py + ["seabed_unet.crossval", "--config", config, "--summarize-only"],
            deps=fold_names,
        ))
    return trains + folds + posts


def next_ready(jobs: list[Job]) -> Job | None:
    """First pending job whose deps are all done (queue order = priority)."""
    by_name = {j.name: j for j in jobs}
    for job in jobs:
        if job.status != "pending":
            continue
        dep_status = [by_name[d].status for d in job.deps]
        if all(s == "done" for s in dep_status):
            return job
        if any(s in ("failed", "skipped") for s in dep_status):
            job.status = "skipped"
            logger.warning(f"    {job.name}: skipped (failed dependency)")
    return None


def _all_terminal(jobs: list[Job]) -> bool:
    return all(j.status in ("done", "failed", "skipped") for j in jobs)


def run_lanes(jobs: list[Job], lanes: list[str], log_dir: Path) -> bool:
    """Run the job graph on the given device lanes; True iff everything succeeded."""
    n_cpu_lanes = sum(1 for d in lanes if d == "cpu") or 1
    cpu_threads = threads_per_job(n_cpu_lanes)
    lock = threading.Lock()
    log_dir.mkdir(parents=True, exist_ok=True)

    def lane_worker(device: str) -> None:
        while True:
            with lock:
                if _all_terminal(jobs):
                    return
                job = next_ready(jobs)
                if job is not None:
                    job.status = "running"
            if job is None:
                time.sleep(1.0)
                continue
            argv = job.argv + (["--device", device] if job.takes_device else [])
            env = dict(os.environ)
            if device == "cpu":
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(cpu_threads)
            t0 = time.monotonic()
            logger.info(f"[>] {job.name} on {device}"
                        + (f" ({cpu_threads} threads)" if device == "cpu" else ""))
            with open(log_dir / f"{job.name.replace(':', '_')}.log", "w") as out:
                code = subprocess.run(
                    argv, stdout=out, stderr=subprocess.STDOUT, env=env
                ).returncode
            minutes = (time.monotonic() - t0) / 60
            with lock:
                job.status = "done" if code == 0 else "failed"
            level = logger.info if code == 0 else logger.error
            level(f"[{'+' if code == 0 else '!'}] {job.name}: "
                  f"{'done' if code == 0 else f'FAILED (exit {code})'} in {minutes:.1f} min")

    threads = [threading.Thread(target=lane_worker, args=(d,), daemon=True) for d in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return all(j.status == "done" for j in jobs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a full experiment round across GPU + CPU lanes."
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["training/config/experiment_3band.yaml",
                 "training/config/experiment_2band.yaml"],
    )
    parser.add_argument(
        "--lanes", default="mps,cpu,cpu",
        help="Comma list of worker devices. One mps lane keeps the GPU saturated; "
             "each cpu lane gets an equal share of the physical cores.",
    )
    parser.add_argument("--predict-polygon", default="polygon4",
                        help="Polygon to map after each blocks training ('' to skip).")
    parser.add_argument("--epochs", type=int, default=None, help="Smoke override.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke override.")
    args = parser.parse_args(argv)

    cfg = load_config(args.configs[0])
    polygons = cfg.split.polygons or FOLD_ORDER
    runs_dir = cfg.base_dir / cfg.runs_dir
    setup_logging()
    add_file_handler(runs_dir / "run_all.log")

    lanes = [d.strip() for d in args.lanes.split(",") if d.strip()]
    jobs = build_jobs(args.configs, polygons, args.epochs, args.limit,
                      args.predict_polygon or None)
    logger.info(f"[+] {len(jobs)} jobs on lanes {lanes} "
                f"({len(args.configs)} experiment(s), {len(polygons)} LOPO folds each)")

    t0 = time.monotonic()
    ok = run_lanes(jobs, lanes, runs_dir / "_runner")
    minutes = (time.monotonic() - t0) / 60
    status = {s: sum(1 for j in jobs if j.status == s)
              for s in ("done", "failed", "skipped")}
    logger.info(f"[+] round finished in {minutes:.1f} min — {status}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
