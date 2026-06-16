"""run_all.py: job graph construction and scheduling logic (no subprocesses)."""

from seabed_unet.run_all import Job, build_jobs, next_ready

POLYGONS = ["polygon1", "polygon3", "polygon4", "polygon5"]
CONFIGS = ["training/config/experiment_3band.yaml", "training/config/experiment_2band.yaml"]


def test_build_jobs_counts_and_deps():
    jobs = build_jobs(CONFIGS, POLYGONS, None, None, "polygon4")
    by_name = {j.name: j for j in jobs}
    # per config: train + eval + predict + 4 folds + summary = 8
    assert len(jobs) == 16
    assert by_name["experiment_3band:eval"].deps == ["experiment_3band:train"]
    assert by_name["experiment_3band:predict"].deps == ["experiment_3band:train"]
    summary = by_name["experiment_3band:lopo_summary"]
    assert sorted(summary.deps) == sorted(
        f"experiment_3band:fold_{p}" for p in POLYGONS
    )
    # trainings first (lanes grab long jobs first), summaries last
    assert jobs[0].name.endswith(":train")
    assert jobs[-1].name.endswith(":lopo_summary")


def test_build_jobs_smoke_flags_and_no_predict():
    jobs = build_jobs(CONFIGS[:1], POLYGONS, 2, 8, None)
    by_name = {j.name: j for j in jobs}
    assert "experiment_3band:predict" not in by_name
    train = by_name["experiment_3band:train"]
    assert "--epochs" in train.argv and "--limit" in train.argv
    # summary must aggregate, never train
    assert "--summarize-only" in by_name["experiment_3band:lopo_summary"].argv


def test_next_ready_respects_queue_order_and_deps():
    jobs = [
        Job("a", []),
        Job("b", [], deps=["a"]),
        Job("c", []),
    ]
    assert next_ready(jobs).name == "a"   # first pending wins
    jobs[0].status = "running"
    assert next_ready(jobs).name == "c"   # b blocked by running dep
    jobs[0].status = "done"
    assert next_ready(jobs).name == "b"


def test_next_ready_skips_dependents_of_failures():
    jobs = [
        Job("a", []),
        Job("b", [], deps=["a"]),
        Job("c", [], deps=["b"]),
    ]
    jobs[0].status = "failed"
    assert next_ready(jobs) is None
    assert jobs[1].status == "skipped"
    # cascades on the next pass
    assert next_ready(jobs) is None
    assert jobs[2].status == "skipped"
