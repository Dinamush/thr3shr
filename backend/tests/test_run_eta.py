from datetime import datetime, timedelta, timezone

from app.api import estimate_run_eta


def test_eta_none_when_not_running() -> None:
    secs, finish = estimate_run_eta(
        status="completed",
        total_images=100,
        processed_images=50,
        started_at=datetime.now(timezone.utc).isoformat(),
        avg_infer_ms_per_image=100.0,
    )
    assert secs is None
    assert finish is None


def test_eta_from_wall_clock_throughput() -> None:
    now = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)
    started = now - timedelta(seconds=100)
    # 20 images in 100s => 0.2 img/s; 80 remaining => 400s
    secs, finish = estimate_run_eta(
        status="running",
        total_images=100,
        processed_images=20,
        started_at=started.isoformat(),
        avg_infer_ms_per_image=9999.0,  # wall clock should win
        now=now,
    )
    assert secs is not None
    assert abs(secs - 400.0) < 1e-6
    assert finish == (now + timedelta(seconds=400)).isoformat()


def test_eta_falls_back_to_avg_infer_ms() -> None:
    now = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)
    secs, finish = estimate_run_eta(
        status="running",
        total_images=10,
        processed_images=1,  # too few for wall-clock path
        started_at=now.isoformat(),
        avg_infer_ms_per_image=500.0,
        now=now,
    )
    assert secs == 9 * 0.5
    assert finish == (now + timedelta(seconds=4.5)).isoformat()


def test_eta_none_when_no_signal() -> None:
    secs, finish = estimate_run_eta(
        status="running",
        total_images=10,
        processed_images=0,
        started_at=None,
        avg_infer_ms_per_image=None,
    )
    assert secs is None
    assert finish is None
