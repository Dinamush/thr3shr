from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.realism_sources import RealismSample
from app.style_detectors import StylePrediction, BUCKET_ANIME, BUCKET_PHOTO


def test_list_style_detectors_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/debug/style-detectors")
        assert resp.status_code == 200, resp.text
        ids = {d["id"] for d in resp.json()["detectors"]}
        assert "wd_taxonomy" in ids
        assert "imgutils_caformer" in ids


def test_debug_style_eval_endpoint(monkeypatch, tmp_path: Path):
    samples = [
        RealismSample(
            sample_id="photo_1",
            label="photo",
            bucket="photo",
            source="wikimedia_commons",
            file_url="https://example.invalid/p1.jpg",
            title="Portrait",
            query="portrait",
        ),
        RealismSample(
            sample_id="anime_1",
            label="anime",
            bucket="anime",
            source="safebooru",
            file_url="https://example.invalid/a1.jpg",
            title="1",
            query="1girl solo",
        ),
        RealismSample(
            sample_id="edge_1",
            label="edge_realistic",
            bucket="anime",
            source="safebooru",
            file_url="https://example.invalid/e1.jpg",
            title="2",
            query="realistic",
        ),
    ]

    monkeypatch.setattr(
        "app.style_eval.collect_realism_samples",
        lambda **_k: samples,
    )

    def fake_download(sample):
        path = tmp_path / f"{sample.sample_id}.jpg"
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr("app.style_eval.download_realism_sample", fake_download)

    def fake_build(detector_id, **_k):
        def predict(path: Path) -> StylePrediction:
            name = path.stem
            if name.startswith("photo"):
                bucket = BUCKET_PHOTO
                label = "real"
                scores = {"real": 0.99, "anime": 0.01}
            else:
                bucket = BUCKET_ANIME
                label = "anime"
                scores = {"real": 0.02, "anime": 0.98}
            return StylePrediction(
                detector_id=detector_id,
                method=f"fake:{detector_id}",
                label=label,
                bucket=bucket,
                confidence=max(scores.values()),
                scores=scores,
                detail={},
            )

        return predict

    monkeypatch.setattr("app.style_eval.build_detector", fake_build)

    with TestClient(app) as client:
        resp = client.post(
            "/api/debug/style-eval",
            json={
                "count_per_class": 5,
                "detectors": ["imgutils_mobilenet", "imgutils_caformer"],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["best_detector"] in {"imgutils_mobilenet", "imgutils_caformer"}
        assert len(data["reports"]) == 2
        assert data["reports"][0]["count_evaluated"] == 3
        assert data["overall_conclusion"]["decision"] in {"GO", "NO_GO"}
