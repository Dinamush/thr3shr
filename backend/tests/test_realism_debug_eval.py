from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.realism_sources import RealismSample


def test_predict_real_life_uses_taxonomy():
    from app.realism_eval import predict_real_life
    from app.taxonomy import reload_taxonomy

    reload_taxonomy()
    is_rl, folder, score, evidence = predict_real_life(
        {"realistic": 0.85, "loli": 0.4},
        selected={"real_life", "loli"},
    )
    assert is_rl is True
    assert folder == "real_life"
    assert score == 0.85
    assert evidence["realistic"] == 0.85

    is_rl, folder, score, _ = predict_real_life(
        {"loli": 0.9, "realistic": 0.01},
        selected={"real_life", "loli"},
    )
    assert is_rl is False
    assert folder == "loli"


def test_debug_realism_eval_endpoint(monkeypatch, tmp_path: Path):
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
        "app.realism_eval.collect_realism_samples",
        lambda **_k: samples,
    )

    def fake_download(sample):
        path = tmp_path / f"{sample.sample_id}.jpg"
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr("app.realism_eval.download_realism_sample", fake_download)

    def fake_scores(path, **_k):
        name = Path(path).stem
        if name.startswith("photo"):
            return {"realistic": 0.9, "photorealistic": 0.7, "1girl": 0.1}
        if name.startswith("edge"):
            return {"realistic": 0.4, "loli": 0.2, "1girl": 0.8}
        return {"loli": 0.85, "1girl": 0.95, "realistic": 0.0}

    monkeypatch.setattr("app.realism_eval.extract_scores", fake_scores)

    with TestClient(app) as client:
        resp = client.post(
            "/api/debug/realism-eval",
            json={"count_per_class": 5, "tagger_model": "wd_eva02_large"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["count_evaluated"] == 3
        assert data["metrics"]["tp"] >= 1
        assert data["conclusion"]["decision"] in {"GO", "NO_GO"}
        assert any(i["bucket"] == "photo" for i in data["items"])
