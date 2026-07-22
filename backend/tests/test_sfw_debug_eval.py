from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.sfw_sources import DanbooruSource, SafebooruSource, SfwPost


def test_safebooru_query_forces_safe():
    source = SafebooruSource()
    assert "rating:safe" in source.build_query(["1girl", "solo"])
    assert source.is_sfw_rating("safe")
    assert source.is_sfw_rating("s")
    assert not source.is_sfw_rating("e")


def test_danbooru_query_forces_general_and_caps_tags():
    source = DanbooruSource()
    query = source.build_query(["1girl", "solo", "smile"])
    assert "rating:g" in query
    assert "smile" not in query.split()
    assert source.is_sfw_rating("g")
    assert not source.is_sfw_rating("e")


def test_debug_sfw_sources_lists_plugins():
    with TestClient(app) as client:
        resp = client.get("/api/debug/sfw-sources")
        resp.raise_for_status()
        ids = {row["id"] for row in resp.json()["sources"]}
        assert ids == {"safebooru", "danbooru"}


def test_debug_sfw_eval_requires_tags():
    with TestClient(app) as client:
        resp = client.post(
            "/api/debug/sfw-eval",
            json={"source": "safebooru", "tags": [], "count": 5},
        )
        assert resp.status_code == 400


def test_debug_sfw_eval_recall_and_preview(monkeypatch, tmp_path: Path):
    posts = [
        SfwPost(
            source_id="safebooru",
            post_id="101",
            file_url="https://example.invalid/101.jpg",
            rating="safe",
            tags=["1girl", "solo", "smile"],
        ),
        SfwPost(
            source_id="safebooru",
            post_id="102",
            file_url="https://example.invalid/102.jpg",
            rating="safe",
            tags=["1girl", "solo"],
        ),
    ]

    def fake_fetch(self, tags, limit):
        return posts[:limit]

    def fake_download(post):
        path = tmp_path / f"{post.post_id}.jpg"
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr("app.sfw_sources.SafebooruSource.fetch_posts", fake_fetch)
    monkeypatch.setattr("app.debug_eval.download_post", fake_download)
    monkeypatch.setattr(
        "app.debug_eval.extract_scores",
        lambda *_a, **_k: {"1girl": 0.92, "solo": 0.88, "smile": 0.2},
    )
    monkeypatch.setattr(
        "app.debug_eval._matched_tags_for_settings",
        lambda _settings: {"1girl"},
    )

    with TestClient(app) as client:
        client.put(
            "/api/settings",
            json={
                "root_repo": str(tmp_path / "root"),
                "categories_root": str(tmp_path / "cats"),
                "confidence_threshold": 0.6,
                "default_migrate_mode": "copy",
                "scan_recursive": True,
                "experimental_media_enabled": False,
                "selected_tags": ["1girl"],
                "max_inference_workers": 1,
                "inference_batch_size": 1,
                "force_cpu_inference": False,
                "tagger_model": "wd_swinv2_v3",
                "wd_general_threshold": 0.35,
            },
        ).raise_for_status()

        resp = client.post(
            "/api/debug/sfw-eval",
            json={"source": "safebooru", "tags": ["1girl", "solo"], "count": 5},
        )
        resp.raise_for_status()
        body = resp.json()
        assert body["count_evaluated"] == 2
        assert body["query"].endswith("rating:safe") or "rating:safe" in body["query"]
        recall = {row["tag"]: row for row in body["recall"]}
        assert recall["1girl"]["present_in_posts"] == 2
        assert recall["1girl"]["hits_at_threshold"] == 2
        assert recall["1girl"]["hit_rate"] == 1.0
        assert body["items"][0]["primary_tag"] == "1girl"
        assert body["items"][0]["needs_review"] is False


def test_danbooru_rejects_too_many_content_tags(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.sfw_sources.DanbooruSource.fetch_posts",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/debug/sfw-eval",
            json={
                "source": "danbooru",
                "tags": ["1girl", "solo", "smile"],
                "count": 5,
            },
        )
        assert resp.status_code == 400
        assert "at most 2" in resp.json()["detail"].lower()
