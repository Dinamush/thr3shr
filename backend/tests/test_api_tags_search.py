"""Tag search should surface taxonomy destinations before raw danbooru tags."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_tags_search_prefers_voyeur_folder_over_voyeurism() -> None:
    client = TestClient(app)
    resp = client.get("/api/tags", params={"query": "voyeur", "limit": 20})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert "Voyeur" in items
    assert items.index("Voyeur") < items.index("voyeurism")


def test_tags_search_includes_nested_voyeur_subfolders() -> None:
    client = TestClient(app)
    resp = client.get("/api/tags", params={"query": "Voyeur/", "limit": 20})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert "Voyeur/panties" in items
    assert "Voyeur/upskirt" in items
