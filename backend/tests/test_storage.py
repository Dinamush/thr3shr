from app.storage import from_json


def test_from_json_fallback_on_invalid_json() -> None:
    assert from_json("{invalid json") == []
    assert from_json("{invalid json", default={}) == {}
