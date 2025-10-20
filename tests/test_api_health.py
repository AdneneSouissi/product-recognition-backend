import requests

BASE = "http://127.0.0.1:8000"


def test_healthz_ok():
    r = requests.get(f"{BASE}/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_version_ok():
    r = requests.get(f"{BASE}/version")
    assert r.status_code == 200
    assert "version" in r.json()
