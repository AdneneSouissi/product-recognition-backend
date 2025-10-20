import io
import time
import requests

API = "http://127.0.0.1:8000"

def test_health_and_version():
    r = requests.get(f"{API}/healthz"); r.raise_for_status()
    r = requests.get(f"{API}/version"); r.raise_for_status()

def test_predict_smoke():
    # Tiny 1x1 white JPEG
    from PIL import Image
    img = Image.new('RGB', (1, 1), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    files = { 'file': ('tiny.jpg', buf, 'image/jpeg') }
    r = requests.post(f"{API}/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert 'predictions' in data
