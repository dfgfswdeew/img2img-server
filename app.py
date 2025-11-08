import os
import io
import requests
from flask import Flask, request, jsonify

# Твой OpenAI API-ключ хранится в Render → Environment → OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = Flask(__name__)

def _bad_request(msg):
    return jsonify({"error": msg}), 400

def _proxy_error(details, status=502):
    return jsonify({"error": "OpenAI error", "details": details}), status

@app.get("/")
def health():
    return "Proxy is running ✅"

@app.post("/image/generate")
def image_generate():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return _bad_request("Invalid JSON")

    prompt = (data or {}).get("prompt")
    size = (data or {}).get("size", "1024x1024")
    if not prompt:
        return _bad_request("Missing prompt")

    resp = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "User-Agent": "RenderImageProxy/1.0",
        },
        json={
            "model": "dall-e-2",
            "prompt": prompt,
            "size": size,
            "n": 1
        },
        timeout=120
    )

    try:
        payload = resp.json()
    except Exception:
        return _proxy_error(resp.text)

    if not resp.ok:
        return _proxy_error(payload)

    url = (payload.get("data") or [{}])[0].get("url")
    if not url:
        return _proxy_error("No url in response")
    return jsonify({"url": url})

@app.post("/image/edit")
def image_edit():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return _bad_request("Invalid JSON")

    prompt = (data or {}).get("prompt")
    image_url = (data or {}).get("image_url")
    mask_url = (data or {}).get("mask_url")
    size = (data or {}).get("size", "1024x1024")

    if not prompt or not image_url:
        return _bad_request("Missing prompt or image_url")

    # Загружаем изображение
    img_resp = requests.get(image_url, stream=True, timeout=60)
    if not img_resp.ok:
        return _bad_request("Cannot fetch image")
    img_ct = (img_resp.headers.get("Content-Type") or "").lower()
    if not any(t in img_ct for t in ("image/png", "image/jpeg", "image/jpg", "image/webp")):
        return _bad_request(f"Unsupported image content-type: {img_ct}")
    img_bytes = io.BytesIO(img_resp.content)

    # Загружаем маску (если есть)
    mask_tuple = None
    if mask_url:
        m_resp = requests.get(mask_url, stream=True, timeout=60)
        if not m_resp.ok:
            return _bad_request("Cannot fetch mask")
        m_ct = (m_resp.headers.get("Content-Type") or "").lower()
        if "image/png" not in m_ct:
            return _bad_request("Mask must be PNG")
        mask_tuple = ("mask", ("mask.png", m_resp.content, "image/png"))

    # Собираем multipart/form-data
    files = [("image", ("image.png", img_bytes.getvalue(), "image/png"))]
    if mask_tuple:
        files.append(mask_tuple)

    form = {
        "model": "dall-e-2",
        "prompt": prompt,
        "size": size,
        "n": "1"
    }

    resp = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "User-Agent": "RenderImageProxy/1.0",
        },
        data=form,
        files=files,
        timeout=180
    )

    try:
        payload = resp.json()
    except Exception:
        return _proxy_error(resp.text)

    if not resp.ok:
        return _proxy_error(payload)

    url = (payload.get("data") or [{}])[0].get("url")
    if not url:
        return _proxy_error("No url in response")
    return jsonify({"url": url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))

