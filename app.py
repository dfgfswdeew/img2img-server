import os
import io
import requests
from flask import Flask, request, jsonify
from PIL import Image

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = Flask(__name__)

def _bad_request(msg):
    return jsonify({"error": msg}), 400

def _proxy_error(details, status=502):
    return jsonify({"error": "OpenAI error", "details": details}), status

@app.get("/")
def health():
    return "Proxy is running ✅"

# ============================================================
# ================  /image/generate  ==========================
# ============================================================

@app.post("/image/generate")
def image_generate():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return _bad_request("Invalid JSON")

    prompt = (data or {}).get("prompt")
    size = (data or {}).get("size", "1024x1024")

    if not OPENAI_API_KEY:
        return _bad_request("OPENAI_API_KEY is not set on server")
    if not prompt:
        return _bad_request("Missing prompt")

    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "User-Agent": "RenderImageProxy/1.0",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-2",  # DALL·E 2, без верификации
                "prompt": prompt,
                "size": size,
                "n": 1,
            },
            timeout=120,
        )
    except Exception as e:
        return _proxy_error(f"Upstream request failed: {e}")

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

# ============================================================
# ================  /image/edit  ==============================
# ============================================================

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

    if not OPENAI_API_KEY:
        return _bad_request("OPENAI_API_KEY is not set on server")
    if not prompt or not image_url:
        return _bad_request("Missing prompt or image_url")

    # --- 1. Загружаем картинку ---
    try:
        img_resp = requests.get(
            image_url,
            stream=True,
            timeout=60,
            headers={"User-Agent": "RenderImageProxy/1.0 (+contact@example.com)"}
        )
    except Exception as e:
        return _bad_request(f"Cannot fetch image: {e}")

    if not img_resp.ok:
        return _bad_request(f"Cannot fetch image, status={img_resp.status_code}")

    # --- 2. Конвертируем в RGBA (DALL·E 2 требует альфа-канал) ---
    try:
        src = Image.open(io.BytesIO(img_resp.content))
        rgba = src.convert("RGBA")
        buf = io.BytesIO()
        rgba.save(buf, format="PNG")
        buf.seek(0)
    except Exception as e:
        return _bad_request(f"Cannot process image: {e}")

    # --- 3. Маска ---
    mask_tuple = None
    if mask_url:
        try:
            m_resp = requests.get(
                mask_url,
                stream=True,
                timeout=60,
                headers={"User-Agent": "RenderImageProxy/1.0 (+contact@example.com)"}
            )
            if not m_resp.ok:
                return _bad_request(f"Cannot fetch mask, status={m_resp.status_code}")
            mask_tuple = ("mask", ("mask.png", m_resp.content, "image/png"))
        except Exception as e:
            return _bad_request(f"Cannot fetch mask: {e}")
    else:
        # Создаём пустую прозрачную маску на весь кадр
        w, h = rgba.size
        empty = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        m_buf = io.BytesIO()
        empty.save(m_buf, format="PNG")
        m_buf.seek(0)
        mask_tuple = ("mask", ("mask.png", m_buf.getvalue(), "image/png"))

    # --- 4. Готовим multipart для запроса к OpenAI ---
    files = [
        ("image", ("image.png", buf.getvalue(), "image/png")),
        mask_tuple,
    ]

    form = {
        "model": "dall-e-2",
        "prompt": prompt,
        "size": size,
        "n": "1",
    }

    # --- 5. Отправляем запрос к OpenAI ---
    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "User-Agent": "RenderImageProxy/1.0",
            },
            data=form,
            files=files,
            timeout=180,
        )
    except Exception as e:
        return _proxy_error(f"Upstream request failed: {e}")

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

# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
