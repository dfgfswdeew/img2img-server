import os
import io
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")  # Nano Banana семейство

app = Flask(__name__)

def bad(msg, code=400):
    return jsonify({"error": msg}), code

def public_base():
    base = (os.getenv("PUBLIC_BASE_URL") or request.host_url).rstrip("/")
    return base

def save_png_and_url(png_bytes):
    name = f"{uuid.uuid4().hex}.png"
    out_dir = "/tmp"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return f"{public_base()}/files/{name}"

@app.get("/")
def root():
    return "Gemini nano-generate proxy ✅"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/files/<path:fname>")
def files(fname):
    return send_from_directory("/tmp", fname, mimetype="image/png")

@app.post("/nano/generate")
def nano_generate():
    if not GEMINI_API_KEY:
        return bad("GEMINI_API_KEY is not set on server", 500)

    # Получаем JSON
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return bad("Invalid JSON")

    prompt = data.get("prompt")
    ref_url = data.get("reference_image_url")

    if not prompt:
        return bad("Missing prompt")

    # Формируем parts
    parts = [{
        "text": (
            "Create a NEW photorealistic image from the instructions below. "
            "Do not copy any reference image exactly.\n\n"
            f"Instructions: {prompt}"
        )
    }]

    # Если есть референс - скачиваем
    if ref_url:
        try:
            r = requests.get(ref_url, timeout=60, stream=True,
                             headers={"User-Agent": "RenderGeminiProxy/1.0"})
            r.raise_for_status()
            img = r.content
            mime = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0]
            if mime not in ("image/png", "image/jpeg", "image/webp"):
                mime = "image/jpeg"

            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(img).decode("utf-8")
                }
            })
        except Exception as e:
            return bad(f"Cannot fetch reference image: {e}")

    # Payload
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "temperature": 0.8
        }
    }

    # Запрос к Gemini
    try:
        g = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180
        )
        body = g.json()

    except Exception as e:
        return bad(f"Gemini network error: {e}", 502)

    if not g.ok:
        return bad({"upstream_error": body}, g.status_code)

    # Парсим картинку
    try:
        parts_out = body["candidates"][0]["content"]["parts"]

        b64 = None

        for p in parts_out:
            # Новый формат
            if "inlineData" in p:
                b64 = p["inlineData"]["data"]
                break
            # Старый формат
            if "inline_data" in p:
                b64 = p["inline_data"]["data"]
                break

        if not b64:
            raise Exception("No image in response")

        img_bytes = base64.b64decode(b64)

    except Exception as e:
        return bad({"error": "No image in response", "raw": body, "exception": str(e)}, 502)

    # Сохраняем файл и отдаем ссылку
    url = save_png_and_url(img_bytes)
    return jsonify({"url": url})
