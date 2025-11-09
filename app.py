import os, io, uuid, base64, requests
from flask import Flask, request, jsonify, send_from_directory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash-image"  # aka Nano Banana

app = Flask(__name__)

def bad(msg, code=400): return jsonify({"error": msg}), code

def save_png_and_url(png_bytes):
    fname = f"{uuid.uuid4().hex}.png"
    out_dir = "/tmp"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as f:
        f.write(png_bytes)
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    return f"{base}/files/{fname}" if base else f"/files/{fname}"

@app.get("/")
def health():
    return "Gemini image edit proxy ✅"

@app.get("/files/<path:fname>")
def files(fname):
    return send_from_directory("/tmp", fname, mimetype="image/png")

@app.post("/nano/edit")
def nano_edit():
    if not GEMINI_API_KEY:
        return bad("GEMINI_API_KEY is not set on server", 500)

    try:
        data = request.get_json(force=True)
    except Exception:
        return bad("Invalid JSON")

    image_url = (data or {}).get("image_url")
    prompt    = (data or {}).get("prompt") or "Edit the image as requested."
    if not image_url:
        return bad("Missing image_url")

    # 1) Скачиваем изображение пользователя
    try:
        r = requests.get(image_url, timeout=60, stream=True,
                         headers={"User-Agent": "RenderGeminiProxy/1.0"})
        r.raise_for_status()
    except Exception as e:
        return bad(f"Cannot fetch image: {e}")

    img_bytes = r.content
    mime = (r.headers.get("Content-Type") or "image/png").split(";")[0]
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        mime = "image/png"

    # 2) Формируем запрос к Gemini (inline base64)
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(img_bytes).decode("utf-8")
                }}
            ]
        }],
        "generationConfig": { "responseModalities": ["IMAGE"] }
    }

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
    except Exception as e:
        return bad(f"Gemini network error: {e}", 502)

    # 3) Разбираем ответ (ищем inline base64 с картинкой)
    try:
        body = g.json()
    except Exception:
        return bad(f"Gemini non-JSON: {g.text}", 502)

    if not g.ok:
        return bad({"upstream_error": body}, g.status_code)

    try:
        parts = (body["candidates"][0]["content"]["parts"])
        b64 = next(p["inline_data"]["data"] for p in parts if "inline_data" in p)
        png = base64.b64decode(b64)
    except Exception:
        return bad({"error": "No image in response", "raw": body}, 502)

    # 4) Сохраняем и возвращаем URL
    url = save_png_and_url(png)
    return jsonify({"url": url})
