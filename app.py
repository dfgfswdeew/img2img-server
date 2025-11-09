import os, uuid, base64, requests, json
from flask import Flask, request, jsonify, send_from_directory, Response

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")
BUILD_ID = os.getenv("BUILD_ID", "local")  # <- добавили

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
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(png_bytes)
    return f"{public_base()}/files/{name}"

@app.get("/")
def root():
    return f"Gemini nano-generate proxy ✅ build={BUILD_ID}"

@app.get("/version")
def version():
    return {"build": BUILD_ID, "model": MODEL}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/files/<path:fname>")
def files(fname):
    return send_from_directory("/tmp", fname, mimetype="image/png")

def _call_gemini(prompt: str, ref_url: str | None):
    parts = [{
        "text": (
            "Create a NEW photorealistic image from the instructions below. "
            "Do not copy any single reference image exactly.\n\n"
            f"Instructions: {prompt}"
        )
    }]
    if ref_url:
        r = requests.get(ref_url, timeout=60, stream=True,
                         headers={"User-Agent": "RenderGeminiProxy/1.0"})
        r.raise_for_status()
        mime = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0]
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            mime = "image/jpeg"
        parts.append({"inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(r.content).decode("utf-8")
        }})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"], "temperature": 0.8}
    }
    g = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=payload, timeout=180
    )
    body = g.json()
    if not g.ok:
        raise RuntimeError(json.dumps({"upstream_error": body}))
    try:
        parts_out = body["candidates"][0]["content"]["parts"]
        b64 = next(p["inline_data"]["data"] for p in parts_out if "inline_data" in p)
        return base64.b64decode(b64)
    except Exception:
        raise RuntimeError(json.dumps({"error": "No image in response", "raw": body}))

def _handle_generate(req_json):
    if not GEMINI_API_KEY:
        return bad("GEMINI_API_KEY is not set on server", 500)
    prompt = (req_json or {}).get("prompt")
    ref_url = (req_json or {}).get("reference_image_url")
    if not prompt:
        return bad("Missing prompt")
    img = _call_gemini(prompt, ref_url)
    url = save_png_and_url(img)
    return url

# Старый эндпоинт (оставляем, но он тоже возвращает только URL)
@app.post("/nano/generate")
def nano_generate():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return bad("Invalid JSON")
    url = _handle_generate(data)
    return jsonify({"url": url})

# Новый «железобетонный» эндпоинт только с URL
@app.post("/nano/generate_url")
def nano_generate_url():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return bad("Invalid JSON")
    url = _handle_generate(data)
    # Возвращаем вручную, чтобы исключить любые сторонние сериализации
    return Response(json.dumps({"url": url}), mimetype="application/json")
