import os
import uuid
import base64
import requests
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOTMOTHER_TOKEN = os.getenv("BOTMOTHER_TOKEN")      # ваш секрет для X-Api-Token
BASE_URL = os.getenv("BASE_URL")                    # https://img2img-server.onrender.com

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

os.makedirs("files", exist_ok=True)
app.mount("/files", StaticFiles(directory="files"), name="files")

@app.get("/test")
def test():
    return {"status": "ok"}

def fetch_image_bytes(url: str) -> bytes:
    """Скачиваем картинку как байты с корректным User-Agent и проверкой content-type."""
    headers = {"User-Agent": "Mozilla/5.0 (img2img-bot/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if not ct.startswith("image/"):
        # для отладки вернём первые символы
        raise ValueError(f"URL не вернул image/*, а {ct!r} (len={len(r.content)})")
    return r.content

def normalize_to_rgb_square_png(raw: bytes, out_size: int = 1024) -> bytes:
    """
    Приводим изображение к RGB (без альфы), вписываем/центрируем в квадрат out_size x out_size
    и сохраняем в PNG. Возвращаем PNG-байты.
    """
    with Image.open(BytesIO(raw)) as im:
        # Конвертация альфы → белый фон, палитры → RGB
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode == "P":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        # Вписываем в квадрат с сохранением пропорций
        im.thumbnail((out_size, out_size), Image.LANCZOS)
        canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
        x = (out_size - im.width) // 2
        y = (out_size - im.height) // 2
        canvas.paste(im, (x, y))

        buf = BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

@app.post("/img2img")
def img2img(payload: dict, x_api_token: str = Header(None)):
    # Проверка токена (если задан)
    if BOTMOTHER_TOKEN and x_api_token != BOTMOTHER_TOKEN:
        raise HTTPException(401, "Invalid X-Api-Token")

    prompt = (payload or {}).get("prompt")
    image_url = (payload or {}).get("image_url")
    size = (payload or {}).get("size") or "1024x1024"

    if not prompt or not image_url:
        raise HTTPException(400, "Fields 'prompt' and 'image_url' are required")

    # 1) скачиваем
    try:
        raw = fetch_image_bytes(image_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download image: {e}")

    # 2) нормализуем к PNG RGB 1024x1024
    try:
        png_bytes = normalize_to_rgb_square_png(raw, out_size=1024)
    except Exception as e:
        raise HTTPException(400, f"Failed to prepare image: {e}")

    # 3) отправляем в OpenAI
    tmp_name = f"tmp_{uuid.uuid4().hex}.png"
    try:
        with open(tmp_name, "wb") as f:
            f.write(png_bytes)

        with open(tmp_name, "rb") as f:
            result = client.images.edits(
                model="gpt-image-1",
                image=f,              # одно изображение
                prompt=prompt,
                size=size,            # '1024x1024'
                n=1,
            )
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 4) сохраняем ответ и отдаём URL
    try:
        b64img = result.data[0].b64_json
        out_bytes = base64.b64decode(b64img)
        out_name = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join("files", out_name)
        with open(out_path, "wb") as f:
            f.write(out_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to save image: {e}")

    public_url = f"{BASE_URL}/files/{out_name}" if BASE_URL else f"/files/{out_name}"
    return {"ok": True, "image_url": public_url}
