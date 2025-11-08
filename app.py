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
BOTMOTHER_TOKEN = os.getenv("BOTMOTHER_TOKEN")
BASE_URL = os.getenv("BASE_URL")

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
    headers = {"User-Agent": "Mozilla/5.0 (img2img-bot/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    if not ct.startswith("image/"):
        raise ValueError(f"URL не вернул image/*, а {ct!r}")
    return r.content


def normalize_to_rgb_square(raw: bytes, out_size: int = 1024) -> Image.Image:
    """Приводим изображение к RGB, центрируем в квадрат 1024x1024."""
    with Image.open(BytesIO(raw)) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode == "P":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        im.thumbnail((out_size, out_size), Image.LANCZOS)
        canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
        x = (out_size - im.width) // 2
        y = (out_size - im.height) // 2
        canvas.paste(im, (x, y))
        return canvas


def to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_full_transparent_mask(size: tuple[int, int]) -> bytes:
    """Создает полностью прозрачную маску (редактировать всю картинку)."""
    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    buf = BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/img2img")
def img2img(payload: dict, x_api_token: str = Header(None)):
    if BOTMOTHER_TOKEN and x_api_token != BOTMOTHER_TOKEN:
        raise HTTPException(401, "Invalid X-Api-Token")

    prompt = (payload or {}).get("prompt")
    image_url = (payload or {}).get("image_url")
    size = (payload or {}).get("size") or "1024x1024"

    if not prompt or not image_url:
        raise HTTPException(400, "Fields 'prompt' and 'image_url' are required")

    # 1) Скачиваем изображение
    try:
        raw = fetch_image_bytes(image_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to download image: {e}")

    # 2) Приводим к нормальному формату и делаем прозрачную маску
    try:
        base_img = normalize_to_rgb_square(raw, out_size=1024)
        png_bytes = to_png_bytes(base_img)
        mask_bytes = make_full_transparent_mask(base_img.size)
    except Exception as e:
        raise HTTPException(400, f"Failed to prepare image: {e}")

    # 3) Отправляем в OpenAI с маской
    tmp_img = f"tmp_{uuid.uuid4().hex}.png"
    tmp_mask = f"mask_{uuid.uuid4().hex}.png"
    try:
        with open(tmp_img, "wb") as f:
            f.write(png_bytes)
        with open(tmp_mask, "wb") as f:
            f.write(mask_bytes)

        with open(tmp_img, "rb") as fimg, open(tmp_mask, "rb") as fmask:
            result = client.images.edits(
                model="gpt-image-1",
                image=fimg,
                mask=fmask,  # ⚠️ обязательный параметр
                prompt=prompt,
                size=size,
                n=1,
            )
    except Exception as e:
        raise HTTPException(502, f"OpenAI error: {e}")
    finally:
        for p in (tmp_img, tmp_mask):
            if os.path.exists(p):
                os.remove(p)

    # 4) Сохраняем результат
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
