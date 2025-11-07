import os
import uuid
import base64
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Читаем переменные окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOTMOTHER_TOKEN = os.getenv("BOTMOTHER_TOKEN")  # можно оставить пустым, тогда проверка заголовка отключится
BASE_URL = os.getenv("BASE_URL")  # пример: https://img2img-server.onrender.com

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в переменных окружения")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# Папка для сохранения готовых изображений
os.makedirs("files", exist_ok=True)
app.mount("/files", StaticFiles(directory="files"), name="files")

@app.get("/test")
def test():
    return {"status": "ok"}

@app.post("/img2img")
def img2img(payload: dict, x_api_token: str = Header(None)):
    # === Проверка секретного токена (если задан) ===
    if BOTMOTHER_TOKEN and x_api_token != BOTMOTHER_TOKEN:
        raise HTTPException(401, "Invalid X-Api-Token")

    prompt = payload.get("prompt")
    image_url = payload.get("image_url")
    size = payload.get("size", "1024x1024")

    if not prompt or not image_url:
        raise HTTPException(400, "Fields 'prompt' and 'image_url' are required")

    # === Скачиваем картинку пользователя ===
    try:
        resp = requests.get(image_url, timeout=25)
        resp.raise_for_status()
        img_bytes = resp.content
    except Exception as e:
        raise HTTPException(400, f"Failed to download image: {e}")

    tmp_name = f"tmp_{uuid.uuid4().hex}.png"
    with open(tmp_name, "wb") as f:
        f.write(img_bytes)

    # === Отправляем картинку + промт в OpenAI ===
    try:
        with open(tmp_name, "rb") as f:
            result = client.images.edits(
                model="gpt-image-1",
                image=f,
                prompt=prompt,
                size=size,
                n=1,
            )
    except Exception as e:
        # удаляем временный файл и пробрасываем ошибку
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise HTTPException(500, f"OpenAI error: {e}")

    # чистим временный файл
    if os.path.exists(tmp_name):
        os.remove(tmp_name)

    # === Получаем Base64 результата и сохраняем ===
    try:
        b64img = result.data[0].b64_json
        output_bytes = base64.b64decode(b64img)
        out_name = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join("files", out_name)
        with open(out_path, "wb") as f:
            f.write(output_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to save image: {e}")

    # Публичный URL файла (если BASE_URL не задан — вернём относительный путь)
    public_url = f"{BASE_URL}/files/{out_name}" if BASE_URL else f"/files/{out_name}"
    return {"ok": True, "image_url": public_url}
