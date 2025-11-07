import os
import uuid
import base64
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("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")
BOTMOTHER_TOKEN = os.getenv("AZCK8484SECRET")
BASE_URL = os.getenv("BASE_URL")    # https://your-app.onrender.com

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# Папка для сохранения готовых изображений
os.makedirs("files", exist_ok=True)
app.mount("/files", StaticFiles(directory="files"), name="files")

@app.post("/img2img")
def img2img(payload: dict, x_api_token: str = Header(None)):
    # === Проверка секретного токена ===
    if x_api_token != BOTMOTHER_TOKEN:
        raise HTTPException(401, "Invalid X-Api-Token")

    prompt = payload.get("prompt")
    image_url = payload.get("image_url")
    size = payload.get("size", "1024x1024")

    # === Скачиваем картинку пользователя ===
    try:
        img_bytes = requests.get(image_url).content
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
                n=1
            )

        os.remove(tmp_name)

        # === Получаем Base64 результата ===
        b64img = result.data[0].b64_json
        output_bytes = base64.b64decode(b64img)

        # === Сохраняем результат в папку files/ ===
        out_name = f"{uuid.uuid4().hex}.png"
        out_path = f"files/{out_name}"

        with open(out_path, "wb") as f:
            f.write(output_bytes)

        # Публичный URL файла
        public_url = f"{BASE_URL}/files/{out_name}"

        return {"ok": True, "image_url": public_url}

    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")
