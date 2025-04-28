# fastapi_app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/generate-caption/")
async def generate_caption_api(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a dummy generated caption.
    Later, replace static output with real model inference.
    """

    contents = await file.read()
    try:
        # Try to open the uploaded file as an image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    # Here you would normally run inference: generate_caption(image, model)
    # For now, we just return a static dummy caption
    dummy_caption = "This is a static caption. (Model not integrated yet.)"

    return {"caption": dummy_caption}
