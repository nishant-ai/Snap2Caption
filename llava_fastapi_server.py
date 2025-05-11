from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
from PIL import Image
import torch
import base64
import io

app = FastAPI()

class CaptionRequest(BaseModel):
    image: str  # base64 string
    prompt: str

# Load model and processor once
LORA_ADAPTER_PATH = "/lora_adapters/lora_adapters_1"
BASE_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(BASE_MODEL_ID)
base_model = LlavaNextForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).merge_and_unload().to("cuda:0")

processor.tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}"
    "{% endfor %}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)

@app.post("/generate_caption")
async def generate_caption(req: CaptionRequest):
    image_data = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": req.prompt}
        ]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to("cuda:0")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=140,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return {"caption": caption}

                                                
