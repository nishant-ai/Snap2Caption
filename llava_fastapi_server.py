from fastapi import FastAPI
from pydantic import BaseModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
from PIL import Image
import torch
import base64
import io
from prometheus_client import Counter, Summary, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Prometheus metrics
INFERENCE_COUNTER = Counter("inference_requests_total", "Total caption requests")
INFERENCE_LATENCY = Summary("inference_latency_seconds", "Time to generate caption")
OUTPUT_LENGTH = Histogram("model_output_length_tokens", "Length of generated caption", buckets=[10, 20, 40, 80, 120])
FAILURE_COUNTER = Counter("inference_failure_total", "Caption generation failures")


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


@INFERENCE_LATENCY.time()
@app.post("/generate_caption")
async def generate_caption(req: CaptionRequest):
    try:
        INFERENCE_COUNTER.inc()

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
        OUTPUT_LENGTH.observe(len(output_ids[0]))
        return {"caption": caption}
    except Exception as e:
        FAILURE_COUNTER.inc()
        raise e

                                                
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
