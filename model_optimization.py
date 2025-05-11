import os
import torch
import mlflow
from PIL import Image
from peft import PeftModel
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from transformers import AutoTokenizer, AutoProcessor
from transformers import LlavaForConditionalGeneration, AutoProcessor

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
from PIL import Image
import torch

config = OmegaConf.load("config_model_inference.yaml")

BASE_MODEL_ID = config.base_model_id
LORA_ADAPTORS = config.lora_adaptors
OUTPUT_PATH = config.output_path
MLFLOW_URI = config.mlflow.uri
MLFLOW_RUN_NAME = config.mlflow.run_name
RUN_ID = config.mlflow.run_id

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

lora_dir = download_artifacts(
    run_id=RUN_ID,
    artifact_path="lora_adapters_1",
    dst_path = OUTPUT_PATH
)
print("LoRA downloaded to:", lora_dir)

# Merging LORA and Base MODEL
base_model_id = BASE_MODEL_ID
adapter_path = LORA_ADAPTORS

# === Load Processor and Model (once at startup) ===
processor = LlavaNextProcessor.from_pretrained(base_model_id)
tokenizer = processor

base_model = LlavaNextForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() 
model.to("cuda:0")

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}"
    "{% endfor %}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)

output_path = OUTPUT_PATH
os.makedirs(output_path, exist_ok=True)

# === Define Functional Interface ===
def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "You are a social media influencer. Write a captivating Instagram caption for this image that will engage more viewers and boost interaction. Analyze the image to decide the tone of the caption."}
        ]
    }]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            repetition_penalty=1.2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()

result = generate_caption("download.jpeg")
print("📝 Generated Caption:", result)