### ### ### ### ### ### Import Libraries ### ### ### ### ### ### 
import os
import json
import torch
import mlflow
import torchao
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from trl import SFTTrainer
from peft import LoraConfig
from torch.optim import AdamW
from omegaconf import OmegaConf
# from evaluate import load as load_metric
from datasets import load_dataset
from torch.utils.data import Dataset
from mlflow.tracking import MlflowClient
from evaluate import load as load_metric
from datasets import Dataset as HFDataset
from multiprocessing import Pool, cpu_count
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()



# ### ### ### ### ### ### Setting up Configs ### ### ### ### ### ### 
config = OmegaConf.load("./config.yaml")
print("Configs ::: ", config)

# --- Experiment Metadata ---
RUN_NAME = config.run_name
PROJECT = config.project
MODEL_NAME = config.model_name
TASK = config.task
NOTES = config.notes
DESCRIPTION = config.description

# --- Dataset Configs ---
DATASET = config.dataset.name
CITIES = config.dataset.cities
BASE_IMAGES_DIR = config.dataset.base_images_dir
BASE_CAPTIONS_DIR = config.dataset.base_captions_dir
OUTPUT_JSONL_PATH = config.dataset.output_jsonl_path

LLAVA_CHAT_TEMPLATE = config.llava_prompt

# --- Model Configs ---
MODEL_ID = config.model.id
MODEL_SAVE_PATH = config.model.save_path

# --- LoRA Configs ---
LORA_R = config.lora.r
LORA_ALPHA = config.lora.alpha
LORA_DROPOUT = config.lora.dropout
TARGET_MODULES = list(config.lora.target_modules)

# --- Training Configs ---
TRAIN_BATCH_SIZE = config.training.batch_size
GRADIENT_ACCUMULATION_STEPS = config.training.grad_accum_steps
NO_OF_EPOCHS = config.training.epochs
LEARNING_RATE = config.training.learning_rate
LOGGING_STEPS = config.training.logging_steps
WEIGHT_DECAY = config.training.weight_decay

# --- Infra Configs ---
MLFLOW_IP = config.mlflow_ip

print(os.path.exists(BASE_IMAGES_DIR))
print(os.path.exists(BASE_CAPTIONS_DIR))

### ### ### ### ### ### Setting up MLFLOW ### ### ### ### ### ### 
mlflow.set_tracking_uri(MLFLOW_IP)
mlflow.start_run(run_name=RUN_NAME, log_system_metrics=True)

mlflow.set_tags({
    "PROJECT": PROJECT,
    "MODEL_NAME": MODEL_NAME,
    "TASK": TASK,
    "NOTES": NOTES,
    "DATASET": DATASET
    
})

DESCRIPTION = f"{RUN_NAME} - {MODEL_NAME} | LoRA_R {LORA_R} | LoRA_ALPHA {LORA_ALPHA} | {DATASET} | EPOCHS={NO_OF_EPOCHS}"
mlflow.set_tag("mlflow.note.content", DESCRIPTION)

print("GPU AVAILABLE - ", torch.cuda.is_available())



### ### ### ### ### ### Dataset Parsing ### ### ### ### ### ###
images_files = []
captions_files = []

for city in CITIES:
    img = BASE_IMAGES_DIR + city + '/' + np.array(os.listdir(BASE_IMAGES_DIR + city))
    caption = BASE_CAPTIONS_DIR + city + '/' + np.array(os.listdir(BASE_CAPTIONS_DIR + city))
    images_files.extend(img)
    captions_files.extend(caption)


# Clean filenames
image_ids = {os.path.splitext(os.path.basename(img))[0] for img in images_files}
caption_ids = {os.path.splitext(os.path.basename(cap))[0] for cap in captions_files}

# Now match
common_ids = image_ids & caption_ids

# Filter
filtered_image_files = [img for img in images_files if os.path.splitext(os.path.basename(img))[0] in common_ids]
filtered_caption_files = [cap for cap in captions_files if os.path.splitext(os.path.basename(cap))[0] in common_ids]

images_files = filtered_image_files
captions_files = filtered_caption_files

images_files = images_files[:3000]
captions_files = captions_files[:3000]

print(len(images_files), len(captions_files))

# --- Worker function ---
def process_pair(i):
    try:
        img_path = images_files[i]
        caption_path = captions_files[i]

        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip().replace('\n', ' ')
            if not caption:
                return None

        # Create messages field directly
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LLAVA_CHAT_TEMPLATE},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ]

        return {
            "image_path": img_path,
            "messages": messages
        }

    except Exception:
        return None


# --- Multiprocessing ---
with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_pair, range(len(images_files))), total=len(images_files)))


data = [entry for entry in results if entry is not None]

# --- Write JSONL File ---
with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")

print(f"JSONL created: {OUTPUT_JSONL_PATH} with {len(data)} samples.")




### ### ### ### ### ### Configuring Model ### ### ### ### ### ###

# --- Configuration ---
model_id = MODEL_ID
data_path = OUTPUT_JSONL_PATH  # path to your formatted JSONL file
output_dir = MODEL_SAVE_PATH

# --- Model Loading (4bit Quantization) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False
# )

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
tokenizer = processor.tokenizer

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}"
    "{% endfor %}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)




# Train Test Split Read JSONL manually
dataset = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line.strip())
        dataset.append(example)

print(f"Loaded {len(dataset)} samples.")


# dataset = HFDataset.from_list(dataset)

# split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

# train_dataset = split_dataset["train"]
# eval_dataset = split_dataset["test"]

# print(len(train_dataset))
# print(len(eval_dataset))


train_dataset = dataset[:2000]
eval_dataset = dataset[2000:3000]

train_dataset = HFDataset.from_list(train_dataset)
eval_dataset = HFDataset.from_list(eval_dataset)

print(len(train_dataset))
print(len(eval_dataset))


### ### ### ### ### ### Evaluation Metrics ### ### ### ### ### ###

from evaluate import load
import numpy as np

def build_compute_metrics(tokenizer):
    bleu = load("bleu")
    rouge = load("rouge")

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        # If preds are tuple (logits, ), take first
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.array(preds)
        labels = np.array(labels)

        # Take argmax if needed
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        # Clean labels (-100 to pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Convert to list
        preds = preds.tolist()
        labels = labels.tolist()

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Clean whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute BLEU and ROUGE
        bleu_score = bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["bleu"]
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]

        return {
            "bleu": round(bleu_score * 100, 2),
            "rougeL": round(rouge_score * 100, 2),
        }

    return compute_metrics

compute_metrics_func = build_compute_metrics(tokenizer)



total_steps = (len(dataset) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NO_OF_EPOCHS
warmup_steps = int(0.05 * total_steps)

# --- SFT Trainer ---
training_args = TrainingArguments(
    
    dataloader_num_workers=4,
    logging_steps=LOGGING_STEPS,
    
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    warmup_steps = warmup_steps,
    num_train_epochs=NO_OF_EPOCHS,
    lr_scheduler_type="constant_with_warmup",
    
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    per_device_eval_batch_size=2,
    eval_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=LOGGING_STEPS,
    
    fp16=True,
    report_to="mlflow",
    
    save_strategy="epoch",
    output_dir=output_dir,

    remove_unused_columns=False
)

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)


mlflow_log_dict = {
    "LORA_CONFIGS": {
            "r":LORA_R,
            "lora_alpha":LORA_ALPHA,
            "target_modules":TARGET_MODULES,
            "lora_dropout":LORA_DROPOUT,
            "bias":"none",
            "task_type":"CAUSAL_LM"
        },
    # "TRAINING_ARGS":training_args.to_dict()
}

mlflow.log_params(mlflow_log_dict)



import random
from trl import SFTTrainer

class SFTTrainerEvalSampling(SFTTrainer):
    def __init__(self, *args, eval_sample_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size

    def get_eval_dataloader(self, eval_dataset=None):
        '''
        Samples the evaluation dataset and returns a subset 
        of size self.eval_sample_size.
        '''
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if self.state.global_step >= self.state.max_steps:
            return super().get_eval_dataloader(eval_dataset)

        
        idxs = random.sample(range(len(eval_dataset)), self.eval_sample_size)
        print(idxs)
        eval_subset = eval_dataset.select(idxs)

        return super().get_eval_dataloader(eval_subset)

class MLflowLoggingSFTTrainer(SFTTrainerEvalSampling):
    def log(self, logs: dict, step: int = None) -> None:
        super().log(logs)

        # Filter only float-compatible keys
        safe_logs = {
            k: float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float, np.float32, np.float64)) and not np.isnan(v)
        }

        step_to_log = self.state.global_step if step is None else int(step)
        mlflow.log_metrics(safe_logs, step=step_to_log)


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
mlflow.log_metric("trainable_params", trainable_params)
mlflow.log_metric("total_params", total_params)


trainer = MLflowLoggingSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_sample_size=10,
    peft_config=lora_config,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_func,
)


print("Starting Training ::::::")

# --- Start Fine-tuning ---
trainer.train()


output_dir = output_dir+"_trained_model"
trainer.model.save_pretrained(output_dir)
print(f"✅ Training complete. Model saved at {output_dir}")



mlflow.log_artifact("train.py")
mlflow.log_artifact("./configs/config.yaml", artifact_path="config")
mlflow.log_artifacts(output_dir + '/adapter_config.json', artifact_path="lora_adapters")
mlflow.log_artifacts(output_dir + '/adapter_model.safetensors', artifact_path="lora_adapters")
mlflow.log_artifacts(output_dir + '/README.md', artifact_path="lora_adapters")


# Make sure you're inside an active MLflow run
if mlflow.active_run() is None:
    raise RuntimeError("MLflow run must be started before model registration.")

# Define variables
lora_adapter_dir = output_dir  # where your adapter_model.bin lives
model_registry_name = MODEL_NAME + " - " + MODEL_ID

mlflow.log_artifacts(lora_adapter_dir, artifact_path="lora_adapters_1")

# Create a model version in the registry
client = MlflowClient()
run_id = mlflow.active_run().info.run_id

# Create registered model entry (if not already created)
try:
    client.get_registered_model(model_registry_name)
except Exception:
    client.create_registered_model(model_registry_name)

# Create new version pointing to adapter artifacts
model_version = client.create_model_version(
    name=model_registry_name,
    source=os.path.join(mlflow.get_artifact_uri(), "lora_adapters_1"),
    run_id=run_id
)

# Add tags for traceability
client.set_model_version_tag(model_registry_name, model_version.version, "model_type", "LoRA")
client.set_model_version_tag(model_registry_name, model_version.version, "base_model", MODEL_ID)
client.set_model_version_tag(model_registry_name, model_version.version, "stage", "None")
client.set_model_version_tag(model_registry_name, model_version.version, "task", "Image Captioning")

print(f"LoRA adapters registered as model version {model_version.version} under '{model_registry_name}'")


mlflow.end_run()



from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
from PIL import Image
import torch

# === Load Processor and Model (once at startup) ===
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

base_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "llava_lora_instagram_trained_model")
model.to("cuda:0")


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
        outputs = model.generate(**inputs, max_new_tokens=100)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()


result = generate_caption("download.jpeg")
print("📝 Generated Caption:", result)