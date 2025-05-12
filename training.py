from omegaconf import OmegaConf
import os, json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from helpers.data_loader import load_split, prepare_jsonl
from datasets import Dataset as HFDataset
from PIL import Image
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# === Load config ===
config = OmegaConf.load("configs/config.yaml")
print("Config loaded.")

# === Load image-caption pairs ===
train_pairs = load_split(config.dataset.train_base_dir, "train", config)
eval_pairs = load_split(config.dataset.eval_base_dir, "eval", config) if os.path.exists(config.dataset.eval_base_dir) else []
test_pairs = load_split(config.dataset.test_base_dir, "test", config) if os.path.exists(config.dataset.test_base_dir) else []

# === Convert to JSONL-ready format ===
train_data = prepare_jsonl("train", train_pairs, config.llava_prompt)
eval_data = prepare_jsonl("eval", eval_pairs, config.llava_prompt) if eval_pairs else []
test_data = prepare_jsonl("test", test_pairs, config.llava_prompt) if test_pairs else []

# === Save all to one JSONL file (optional) ===
with open(config.dataset.output_jsonl_path, 'w', encoding='utf-8') as f:
    for entry in train_data + eval_data + test_data:
        f.write(json.dumps(entry) + "\n")
print(f"Combined JSONL saved to: {config.dataset.output_jsonl_path}")

# === Convert to HF Datasets ===
train_dataset = HFDataset.from_list(train_data)
eval_dataset = HFDataset.from_list(eval_data) if eval_data else None
test_dataset = HFDataset.from_list(test_data) if test_data else None

print(f"Train samples: {len(train_dataset)}")
if eval_dataset:
    print(f"Eval samples: {len(eval_dataset)}")
if test_dataset:
    print(f"Test samples: {len(test_dataset)}")

# === Load model and processor ===
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

base_model_id = config.model.id
save_path = config.model.save_path
print(f"Loading base model: {base_model_id}")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)
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

lora_config = LoraConfig(
    r=config.lora.r,
    lora_alpha=config.lora.alpha,
    target_modules=list(config.lora.target_modules),
    lora_dropout=config.lora.dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === MLflow Setup and Resume Logic ===
#import mlflow
#from mlflow.tracking import MlflowClient

#mlflow.set_tracking_uri(config.mlflow_ip)
#mlflow.start_run(run_name=config.run_name, log_system_metrics=True)

#client = MlflowClient()
##experiment = client.get_experiment_by_name(config.project)
#experiment = client.get_experiment_by_name(config.project)
#if experiment is None:
#    print(f"⚠️ No MLflow experiment named '{config.project}' found. Creating it now.")
#    experiment_id = client.create_experiment(config.project)
#else:
#    experiment_id = experiment.experiment_id
#
#previous_runs = client.search_runs(
#    experiment_ids=[experiment.experiment_id],
#    filter_string=f"tags.mlflow.runName = '{config.run_name}'",
#    order_by=["start_time DESC"]
#)
#
#if previous_runs:
#    print(f"Found previous run: {previous_runs[0].info.run_id}")
#    lora_uri = previous_runs[0].info.artifact_uri + "/lora_adapters_1"
#    print(f"Loading LoRA weights from: {lora_uri}")
#    from peft import PeftModel
#    model = PeftModel.from_pretrained(model, lora_uri)
#else:
#    print("No previous run found. Starting from fresh LoRA config.")
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(config.mlflow_ip)
mlflow.start_run(run_name=config.run_name, log_system_metrics=True)

client = MlflowClient()
experiment = client.get_experiment_by_name(config.project)

if experiment is None:
    print(f"⚠️ No MLflow experiment named '{config.project}' found. Creating it now.")
    experiment_id = client.create_experiment(config.project)
else:
    experiment_id = experiment.experiment_id

# ✅ Use the safe variable experiment_id here
previous_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.runName = '{config.run_name}'",
    order_by=["start_time DESC"]
)

if previous_runs:
    print(f"Found previous run: {previous_runs[0].info.run_id}")
    lora_uri = previous_runs[0].info.artifact_uri + "/lora_adapters_1"
    print(f"Loading LoRA weights from: {lora_uri}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_uri)
else:
    print("No previous run found. Starting from fresh LoRA config.")



trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# === Training Arguments ===
from transformers import TrainingArguments

total_steps = (len(train_dataset) // (config.training.batch_size * config.training.grad_accum_steps)) * config.training.epochs
warmup_steps = int(0.05 * total_steps)

training_args = TrainingArguments(
    output_dir=config.model.save_path,
    per_device_train_batch_size=config.training.batch_size,
    gradient_accumulation_steps=config.training.grad_accum_steps,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=4,
    num_train_epochs=config.training.epochs,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    warmup_steps=warmup_steps,
    lr_scheduler_type="constant_with_warmup",
    optim="adamw_torch_fused",
    logging_steps=config.training.logging_steps,
    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=config.training.logging_steps if eval_dataset else None,
    fp16=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    report_to="mlflow"
)

# === Metrics Function ===
from evaluate import load
import numpy as np


def build_compute_metrics(tokenizer):
    bleu = load("bleu")
    rouge = load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # If preds is a tuple (logits, …), extract actual values
        preds = preds[0] if isinstance(preds, tuple) else preds

        # Safety check
        if preds is None or len(preds) == 0:
            return {"bleu": 0.0, "rougeL": 0.0}

        preds = np.array(preds)
        labels = np.where(np.array(labels) != -100, np.array(labels), tokenizer.pad_token_id)

        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print("⚠️ Decode error:", e)
            return {"bleu": 0.0, "rougeL": 0.0}

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu_score = bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["bleu"]
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]

        return {
            "bleu": round(bleu_score * 100, 2),
            "rougeL": round(rouge_score * 100, 2)
        }

    return compute_metrics



#def build_compute_metrics(tokenizer):
#    bleu = load("bleu")
#    rouge = load("rouge")
#
#    def compute_metrics(eval_preds):
#        preds, labels = eval_preds
#        preds = preds[0] if isinstance(preds, tuple) else preds
#        preds = np.array(preds)
#        labels = np.where(np.array(labels) != -100, np.array(labels), tokenizer.pad_token_id)
#
#        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#        decoded_preds = [p.strip() for p in decoded_preds]
#        decoded_labels = [l.strip() for l in decoded_labels]
#
#        bleu_score = bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])["bleu"]
#        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
#
#        return {
#            "bleu": round(bleu_score * 100, 2),
#            "rougeL": round(rouge_score * 100, 2)
#        }
#
#    return compute_metrics

compute_metrics_func = build_compute_metrics(tokenizer)

# === Trainer with eval sampling ===
import random
from trl import SFTTrainer

class SFTTrainerEvalSampling(SFTTrainer):
    def __init__(self, *args, eval_sample_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            return None
        if self.state.global_step < self.state.max_steps:
            sampled_idxs = random.sample(range(len(eval_dataset)), min(self.eval_sample_size, len(eval_dataset)))
            eval_dataset = eval_dataset.select(sampled_idxs)
        return super().get_eval_dataloader(eval_dataset)

trainer = SFTTrainerEvalSampling(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if eval_dataset else None,
    peft_config=None,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_func if eval_dataset else None,
    eval_sample_size=16
)

print("Starting training...")
trainer.train()


def preprocess_eval_dataset(dataset, processor):
    input_ids, attention_masks, labels = [], [], []

    for example in dataset:
        prompt = processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=True
        )
        image = Image.open(example["image_path"]).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)

        input_ids.append(inputs["input_ids"][0])
        attention_masks.append(inputs["attention_mask"][0])
        labels.append(inputs["input_ids"][0])  # Can adjust if you want true labels

    #dataset = dataset.add_column("input_ids", input_ids)
    #dataset = dataset.add_column("attention_mask", attention_masks)
    #dataset = dataset.add_column("labels", labels)
    
    dataset = dataset.add_column("input_ids", [x.tolist() for x in input_ids])
    dataset = dataset.add_column("attention_mask", [x.tolist() for x in attention_masks])
    dataset = dataset.add_column("labels", [x.tolist() for x in labels])


    return dataset




# === Final test evaluation and conditional registration ===
#test_metrics = {}
#if test_dataset:
#    print("Running final evaluation on test dataset...")
#    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
#    for k, v in test_metrics.items():
#        if isinstance(v, (int, float)):
#            mlflow.log_metric(f"test_{k}", v)
#else:
#    print("No test dataset found — skipping final evaluation.")


test_metrics = {}
if test_dataset:
    print("Running final evaluation on test dataset...")
    test_dataset = preprocess_eval_dataset(test_dataset, processor)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"test_{k}", v)
else:
    print("No test dataset found — skipping final evaluation.")



test_loss = test_metrics.get("test_loss", None)
should_register = False
save_reason = ""

previous_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{config.run_name}'",
    order_by=["metrics.test_loss ASC"]
)
previous_best_loss = previous_runs[0].data.metrics.get("test_loss") if previous_runs else None

if test_loss is None:
    save_reason = "missing_test_loss"
elif not previous_best_loss:
    should_register = True
    save_reason = "first_run"
elif test_loss < previous_best_loss:
    should_register = True
    save_reason = "better"
elif config.bypass_save:
    should_register = True
    save_reason = "forced_save"
else:
    save_reason = "not_best"

mlflow.set_tag("save_reason", save_reason)

if test_loss is not None:
    mlflow.log_metric("test_loss", test_loss)

#final_model_path = config.model.save_path + "_trained_model"
#model.save_pretrained(final_model_path)
#mlflow.log_artifacts(final_model_path, artifact_path="lora_adapters_1")

import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    mlflow.log_artifacts(tmpdir, artifact_path="lora_adapters_1")


mlflow.log_artifact("training.py")
mlflow.log_artifact("configs/config.yaml", artifact_path="config")

if should_register:
    run_id = mlflow.active_run().info.run_id
    model_registry_name = f"{config.model_name} - {config.model.id}"

    try:
        client.get_registered_model(model_registry_name)
    except Exception:
        client.create_registered_model(model_registry_name)

    model_version = client.create_model_version(
        name=model_registry_name,
        source=os.path.join(mlflow.get_artifact_uri(), "lora_adapters_1"),
        run_id=run_id
    )

    client.set_model_version_tag(model_registry_name, model_version.version, "model_type", "LoRA")
    client.set_model_version_tag(model_registry_name, model_version.version, "base_model", config.model.id)
    client.set_model_version_tag(model_registry_name, model_version.version, "task", config.task)

    print(f"LoRA adapters registered as version {model_version.version} of '{model_registry_name}'")
else:
    print("LoRA adapters not registered (not best run).")

