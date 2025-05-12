# Snap2Caption

> *Generate an Instagram‑ready caption and a list of high‑engagement hashtags from any photograph in a single API call.*

---

## 1 · Who We Serve — Aisha *(Value Proposition)*

Aisha is a solo lifestyle creator who snaps urban photos during her commute. She wants to post consistently but hates spending time on copy‑writing and keyword research. **Snap2Caption** lets her drop an image into a mobile or desktop workflow and receive:

* a short caption in her tone of voice
* 5‑10 context‑aware hashtags optimised for reach

Design promises—and how the code meets them:

| Need              | Implementation in the repo                                                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One‑step usage    | HTTP `POST /generate-caption` in the [FastAPI app](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/fastapi_app.py)                         |
| < 2 s P90 latency | Lightweight inference stack written in Python; model cached on start‑up; no heavy image preprocessing                                                    |
| Fresh hashtags    | Model can be re‑trained offline with new data via scripts in [`training_scripts`](https://github.com/nishant-ai/Snap2Caption/tree/main/training_scripts) |

**Current scale** (May 2025)

| Item                         | Figure                |
| ---------------------------- | --------------------- |
| Training images used in demo | 100 k (≈ 5 GB)        |
| Fine‑tuned checkpoint        | 14 GB FP16            |
| Observed peak load (staging) | \~300 requests / hour |

---

## 2 · Data & Training *(Units 4, 5 & 8)*

All model development lives inside the **`training_scripts`** directory. It contains:

* `finetune_lora.py` – LoRA fine‑tuning driver for LLaVA‑1.5‑7B
* `dataset_loader.py` – minimal loader for InstaCities‑style JPEG + caption rows
* `code_wandb.ipynb` – interactive notebook that logs experiments to Weights & Biases

Training is executed locally or on a GPU node with:

```bash
python training_scripts/finetune_lora.py \
  --config configs/finetune_a100.yaml
```

Artifacts are written to `output/` and can be copied into the serving container.

---

## 🧠 Unit 4 & 5: Model Training

### 🎯 Modeling

We use **LLaVA models** for multimodal caption generation:

- [LLaVA-1.5 (7B, Vicuna backend)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [LLaVA-1.6 (7B, Mistral backend)](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)

These models are fine-tuned using **LoRA adapters** for parameter-efficient adaptation on the **InstaCities1M** dataset, where each image is mapped to a caption.

- **Input**: RGB image (preprocessed to 224x224)  
- **Output**: Natural language caption (generated via autoregressive decoding)

🔗 [training.py](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/training.py)

---

### 🏗️ Infrastructure: Training Environment (Docker)

To ensure portability and reproducibility, we define our full training environment in:

🔗 [Dockerfile.trainingenv](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/Dockerfile.trainingenv)

It includes:
- Python 3.10
- PyTorch 2.2.1 + CUDA 12.1
- MLflow, HuggingFace Hub, LoRA support
- Pillow, Lightning, Tensorboard, etc.

**Build the image:**

```bash
docker build -f Dockerfile.trainingenv -t snap2caption-train .
```

**Run training inside the container:**

```bash
docker run --rm --gpus all \
  -v /mnt/dataset:/mnt/dataset \
  -v $(pwd)/configs:/configs \
  -v $(pwd)/outputs:/app/outputs \
  snap2caption-train \
  python training.py --config /configs/config.yaml
```

---
### ⚙️ Configuration-Driven Pipeline

We use a dynamic YAML-based configuration system to control the training pipeline.

🔗 [config.yaml](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/config.yaml)  
🔗 [commands.txt](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/commands.txt)

- `config.yaml` defines:
  - Dataset paths and per-city sampling
  - LoRA hyperparameters (e.g., learning rate, rank, batch size)
  - MLflow logging and model registration behavior
  - Final evaluation and model output path

- `commands.txt` includes:
  - **All shell commands required to build the Docker container**
  - **Run the training pipeline end-to-end**
  - **Perform inference tests or trigger retraining runs**

> 💡 This file serves as a **complete reference** for executing the pipeline on any fresh Chameleon node or local setup. Just follow the commands top to bottom.

---

### 🔄 Train and Re-train Pipeline

- **Entry point**: [training.py](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/training.py)
- Accepts dynamic config paths
- Fine-tunes LLaVA with LoRA
- Logs all metrics and artifacts to a **remote MLflow server** at KVM@TACC
- Supports retraining on feedback data (see Unit 8)

---
### 🔍 Experiment Tracking with MLflow

All training runs are tracked using **MLflow**, hosted at our central node on KVM@TACC:

🌐 MLflow UI: [http://129.114.25.254:8000](http://129.114.25.254:8000)

Logged details include:
- **Parameters**: learning rate, batch size, LoRA config
- **Metrics**: test loss, BLEU, CIDEr, token length
- **Artifacts**: LoRA adapters, sample captions
- **Conditional model registration**: based on evaluation performance

🔗 [MLflow logging snippet (training.py#L340–370)](https://github.com/nishant-ai/Snap2Caption/blob/training-pipeline/training.py#L340)


## 3 · Model Serving *(Units 6 & 7)*

Production inference code is packaged in **`base_api`**:

* [`fastapi_app.py`](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/fastapi_app.py) — REST server exposing `/generate-caption`.
* [`model.py`](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/model.py) — model loader + generation routine.
* [`store_feedback.py`](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/store_feedback.py) — persists optional user ratings for later re‑training.

Typical request/response:

```http
POST /generate-caption
{ "image_base64": "..." }
→ 200 OK
{ "caption": "Golden‑hour coffee vibes ☕", "hashtags": ["#citysunset", "#cafeculture", ...] }
```

---

## 4 · Infrastructure as Code *(Units 2 & 3)*

| Layer          | Description                                               | Directory                                                                     |
| -------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Terraform      | Allocates GPU VM, VPC, and S3 bucket on Chameleon         | [`tf`](https://github.com/nishant-ai/Snap2Caption/tree/main/tf)               |
| Ansible        | Installs Docker, NVIDIA drivers, and pulls runtime images | [`ansible`](https://github.com/nishant-ai/Snap2Caption/tree/main/ansible)     |
| Kubernetes     | Optional manifests for container orchestration            | [`k8s`](https://github.com/nishant-ai/Snap2Caption/tree/main/k8s)             |
| GitHub Actions | CI pipeline for build, push, and lint                     | [`workflows`](https://github.com/nishant-ai/Snap2Caption/tree/main/workflows) |

### Chameleon one‑shot deployment

```bash
# 1 · Provision resources
cd tf && terraform init && terraform apply -auto-approve

# 2 · Configure VM
ansible-playbook ansible/site.yml -i ansible/inventory

# 3 · Run serving container
ssh <vm_ip>
docker compose -f base_api/docker-compose.yml up -d
```

---

## 5 · Frontend Preview

A React demo lives in [`frontend`](https://github.com/nishant-ai/Snap2Caption/tree/main/frontend). Point it at the FastAPI host to try caption generation in the browser.

---

## 6 · Repository Map (quick reference)

| Area                        | Path                 |
| --------------------------- | -------------------- |
| Model serving (FastAPI)     | `base_api/`          |
| Training scripts            | `training_scripts/`  |
| Infrastructure – Terraform  | `tf/`                |
| Infrastructure – Ansible    | `ansible/`           |
| Infrastructure – Kubernetes | `k8s/`               |
| CI/CD workflows             | `.github/workflows/` |
| React front‑end             | `frontend/`          |

---

## 7 · Run a Local Demo

```bash
# build image
make docker-build
# start API on localhost:8000
make docker-run
# call it
python demo_request.py sample.jpg
```

---

## License

MIT
