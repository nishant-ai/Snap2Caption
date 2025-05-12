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

---

## 🚀 Unit 6 & 7: Model Serving and Optimization

### 🧩 Model Inference Server (FastAPI-based)

This component is the **model inference server**, responsible for **serving the trained image captioning model via a production-grade FastAPI endpoint**.

It exposes a REST API where users can send an image and receive a caption in real time. This server is containerized and instrumented with monitoring hooks for production deployment.

- **API Input**: RGB image (base64-encoded JPEG or PNG)  
- **API Output**: A generated natural language caption in JSON format

The API also includes monitoring instrumentation via **Prometheus** and dashboards powered by **Grafana**, both hosted on our central KVM@TACC node.

> 🌐 **Monitoring Dashboard (Hosted at KVM@TACC)**  
> - **Floating IP**: `129.114.25.254`  
> - **Grafana Dashboard**: [http://129.114.25.254:3000/?orgId=1&from=now-6h&to=now&timezone=browser](http://129.114.25.254:3000/?orgId=1&from=now-6h&to=now&timezone=browser)  
> - **Prometheus Metrics UI**: [http://129.114.25.254:9090/query](http://129.114.25.254:9090/query)  
> - **FastAPI `/metrics` Endpoint**: [http://129.114.25.254:8000/metrics](http://129.114.25.254:8000/metrics)

**Main Serving Code**:

- 🔗 [`inference_server_setup.py`](https://github.com/nishant-ai/Snap2Caption/blob/model_serving_endpoint/inference_server_setup.py) – Main FastAPI app with model loading and inference logic
- 🔗 [`llava_fastapi_server.py`](https://github.com/nishant-ai/Snap2Caption/blob/model_serving_endpoint/llava_fastapi_server.py) – Optional wrapper/entry script for uvicorn
- 🔗 [`request.py`](https://github.com/nishant-ai/Snap2Caption/blob/model_serving_endpoint/request.py) – Defines API request/response formats using Pydantic

Start the API:

```bash
uvicorn inference_server_setup:app --host 0.0.0.0 --port 8000
```

Access Swagger docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

FastAPI inference endpoint:
http://129.114.109.50:8000/generate_caption

🔗 Feedback handling API:
-[store_feedback.py](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/store_feedback.py)

Start the Feedback API:
```bash
source minio_config.sh && uvicorn store_feedback:app --host 0.0.0.0 --port 8010
```
Those feedback samples are then:

1.Annotated in Label Studio.

2.Used for retraining and fine-tuning the captioning model.


## 🗄️ Unit 8: Persistent Storage & Data Services

### 💾 Centralized Block Storage at KVM@TACC

We provisioned a persistent **block storage volume** on our main node at **KVM@TACC**. This storage is mounted locally using `rclone` and is used to persist the runtime data of all critical MLOps services.

> 📍 Mounted at: `/mnt/<folder_name>`  
> 🔧 Mounted using: `rclone`  
> 🌐 Node: KVM@TACC (`129.114.25.254`)

#### ⬆️ Push New Data to Block Storage

```bash
rclone copy <local_path> <remote>:<bucket_or_path>
```

Example:

```bash
rclone copy ./feedback/user123/feedback.json chameleon:mlops-data/feedback/user123/
```

#### Services Using Block Storage

| Service       | Description                          |
|---------------|--------------------------------------|
| MLflow        | Logs experiments, artifacts          |
| MinIO         | Hosts feedback + intermediate data   |
| Prometheus    | Stores metrics (latency, counts)     |
| Grafana       | Persists dashboards and settings     |
| PostgreSQL    | Backs MLflow’s tracking DB           |
| Ray           | For distributed job execution (future)|

---

### 🪣 Object Storage at CHI@UC – Dataset Hosting

For hosting the **training dataset (InstaCities1M)**, we provisioned and mounted **object storage** at the **CHI@UC** site.

- This allows high-throughput access during model training
- Reduces I/O load on the central KVM node
- Scales with storage and compute demand

> 📍 Hosted on: CHI@UC  
> 📁 Mounted via `rclone` or FUSE into the training node  
> 🏷️ Mounted at paths like: `/mnt/images`, `/mnt/captions`

#### Example Mount Command

```bash
rclone mount chameleon:instacities1m/images /mnt/images --daemon
rclone mount chameleon:instacities1m/captions /mnt/captions --daemon
```

These paths are then passed into the training container:

```bash
docker run ... \
  --mount type=bind,source=/mnt/images,target=/mnt/InstaCities1M/images,readonly \
  --mount type=bind,source=/mnt/captions,target=/mnt/InstaCities1M/captions,readonly \
  ...
```

🔄 Data Pipeline (ETL)
The offline data pipeline is composed of three main stages:

🔹 Extract
Downloads the InstaCities1M.zip dataset using aria2 for fast, parallelized download via multiple connections.

🔹 Transform
Filters image-caption pairs for the following cities:
["newyork", "chicago", "sanfrancisco"]

Converts all folder names to lowercase for consistency.

Verifies and matches each image to a valid caption.

🔹 Load
Uploads the cleaned and structured dataset to Chameleon Object Storage.

Container: object-persist-sbb9447-project54

Accessible at: [Chameleon UC Dashboard - Object Storage]("https://chi.uc.chameleoncloud.org/project/containers/container/object-persist-sbb9447-project54")

📦 Storage Volume
All dataset stages are persisted under the volume: instacities1m

🔗 Code References
See the extract-data, transform-data, and load-data services in
[docker-compose.yml]("https://github.com/nishant-ai/Snap2Caption/blob/main/docker-compose.yml")

🧹 Data Split & Preprocessing
Dataset was shuffled and split using an 80/10/10 ratio:

80% Training

10% Validation

10% Testing

Care was taken to avoid data leakage between partitions.

Preprocessing Steps:
Lowercased all folder and city names

Ensured one-to-one matching between images and captions

Removed corrupted or unmatched image-caption pairs
---

### ✅ Storage Strategy Summary

| Storage Type  | Location     | Purpose                                   |
|---------------|--------------|-------------------------------------------|
| Block Storage | KVM@TACC     | Persistent logs, system state, feedback   |
| Object Storage| CHI@UC       | Training datasets (images + captions)     |

This hybrid setup ensures:
- Efficient I/O for training
- Secure persistence of all service-level data
- Separation of concerns between system and training workloads


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
🏗️ Infrastructure Provisioning with Terraform
Terraform was used to provision the required infrastructure on Chameleon Cloud, which included:

Three Virtual Machines (VMs) for a self-managed Kubernetes cluster.

Network configurations to allow secure communication between nodes.

🔧 Configuration with Ansible
After provisioning, Ansible was utilized for configuration and package installations. Key tasks included:

SSH configuration and key management.

Installing dependencies like Docker and Kubernetes packages.

Running Ansible playbooks to configure the VMs for Kubernetes deployment.

☸️ Kubernetes Deployment
Kubernetes was deployed using Kubespray, an Ansible-based Kubernetes installer. This allowed for:

Setting up a three-node Kubernetes cluster.

Applying essential configurations like Docker networking and registry.

Deploying the cluster in a resilient, highly-available configuration.

💡 Key Takeaways & Issues Faced
The primary takeaways from this setup included:

Understanding how to manage cloud infrastructure using Terraform and Ansible.

Automating complex multi-node Kubernetes deployments with Kubespray.

Issues faced included:

Firewall configurations.

SSH permission adjustments.

These were resolved by updating Ansible playbooks.

🔄 Next Steps
Debug the issues in the Argo Workflows and finalize the CI/CD pipelines.

Extend the deployment to include ArgoCD for continuous delivery.

Optimize the resource allocation for better scalability and cost-effectiveness.

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
