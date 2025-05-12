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
