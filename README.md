# Snap2Caption

> *Turn any Instagram photo into a scroll‑stopping post, complete with a tailored caption and trending hashtags, in under two seconds.*

---

## Meet **Aisha**, Our Customer  *(Unit 1 – Value Proposition)*

Aisha is a solo lifestyle creator who posts colourful city snapshots every evening after work. She loves sharing, hates word‑smithing, and checks insights obsessively. Snap2Caption exists for creators like her: upload a photo, get a punchy caption in her voice plus hashtags tuned to what’s hot **right now**.

**Design commitments influenced by Aisha**

| Promise                | Design consequence                                             |
| ---------------------- | -------------------------------------------------------------- |
| Zero friction          | Single REST endpoint (`/generate-caption`) callable via `curl` |
| Timely trend awareness | Overnight automated re‑training on fresh InstaCities data      |
| Mobile‑first speed     | P90 end‑to‑end latency < 2 s; GPU‑backed autoscaling           |

**Scale (May 2025)**

| Component             | Value today                      | 12‑month projection   |
| --------------------- | -------------------------------- | --------------------- |
| Training data volume  | 50 GB (100 k images)             | 1 M images (≈ 500 GB) |
| Fine‑tuned model size | 14 GB FP16 (LLaVA‑1.5‑7B + LoRA) | 8 GB INT8 distilled   |
| Training duration     | 5 h on 4× A100‑80G               | 3 h with DDP + FP8    |
| Peak inference load   | 300 req / h                      | 5 000 req / h         |

---

## From Pixels to Parquet — **Data Foundations**  *(Unit 8)*

The nightly ETL pipeline fetches raw JPEGs from InstaCities, validates and resizes them to `300×300`, wraps each in a chat template, then materialises Parquet manifests. Splits are stratified 80 / 10 / 10 by *user* to prevent leakage. Processed images live in a Ceph‑backed object store; metadata in Parquet drives training and offline tests.

| Persistent mount               | Purpose                   | Size  |
| ------------------------------ | ------------------------- | ----- |
| `/mnt/object/snap2caption-raw` | Raw & processed images    | 50 GB |
| `/mnt/block/checkpoints`       | Model & tokenizer weights | 25 GB |
| `/mnt/block/experiments`       | Offline W\&B cache        | 10 GB |

During production, each inference and user rating is streamed via Kafka into `/mnt/block/online_events`. A Spark batch joins engagement metrics, computes KL‑drift, and—if drift > 0.15—raises a Prometheus alert that triggers the automated re‑train job.

---

## Teaching the Model — **Training & Experimentation**  *(Units 4 & 5)*

We cast caption + hashtag generation as *image‑conditioned sequence generation*. The model of record is **LLaVA‑1.5‑7B** fine‑tuned with rank‑16 LoRA adapters.

| Element                  | Implementation reference                                                                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training scripts         | [https://github.com/nishant-ai/Snap2Caption/tree/main/training\_scripts](https://github.com/nishant-ai/Snap2Caption/tree/main/training_scripts)                                    |
| Full training pipeline   | [https://github.com/nishant-ai/Snap2Caption/tree/training-pipeline](https://github.com/nishant-ai/Snap2Caption/tree/training-pipeline)                                             |
| W\&B experiment notebook | [https://github.com/nishant-ai/Snap2Caption/blob/main/training\_scripts/code\_wandb.ipynb](https://github.com/nishant-ai/Snap2Caption/blob/main/training_scripts/code_wandb.ipynb) |

*Distributed training (DDP + FP16) cut wall‑clock from 8 h → 5 h while BLEU‑4 improved to 0.31.* Weekly CronJob retrains, or Prometheus fires an on‑demand run when drift is detected.

---

## Shipping Intelligence — **Infrastructure & Continuous Delivery**  *(Units 2 & 3)*

All infrastructure is codified and repeatable:

| Layer             | Repository location                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Terraform         | [https://github.com/nishant-ai/Snap2Caption/tree/main/tf](https://github.com/nishant-ai/Snap2Caption/tree/main/tf)               |
| Ansible           | [https://github.com/nishant-ai/Snap2Caption/tree/main/ansible](https://github.com/nishant-ai/Snap2Caption/tree/main/ansible)     |
| Kubernetes        | [https://github.com/nishant-ai/Snap2Caption/tree/main/k8s](https://github.com/nishant-ai/Snap2Caption/tree/main/k8s)             |
| CI / CD Workflows | [https://github.com/nishant-ai/Snap2Caption/tree/main/workflows](https://github.com/nishant-ai/Snap2Caption/tree/main/workflows) |

GitHub Actions builds every commit; Argo CD syncs to **staging**. Argo Rollouts governs progressive promotion to **canary** and **prod**; models advance automatically after SLOs stay green for 24 h. When data‑drift alerts post a GitHub Release, the same pipeline re‑trains and redeploys, closing the loop.

**Chameleon quick‑start**

```bash
# provision GPU K8s + storage via Terraform
make chi-init && make chi-apply

# bootstrap Argo CD (Ansible playbook)
make argocd-bootstrap

# monitor rollout, then try the API
make argocd-watch &
python demo_request.py sample.jpg
```

---

## Serving the Magic — **Real‑time Inference & Evaluation**  *(Units 6 & 7)*

Serving stack:

| Component              | Link                                                                                                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base API package       | [https://github.com/nishant-ai/Snap2Caption/tree/main/base\_api](https://github.com/nishant-ai/Snap2Caption/tree/main/base_api)                                      |
| FastAPI endpoint       | [https://github.com/nishant-ai/Snap2Caption/blob/main/base\_api/fastapi\_app.py](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/fastapi_app.py)       |
| Model invocation logic | [https://github.com/nishant-ai/Snap2Caption/blob/main/base\_api/model.py](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/model.py)                    |
| Feedback loop backend  | [https://github.com/nishant-ai/Snap2Caption/blob/main/base\_api/store\_feedback.py](https://github.com/nishant-ai/Snap2Caption/blob/main/base_api/store_feedback.py) |

The FastAPI server loads an 8‑bit ONNX checkpoint, decodes images with turbo‑JPEG, and answers:

```http
POST /generate-caption
{ "image_base64": "…" }
→ 200 OK
{ "caption": "Golden hour coffee vibes ☕🌇", "hashtags": ["#citysunset", …] }
```

KEDA scales replicas 0 → 10 based on queue depth. Offline tests (`pytest`), load tests (`locust`) and Grafana dashboards validate quality and performance before promotion.

---

## Frontend Touchpoint

A lightweight React front‑end consumes the API and showcases caption suggestions:
[https://github.com/nishant-ai/Snap2Caption/tree/main/frontend](https://github.com/nishant-ai/Snap2Caption/tree/main/frontend)

---

## Repository Atlas

| Area                       | Real link                                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Infrastructure (Terraform) | [https://github.com/nishant-ai/Snap2Caption/tree/main/tf](https://github.com/nishant-ai/Snap2Caption/tree/main/tf)                              |
| Configuration (Ansible)    | [https://github.com/nishant-ai/Snap2Caption/tree/main/ansible](https://github.com/nishant-ai/Snap2Caption/tree/main/ansible)                    |
| K8s manifests              | [https://github.com/nishant-ai/Snap2Caption/tree/main/k8s](https://github.com/nishant-ai/Snap2Caption/tree/main/k8s)                            |
| CI/CD workflows            | [https://github.com/nishant-ai/Snap2Caption/tree/main/workflows](https://github.com/nishant-ai/Snap2Caption/tree/main/workflows)                |
| Model training scripts     | [https://github.com/nishant-ai/Snap2Caption/tree/main/training\_scripts](https://github.com/nishant-ai/Snap2Caption/tree/main/training_scripts) |
| Full training pipeline     | [https://github.com/nishant-ai/Snap2Caption/tree/training-pipeline](https://github.com/nishant-ai/Snap2Caption/tree/training-pipeline)          |
| Serving code               | [https://github.com/nishant-ai/Snap2Caption/tree/main/base\_api](https://github.com/nishant-ai/Snap2Caption/tree/main/base_api)                 |
| Frontend (React)           | [https://github.com/nishant-ai/Snap2Caption/tree/main/frontend](https://github.com/nishant-ai/Snap2Caption/tree/main/frontend)                  |

---

## Local Taste Test

```bash
make docker-build
make docker-run
python demo_request.py sample.jpg
```

---

## License

MIT
