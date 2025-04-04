# Snap2Caption

## AI-Powered Image Analysis System

### Value Proposition
Snap2Caption is an AI-driven image analysis system that automates image captioning and object detection. We propose a machine learning solution that can be integrated into existing social media platforms such as LinkedIn, Instagram, or Pinterest to enhance user experience by automatically generating meaningful image captions.

### Current Non-ML Status Quo

In many platforms, image management relies on manual processes:

- **Manual Tagging**: Users add captions and tags inconsistently, leading to unstructured metadata.
- **Limited Searchability**: Incomplete metadata hampers content discovery.
- **Accessibility Gaps**: Lack of descriptive text affects visually impaired users.
- **Operational Inefficiencies**: Scaling manual processes is resource-intensive.

### Business Metrics

Snap2Caption aims to improve the following metrics:

- **Metadata Coverage**: Achieve comprehensive image tagging.
- **Content Discoverability**: Enhanced search precision and recall.
- **Accessibility Compliance**: More content meeting accessibility standards.
- **Operational Efficiency**: Reduced manual effort in content management.

### Contributors

| Name               | Role                                 | Contribution Links                      |
|--------------------|--------------------------------------|-----------------------------------------|
| **All Members**    | Shared Responsibilities              | [Commits](#)                            |
| Jay Doshi          | Model Training, Serving Implementation | [Commits](#)                            |
| Nishant Sharma     | Training Pipeline, Experiment Tracking | [Commits](#)                            |
| Shreyansh Bhalani  | Data Pipelining, Model Serving       | [Commits](#)                            |
| Harsh Golani       | Data Pipeline, System Monitoring     | [Commits](#)                            |

### System Diagram

![System Diagram](https://github.com/nishant-ai/Snap2Caption/blob/main/SystemDesign.jpeg)

### Summary of Outside Materials

| Component       | Creation Details               | Conditions of Use                         | Official Link |
|----------------|--------------------------------|-------------------------------------------|---------------|
| **MS COCO Dataset** | Created by Microsoft for research | Open access for research and commercial use | [cocodataset.org](https://cocodataset.org/#home) |
| **Flickr30K**       | Academic project dataset       | Research use with attribution             | [Flickr30K Dataset](https://shannon.cs.illinois.edu/DenotationGraph/) |
| **YOLOv5**          | Developed by Ultralytics       | Open-source under MIT license             | [YOLOv5 GitHub](https://github.com/ultralytics/yolov5) |
| **ResNet-50**       | Microsoft Research             | Open model weights available              | [ResNet-50 on PyTorch Hub](https://pytorch.org/vision/stable/models.html#id14) |
| **GPT-2 Medium**    | OpenAI                         | Open weights with usage restrictions      | [GPT-2 (Hugging Face)](https://huggingface.co/gpt2-medium) |


### Infrastructure Requirements

| Requirement     | Quantity/Duration              | Justification                             |
|-----------------|-------------------------------|-------------------------------------------|
| `m1.medium` VMs | 3 for project duration        | Hosting API, MLFlow server, and data pipeline |
| `gpu_mi100`     | 4-hour blocks, twice weekly   | Model training and fine-tuning            |
| Floating IPs    | 2 permanent                   | Exposing services to external users       |
| Persistent Storage | 100GB                      | Storing datasets, model checkpoints, and logs |

### Detailed Design Plan

#### Model Training and Training Platforms

**Strategy**: Utilize Chameleon Cloud with NVIDIA Tesla V100 GPUs for efficient model training.

**Components**:
- **Object Detection**: YOLOv5 model.
- **Image Captioning**: Combination of ResNet-50 and GPT-2 Medium.

**Justification**: These models balance performance and resource utilization, aligning with project goals.

**Implementation Details**:
- **Training Techniques**: Gradient accumulation, LoRA fine-tuning, mixed-precision computations.
- **Hyperparameter Optimization**: Conducted using Ray Tune for optimal performance.
- **Experiment Tracking**: Managed with MLFlow, ensuring reproducibility and transparency.

**Lecture Alignment**: Incorporates concepts from [Unit 4](https://ffund.github.io/ml-sys-ops/docs/units/4-model-training-scale.html) and [Unit 5](https://ffund.github.io/ml-sys-ops/docs/units/5-training-platform.html), focusing on model development and optimization.

**Optional Difficulty Points**: Implementing distributed training and advanced hyperparameter tuning.

#### Model Serving and Monitoring Platforms

**Strategy**: Deploy models using FastAPI for asynchronous processing, optimized with ONNX and TensorRT.

**Performance Targets**:
- **Latency**: Under 200ms per image.
- **Throughput**: At least 10 images per second.
- **Concurrency**: Handle 50+ simultaneous requests.

**Monitoring**: Integrate Prometheus and Grafana for real-time performance tracking and alerting.

**Justification**: Ensures responsive and reliable service delivery.

**Lecture Alignment**: Addresses [Unit 6](https://ffund.github.io/ml-sys-ops/docs/units/6-serving.html) and [Unit 7](https://ffund.github.io/ml-sys-ops/docs/units/7-eval-monitor.html), covering deployment and monitoring.

**Optional Difficulty Points**: Implementing model quantization and real-time monitoring dashboards.

#### Data Pipeline

**Offline Pipeline**:
- **ETL Processes**: Efficient extraction, transformation, and loading of data.
- **Version Control**: Managed with DVC for dataset and model tracking.

**Online Pipeline**:
- **Feature Extraction**: Real-time processing of incoming data.
- **Preprocessing**: Standardized to maintain consistency.
- **Inference Logging**: Capturing results for continuous improvement.

**Justification**: Facilitates seamless data flow and model performance tracking.

**Lecture Alignment**: Pertains to [Unit 8](#) on data engineering.

**Optional Difficulty Points**: Real-time data processing and integration with CI/CD pipelines.

#### Continuous Integration and Deployment

**Infrastructure as Code**: Managed using python-chi, Ansible, and Docker for consistent environments.

**CI/CD Practices**:
- **Automated Testing**: Ensures code reliability.
- **Deployment Pipelines**: Streamlined with GitHub Actions.
- **Release Strategies**: Implementing canary releases and rollback capabilities.
- **Documentation**: Automated generation for maintainability.

**Justification**: Promotes development efficiency and system reliability.

**Lecture Alignment**: Relates to [Unit 3](https://ffund.github.io/ml-sys-ops/docs/units/3-devops.html) on software engineering practices.

**Optional Difficulty Points**: Implementing blue-green deployments and comprehensive test coverage.
