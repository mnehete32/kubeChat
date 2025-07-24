<h1 style="text-align:center;">KubeChat</h1>
<h2 style="text-align:center;">Conversational Kubernetes Management via Fine-Tuned LLMs</h2>
<p style="text-align:center;"> KubeChat is an advanced, MLOps-powered solution that transforms Kubernetes cluster management using natural language. By fine-tuning a Large Language Model (LLM) to translate user queries into `kubectl` commands, KubeChat enables intuitive, conversational interaction with Kubernetes</p>

<p style="text-align:center;">The project is structured as a comprehensive MLOps solution, leveraging Kubeflow Pipelines (KFP) for orchestrating the entire LLM fine-tuning lifecycle, from data preparation and model training to evaluation. It incorporates best practices for machine learning operations, including modular components, hyperparameter tuning with Katib, and experiment tracking with MLflow.</p>


## Tools

| Tool                   | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| **Kubeflow Pipelines** | Orchestration of end-to-end ML workflow  |
| **Katib**              | Hyperparameter tuning for LLM training   |
| **MLflow**             | Experiment tracking and model versioning |
| **Docker**             | Containerization of modular components   |
| **Kubernetes**         | Native deployment of the platform        |



## Key Features
1. Natural Language to kubectl Command Translation: Converts user queries.

2. Automated LLM Fine-tuning Pipeline: Utilizes Kubeflow Pipelines (pipeline.py) to automate the entire process of preparing data, training, and evaluating the LLM, ensuring reproducibility and scalability.

3. Modular Component Design: The project is broken down into distinct, reusable components (e.g., data_prep, download_dataset, train_test_split, training, testing), each with its own Dockerfile and dependencies, facilitating independent development and deployment.

4. Hyperparameter Tuning with Katib: Integrates Kubeflow Katib within the training component to enable automated hyperparameter optimization for the LLM, improving model performance.

5. MLflow Integration: Includes configurations for MLflow, allowing for robust experiment tracking, model versioning, and artifact management.

6. Kubernetes-Native Deployment: Designed for seamless deployment and operation within a Kubernetes environment, with dedicated kube manifests for various cluster configurations, storage, and service accounts.

7. Python-based Client for KFP: Provides a Python client [pipeline.py](pipeline.py) to programmatically interact with Kubeflow Pipelines, enabling easy pipeline submission and management.

## Environment Setup

Follow instructions in [setup.md](setup.md) to configure your environment, dependencies, and cluster.