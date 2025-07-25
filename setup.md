# Setup Guide for KubeChat

This guide outlines the environment setup, tool installation, and configuration steps required to run the **KubeChat** project

## Prerequisites

Ensure the following tools are installed on your system:

1. [Docker](https://www.docker.com/)
2. [Kubernetes](https://kubernetes.io/)
3. [Kubeflow](https://github.com/kubeflow/manifests)
4. [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)

## Install NGINX Ingress Controller


To expose **Kubeflow** and **MLflow** via custom domain URLs, install the **NGINX Ingress Controller** using the official documentation:

[Install NGINX Ingress](https://kubernetes.github.io/ingress-nginx/deploy/)

Once installed, you will be able to access:

- [http://kubeflow.nehete.com/](http://kubeflow.nehete.com/)
- [http://mlflow.nehete.com/](http://mlflow.nehete.com/)

## KFP Client Login Manager

KubeChat uses a customized Python KFP client manager to authenticate and submit pipelines to the Kubeflow Pipelines API server using Dex login.

Reference: [KFP Client Auth Guide](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/connect-api/)

> Already implemented in: [`kfp_client/kfp_client_manager.py`](kfp_client/kfp_client_manager.py)


### Steps to run the project
### 1. Fork & Setup GitHub Self-Hosted Runner
- Fork the repository.
- Navigate to: `Settings > Actions > Runners`.
- Add and install a self-hosted runner on your machine where Docker, Kubernetes, and Kubeflow are installed.

### 2. Create `kubechat` User Profile in Kubeflow

Run the following command:

```bash
kubectl apply -k kube/profile.yaml
```
Edit the Dex configmap to add static credentials:


```bash
kubectl edit configmap dex -n auth -o yaml
```
Append the following to staticPasswords:


```yaml
- email: mayur@nehete.com
  hash: $2y$10$4tCMuyoSo76v/HZ4cVANQ.lZIEH3e/3k8PRdT.06boH1qBA2Gialu
  username: mayur
```

Add below details to login using dex
- Email: `mayur@nehete.com`
- Password: `password`

### 3. Configure Storage Path
Update the path at [Storage.yaml](kube/storage.yaml#13) to store all the files in persistent volume.

### 4. Trigger the ML Pipeline
The pipeline is automatically triggered on each commit to the master branch.

To manually trigger:
```bash
git commit --allow-empty -m "trigger pipeline"
git push
```


## Accessing Kubeflow & MLflow UIs
### Option 1: Using Custom Domains (Ingress Setup Required)
Edit your /etc/hosts file to map domain names:
```bash
127.0.0.1 kubeflow.nehete.com
127.0.0.1 mlflow.nehete.com
```
If using Minikube, get the IP with:
```bash
minikube ip
```

Example:
```
192.168.49.2 kubeflow.nehete.com
192.168.49.2 mlflow.nehete.com
```


### Option 2: Port Forward (No Ingress Required)

1. Access Kubeflow UI
```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```
Now open 
[http://localhost:8080/](http://localhost:8080/)

2. Access MLflow UI
```bash
kubectl port-forward svc/mlflow-service -n kubechat 5555:80
```
Now open[http://localhost:5555/](http://localhost:5555/)