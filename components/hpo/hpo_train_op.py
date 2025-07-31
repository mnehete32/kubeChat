from kfp.v2.dsl import component

@component
def create_hpo_experiment(
    experiment_name: str,
    namespace: str,
    training_dataset_path: str,
    output_model_dir: str,
    base_yaml_path: str,
    image: str
) -> str:
    """Component to create Katib experiment with pipeline inputs.
    
    Args:
        experiment_name: Name for the Katib experiment
        namespace: Kubernetes namespace
        training_dataset_path: Path to training dataset (from pipeline)
        output_model_dir: Output directory for models (from pipeline)
        base_yaml_path: Path to base YAML template
    """
    import time
    import yaml
    from kubernetes import client, config

    # Load in-cluster config
    config.load_incluster_config()

    # Read base YAML
    # base_yaml_path is path to katib.yaml file in pipeline container
    with open(base_yaml_path, "r") as f:
        experiment_config = yaml.safe_load(f)

    # Update metadata
    experiment_config["metadata"]["name"] = experiment_name
    experiment_config["metadata"]["namespace"] = namespace

    # Patch trial container command args with pipeline inputs
    trial_spec = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                    "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": {
                    "serviceAccountName": "default-editor",
                    "containers": [
                        {
                            "name": "training-container",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python", 
                                "/app/train.py",
                                f"--training_dataset_path={training_dataset_path}",
                                f"--output_model_dir={output_model_dir}",
                                "--lora_r=${trialParameters.loraR}",
                                # "--lora_alpha=${trialParameters.loraAlpha}",
                                "--lora_dropout=${trialParameters.loraDropout}",
                                # "--lora_target_modules=${trialParameters.loraTargetModules}",
                                "--test_run",
                            ],
                            "envFrom": [
                                {
                                    "configMapRef": {
                                    "name": "common-config"
                                    }
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "mountPath": "/data",
                                    "name": "shared-storage"
                                }
                            ],
                            "resources": {   
                                "requests": {
                                    "nvidia.com/gpu": "1",
                                },

                                "limits":{
                                    "nvidia.com/gpu": "1",
                                    }
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "shared-storage",
                            "persistentVolumeClaim": {
                            "claimName": "shared-pvc"
                            },
                        }
                    ],
                    "restartPolicy": "Never",
                    "tolerations": [
                        {   
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ]
                        
                }
            }
        }
    }
    experiment_config["spec"]["trialTemplate"]["trialSpec"] = trial_spec

    # API client
    api = client.CustomObjectsApi()

    # Create Experiment CR
    api.create_namespaced_custom_object(
        group="kubeflow.org",
        version="v1beta1",
        namespace=namespace,
        plural="experiments",
        body=experiment_config,
    )

    # Poll for completion
    while True:
        status = api.get_namespaced_custom_object_status(
            group="kubeflow.org", version="v1beta1", namespace=namespace,
            plural="experiments", name=experiment_name
        )
        conditions = status.get("status", {}).get("conditions", [])
        if any(c.get("type") == "Succeeded" and c.get("status") == "True" for c in conditions):
            break
        if any(c.get("type") == "Failed" and c.get("status") == "True" for c in conditions):
            raise RuntimeError(f"Katib experiment {experiment_name} failed.")
        time.sleep(15)

    # Extract best trial parameters
    optimal = status["status"]["currentOptimalTrial"]["parameterAssignments"]
    # Return JSON string
    import json
    return json.dumps(optimal)