from kfp.v2.dsl import component

@component(base_image="python:3.11.4-slim-buster", packages_to_install=["kubernetes"])
def create_katib_experiment(
    experiment_name: str,
    namespace: str,
    training_dataset_path: str,
    output_model_dir: str,
    base_yaml_path: str,
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
    with open(base_yaml_path, 'r') as f:
        experiment_config = yaml.safe_load(f)

    # Update metadata
    experiment_config['metadata']['name'] = experiment_name
    experiment_config['metadata']['namespace'] = namespace

    # Patch trial container command args with pipeline inputs
    trial_template = experiment_config['spec']['trialTemplate']
    containers = trial_template['trialSpec']['spec']['template']['spec']['containers']
    for c in containers:
        if c.get('name') == 'training-container':
            c['command'] = [
                'python', '/app/train.py',
                f'--training_dataset_path={training_dataset_path}',
                f'--output_model_dir={output_model_dir}',
                '--lora_r=${trialParameters.loraR}',
                # "--lora_alpha=${trialParameters.loraAlpha}",
                '--lora_dropout=${trialParameters.loraDropout}',
                # "--lora_target_modules=${trialParameters.loraTargetModules}",
                '--test_run',
            ]
    trial_template['trialSpec']['spec']['template']['spec']['containers'] = containers
    experiment_config['spec']['trialTemplate'] = trial_template

    # API client
    api = client.CustomObjectsApi()

    # Create Experiment CR
    api.create_namespaced_custom_object(
        group='kubeflow.org',
        version='v1beta1',
        namespace=namespace,
        plural='experiments',
        body=experiment_config,
    )

    # Poll for completion
    while True:
        status = api.get_namespaced_custom_object_status(
            group='kubeflow.org', version='v1beta1', namespace=namespace,
            plural='experiments', name=experiment_name
        )
        conditions = status.get('status', {}).get('conditions', [])
        if any(c.get('type') == 'Succeeded' and c.get('status') == 'True' for c in conditions):
            break
        if any(c.get('type') == 'Failed' and c.get('status') == 'True' for c in conditions):
            raise RuntimeError(f"Katib experiment {experiment_name} failed.")
        time.sleep(15)

    # Extract best trial parameters
    optimal = status['status']['currentOptimalTrial']['parameterAssignments']
    # Return JSON string
    import json
    return json.dumps(optimal)