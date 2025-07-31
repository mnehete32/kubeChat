import os
import argparse
from dataclasses import dataclass, field
from typing import List
import ast

@dataclass
class TrainingConfig:
    """Configuration class for the fine-tuning pipeline."""
    # File and- Model Paths
    training_dataset_path: str
    output_model_dir: str
    # model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    test_model_name: str = "sshleifer/tiny-gpt2"

    # Training Hyperparameters
    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32

    optimizer: str = "adamw_8bit"
    
    # LoRA Configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ])

    # Runtime Flags
    training: bool = False
    test_run: bool = False
    resume_from_checkpoint: str = None
    seed: int = 32
    
    # MLflow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/")
    mlflow_experiment_name: str = "kubeChat"

    # Dataset formatting
    order_of_response_list: List[str] = field(default_factory=lambda: [
        "objective", "chain_of_thought", "syntax","command"
    ])
    prompt_text_column : str = "question"
    
    @classmethod
    def from_args(cls):
        """Creates a config instance from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a Causal LM with QLoRA and MLflow.")
        parser.add_argument("--training_dataset_path", type=str, required=True)
        parser.add_argument("--output_model_dir", type=str, required=True)
        parser.add_argument("--training", action="store_true", help="Use this flag to mention model is training")
        parser.add_argument("--hpo", action="store_true", help="Use this flag to script is run to do hpo")
        parser.add_argument("--test_run", action="store_true", help="Run a fast test with a tiny model and dataset.")
        parser.add_argument("--lora_r", type=int, default=32,help="LoRA attention dimension.")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
        parser.add_argument("--lora_target_modules", type=str, default='["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]', help="Modules to apply LoRA to.")
        parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to a saved model directory to resume training from.")

        args = parser.parse_args()
        
        return cls(
            training_dataset_path=args.training_dataset_path,
            output_model_dir=args.output_model_dir,
            training=args.training,
            test_run=args.test_run,
            lora_r=args.lora_r,
            lora_dropout=args.lora_dropout,
            lora_target_modules=ast.literal_eval(args.lora_target_modules),
            resume_from_checkpoint=args.resume_from_checkpoint,


        )