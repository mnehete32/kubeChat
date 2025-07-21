import os
import ast
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import mlflow
import evaluate
from datasets import Dataset, ClassLabel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from trl import SFTTrainer

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CURL_CA_BUNDLE'] = ''


# --- Configuration ---
@dataclass
class TrainingConfig:
    """Configuration class for the fine-tuning pipeline."""
    # File and- Model Paths
    training_dataset_path: str
    output_model_dir: str
    model_name: str = "unsloth/Llama-3.2-1B"
    test_model_name: str = "sshleifer/tiny-gpt2"

    # Training Hyperparameters
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # LoRA Configuration
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ])

    # Runtime Flags
    save_model: bool = False
    test_run: bool = False
    seed: int = 32
    
    # MLflow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/")
    mlflow_experiment_name: str = "kubeChat"

    # Dataset formatting
    order_of_text_list: List[str] = field(default_factory=lambda: [
        "objective", "chain_of_thought", "question"
    ])
    target_text_column : str = "command"
    
    @classmethod
    def from_args(cls):
        """Creates a config instance from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a Causal LM with QLoRA and MLflow.")
        parser.add_argument("--training_dataset_path", type=str, required=True)
        parser.add_argument("--output_model_dir", type=str, required=True)
        parser.add_argument("--save_model", action="store_true", help="Save the final model artifacts.")
        parser.add_argument("--test_run", action="store_true", help="Run a fast test with a tiny model and dataset.")
        parser.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension.")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
        parser.add_argument("--lora_target_modules", type=str, default='["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]', help="Modules to apply LoRA to.")
        
        args = parser.parse_args()
        
        return cls(
            training_dataset_path=args.training_dataset_path,
            output_model_dir=args.output_model_dir,
            save_model=args.save_model,
            test_run=args.test_run,
            lora_r=args.lora_r,
            lora_dropout=args.lora_dropout,
            lora_target_modules=ast.literal_eval(args.lora_target_modules)
        )


# --- Metrics and Callbacks ---
class MetricsComputer:
    """A class to compute and handle evaluation metrics."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

    def compute(self, eval_preds) -> Dict[str, float]:
        """Computes BLEU and ROUGE scores from model predictions."""
        logits, labels = eval_preds
        # In causal LM, logits are shifted, so we use argmax on them directly.
        predictions = np.argmax(logits, axis=-1)

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculate scores
        rouge_result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bleu_result = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": bleu_result["bleu"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
        }


class MLflowLoggingCallback(TrainerCallback):
    """Logs metrics to MLflow during training."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


# --- Core Components ---
class DataProcessor:
    """Handles loading, preprocessing, and tokenizing the dataset."""
    def __init__(self, config: TrainingConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def _format_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a single data example into a structured string."""
        final_str = ""
        for key in self.config.order_of_text_list:
            content = example.get(key)
            if content:
                final_str += f"### {key}:\n{content}\n"
        example["prompt"] = final_str
        example["response"] = example[self.config.target_text_column]
        return example

    def _tokenize(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        """Tokenizes a formatted text example."""
        formatted_example = self._format_text(example)
        return self.tokenizer(
            formatted_example["prompt"],
            text_target=example["response"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    def prepare_datasets(self) -> (Dataset, Dataset):
        """Loads, splits, and tokenizes the dataset."""
        logging.info(f"Loading dataset from {self.config.training_dataset_path}")
        df = pd.read_parquet(self.config.training_dataset_path)
        if self.config.test_run:
            df = df.head(20)

        dataset = Dataset.from_pandas(df)
        
        # Stratified split and tokenization
        split_dataset = dataset.train_test_split(test_size=0.2, seed=self.config.seed)
        
        logging.info("Tokenizing datasets...")
        train_dataset = split_dataset["train"].map(self._tokenize, batched=False)
        val_dataset = split_dataset["test"].map(self._tokenize, batched=False)

        return train_dataset, val_dataset


class ModelManager:
    """Handles model and tokenizer loading, including quantization and PEFT setup."""
    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Loads the tokenizer for the specified model."""
        model_id = self.config.test_model_name if self.config.test_run else self.config.model_name
        logging.info(f"Loading tokenizer for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self) -> PreTrainedModel:
        """Loads the model, applying quantization and LoRA if not in test mode."""
        if self.config.test_run:
            logging.info(f"Loading test model: {self.config.test_model_name}")
            return AutoModelForCausalLM.from_pretrained(self.config.test_model_name)

        logging.info(f"Loading 4-bit quantized model: {self.config.model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        logging.info("Applying PEFT LoRA configuration...")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model


class FineTuningPipeline:
    """Orchestrates the entire fine-tuning process."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        set_seed(self.config.seed)
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initializes MLflow tracking."""
        logging.info(f"Setting MLflow tracking URI to: {self.config.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def run(self):
        """Executes the fine-tuning pipeline from start to finish."""
        run_name = "finetuning_test" if self.config.test_run else "finetuning_llama3"
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logging.info(f"🚀 Starting MLflow Run: {run_name} ({run_id})")

            # 1. Load Model and Tokenizer
            model_manager = ModelManager(self.config)
            tokenizer = model_manager.load_tokenizer()
            model = model_manager.load_model()
            
            # 2. Prepare Datasets
            data_processor = DataProcessor(self.config, tokenizer)
            train_dataset, val_dataset = data_processor.prepare_datasets()

            # 3. Configure Trainer
            model_save_dir = f"./data/model/{run_id}"
            training_args = TrainingArguments(
                output_dir=model_save_dir,
                per_device_train_batch_size=1 if self.config.test_run else self.config.per_device_train_batch_size,
                gradient_accumulation_steps=1 if self.config.test_run else self.config.gradient_accumulation_steps,
                num_train_epochs=1 if self.config.test_run else self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                fp16=not self.config.test_run,
                logging_steps=5,
                save_strategy="epoch" if not self.config.test_run else "no",
                eval_strategy="epoch",
                report_to="none", # MLflow callback will handle reporting
            )
            
            metrics_computer = MetricsComputer(tokenizer)

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=metrics_computer.compute,
                callbacks=[MLflowLoggingCallback()],
                # max_seq_length=128,
            )

            # 4. Train and Evaluate
            logging.info("Starting model training...")
            trainer.train()
            
            logging.info("Evaluating model...")
            metrics = trainer.evaluate()
            logging.info(f"Evaluation metrics: {metrics}")
            mlflow.log_metrics({k.replace('eval_', ''): v for k, v in metrics.items()})

            # 5. Log parameters and save artifacts
            self._log_params()
            if self.config.save_model:
                self._save_artifacts(model, tokenizer, model_save_dir, run_id)
            
            # Adding print to capture by katib for hyperparameter tunning
            print(f"eval-bleu={metrics['eval_bleu']}")
            logging.info(f"✅ Training complete. Run ID: {run_id}")


    def _log_params(self):
        """Logs hyperparameters to MLflow."""
        params_to_log = {
            "model_name": self.config.model_name,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.num_train_epochs,
            "batch_size": self.config.per_device_train_batch_size,
            "lora_r": self.config.lora_r,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": str(self.config.lora_target_modules),
        }
        mlflow.log_params(params_to_log)
        logging.info("Logged hyperparameters to MLflow.")

    def _save_artifacts(self, model, tokenizer, save_dir, run_id):
        """Saves model, tokenizer, and artifact URI."""
        logging.info(f"Saving model and tokenizer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        artifact_path = f"model_{run_id}"
        mlflow.log_artifacts(save_dir, artifact_path=artifact_path)
        model_artifact_uri = mlflow.get_artifact_uri(artifact_path)
        
        os.makedirs(os.path.dirname(self.config.output_model_dir), exist_ok=True)
        with open(self.config.output_model_dir, "w") as f:
            f.write(model_artifact_uri)
        logging.info(f"Model artifact URI saved to {self.config.output_model_dir}")


def main():
    """Main execution function."""
    try:
        config = TrainingConfig.from_args()
        pipeline = FineTuningPipeline(config)
        pipeline.run()
    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
        # exc_info=True will log the full stack trace

if __name__ == "__main__":
    main()