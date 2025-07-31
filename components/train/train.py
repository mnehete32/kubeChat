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

from transformers.trainer_utils import set_seed
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
import torch.serialization
import numpy.core.multiarray
import codecs

from utils.config import TrainingConfig
from utils.metrics import MetricsComputer
from utils.preprocess_data import DataPreprocessor
from utils.model import ModelManager




# refer
# https://github.com/huggingface/transformers/commit/1339a14dca0c633f74bc8fb771aa8a651dd472b0
allowlist = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, codecs.encode]
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)
# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CURL_CA_BUNDLE'] = ''

class MLflowLoggingCallback(TrainerCallback):
    """Logs metrics to MLflow during training."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


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
            data_processor = DataPreprocessor(self.config, tokenizer)
            train_dataset, val_dataset = data_processor.prepare_datasets()

            # 3. Configure Trainer
            model_save_dir = f"./data/model/{run_id}"
            training_args = TrainingArguments(
                output_dir=model_save_dir,
                per_device_train_batch_size=1 if self.config.test_run else self.config.per_device_train_batch_size,
                per_device_eval_batch_size=1 if self.config.test_run else self.config.per_device_train_batch_size,
                gradient_accumulation_steps=1 if self.config.test_run else self.config.gradient_accumulation_steps,
                num_train_epochs=1 if self.config.test_run else self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                weight_decay=0.01,
                fp16=not self.config.test_run,
                logging_steps=1,
                save_strategy="steps" if not self.config.test_run else "no",
                save_steps=40,
                eval_strategy="epoch",
                # eval_steps=10,
                optim=self.config.optimizer,
                report_to="none", # MLflow callback will handle reporting
                # resume_from_checkpoint=self.config.resume_from_checkpoint
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
            if self.config.resume_from_checkpoint:
                logging.info(f"Loading checkpoint: {self.config.resume_from_checkpoint}")
                trainer.train(self.config.resume_from_checkpoint)
            else:
                trainer.train()
            
            logging.info("Evaluating model...")
            metrics = trainer.evaluate()
            logging.info(f"Evaluation metrics: {metrics}")
            mlflow.log_metrics({k.replace('eval_', ''): v for k, v in metrics.items()})

            # 5. Log parameters and save artifacts
            self._log_params()
            if self.config.train:
                self._save_artifacts(model, tokenizer, model_save_dir, run_id)
            
            # Adding print to capture by katib for hyperparameter tunning
            print(f"eval-bleu={metrics['eval_bleu']}")
            logging.info(f"Training complete. Run ID: {run_id}")


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

if __name__ == "__main__":
    main()