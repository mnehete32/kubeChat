import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from evaluate import load as load_metric

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
@dataclass
class EvaluationConfig:
    """Configuration for the LLM evaluation pipeline."""
    model_path: str
    dataset_path: str
    output_dir: str
    
    # Dataset column mapping
    prompt_template_columns: List[str] = field(default_factory=lambda: ["objective", "chain_of_thought", "question"])
    reference_answer_column: str = "command"
    question_column: str = "question"

    # Runtime flags
    test_run: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MLflow settings
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/")
    mlflow_experiment_name: str = "kubeChat-Evaluation"

    @classmethod
    def from_args(cls):
        """Creates a config instance from command-line arguments."""
        parser = argparse.ArgumentParser(description="Evaluate a Question-Answering LLM.")
        parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model.")
        parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset (Parquet).")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results.")
        parser.add_argument("--reference_answer_column", type=str, default="command", help="Column with the ground-truth answers.")
        parser.add_argument("--question_column", type=str, default="question", help="Column containing the user question.")
        parser.add_argument("--test_run", action="store_true", help="Run a fast test on a small subset of data.")
        args = parser.parse_args()
        
        return cls(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            reference_answer_column=args.reference_answer_column,
            question_column=args.question_column,
            test_run=args.test_run,
        )

# --- Core Components ---
class DataLoader:
    """Handles loading and preparing evaluation data."""
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Formats a data row into a single prompt string for the model."""
        prompt = ""
        for key in self.config.prompt_template_columns:
            content = example.get(key)
            if content:
                prompt += f"### {key}:\n{content}\n"
        # prompt += f"### {self.config.reference_answer_column}:\n" # Prompt for the answer
        return prompt

    def get_data(self) -> (List[str], List[str], List[str]):
        """Loads data and returns lists of prompts, reference answers, and questions."""
        logging.info(f"Loading dataset from {self.config.dataset_path}")
        df = pd.read_parquet(self.config.dataset_path)
        if self.config.test_run:
            df = df.head(20)

        prompts = [self._format_prompt(row) for _, row in df.iterrows()]
        reference_answers = df[self.config.reference_answer_column].tolist()
        questions = df[self.config.question_column].tolist()
        
        logging.info(f"Loaded {len(prompts)} records for evaluation.")
        return prompts, reference_answers, questions

class ModelPredictor:
    """Wraps the Hugging Face model and tokenizer for efficient batch prediction."""
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path)
        self.model.to(self.device)

    def _load_model_and_tokenizer(self, model_path: str) -> (PreTrainedModel, PreTrainedTokenizerBase):
        """Loads the model and tokenizer from the specified path."""
        logging.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        return model, tokenizer

    def predict(self, prompts: List[str], responses: List[str], batch_size: int = 8) -> List[str]:
        """Generates predictions for a list of prompts in batches."""
        predictions = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Predictions"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, text_target=responses, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(self.device)
            
            # Generate output, ensuring we only decode the newly generated tokens
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.pad_token_id
            )
            # Slice the output to get only the generated part
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            batch_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
        return predictions

class MetricsCalculator:
    """Computes all relevant evaluation metrics for QA tasks."""
    def __init__(self, questions: List[str], reference_answers: List[str], predicted_answers: List[str]):
        self.questions = questions
        self.preds = predicted_answers
        self.refs = reference_answers
        self.rouge = Rouge()
        self.bertscore_metric = load_metric("bertscore")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _compute_rouge(self) -> Dict[str, Any]:
        try:
            return self.rouge.get_scores(self.preds, self.refs, avg=True)
        except ValueError:
            logging.warning("ROUGE calculation failed. Skipping.")
            return {}

    def _compute_meteor(self) -> Dict[str, float]:
        scores = [meteor_score([ref.split()], pred.split()) for pred, ref in zip(self.preds, self.refs)]
        return {"meteor": np.mean(scores).item()}

    def _compute_bertscore(self) -> Dict[str, float]:
        results = self.bertscore_metric.compute(predictions=self.preds, references=self.refs, lang="en")
        return {
            "bertscore_precision": np.mean(results["precision"]).item(),
            "bertscore_recall": np.mean(results["recall"]).item(),
            "bertscore_f1": np.mean(results["f1"]).item(),
        }

    def _compute_semantic_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Helper to compute cosine similarity between two lists of strings."""
        emb1 = self.sbert_model.encode(list1, convert_to_tensor=True)
        emb2 = self.sbert_model.encode(list2, convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(emb1, emb2).mean().item()

    def calculate_all(self) -> Dict[str, Any]:
        """Calculates and consolidates all QA metrics."""
        logging.info("Calculating evaluation metrics...")
        metrics = {}
        metrics.update(self._compute_rouge())
        metrics.update(self._compute_meteor())
        metrics.update(self._compute_bertscore())
        
        # QA-specific semantic metrics
        metrics["semantic_answer_similarity"] = self._compute_semantic_similarity(self.refs, self.preds)
        metrics["answer_relevancy"] = self._compute_semantic_similarity(self.questions, self.preds)
        
        return metrics

class EvaluationPipeline:
    """Orchestrates the entire model evaluation process."""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initializes MLflow tracking."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        
    def run(self):
        """Executes the evaluation pipeline."""
        with mlflow.start_run(run_name=f"eval-{os.path.basename(self.config.model_path)}") as run:
            run_id = run.info.run_id
            logging.info(f"🚀 Starting MLflow Evaluation Run: {run_id}")
            mlflow.log_params({"model_path": self.config.model_path, "dataset": self.config.dataset_path})
            
            # 1. Load Data
            data_loader = DataLoader(self.config)
            prompts, references, questions = data_loader.get_data()
            
            # 2. Generate Predictions
            predictor = ModelPredictor(self.config.model_path, self.config.device)
            predictions = predictor.predict(prompts, references)
            
            # 3. Compute Metrics
            calculator = MetricsCalculator(questions, references, predictions)
            metrics = calculator.calculate_all()
            
            # 4. Log and Save Results
            self._log_and_save(run_id, predictions, references, metrics)
            logging.info(f"✅ Evaluation complete. Metrics: {json.dumps(metrics, indent=2)}")

    def _log_and_save(self, run_id: str, predictions: List[str], references: List[str], metrics: Dict):
        """Logs metrics to MLflow and saves artifacts."""
        # Log flattened metrics
        for key, value in metrics.items():
            if isinstance(value, dict): # Handle nested dicts from ROUGE
                for sub_key, sub_val in value.items():
                    mlflow.log_metric(f"rouge_{sub_key}", sub_val)
            else:
                mlflow.log_metric(key, value)
        
        # Prepare and save artifact file
        output_data = {
            "metrics": metrics,
            "evaluation_data": [
                {"prediction": p, "reference": r} for p, r in zip(predictions, references)
            ]
        }
        
        run_output_dir = os.path.join(self.config.output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        metrics_path = os.path.join(run_output_dir, "evaluation_results.json")
        
        with open(metrics_path, "w") as f:
            json.dump(output_data, f, indent=4)
            
        mlflow.log_artifact(metrics_path)
        logging.info(f"Results saved and logged to MLflow run {run_id}.")


if __name__ == "__main__":
    # NLTK data download for METEOR
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    config = EvaluationConfig.from_args()
    pipeline = EvaluationPipeline(config)
    pipeline.run()