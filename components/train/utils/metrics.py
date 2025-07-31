import numpy as np
import evaluate
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Any

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