import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from datasets import Dataset
from utils.config import TrainingConfig
from utils.prompt import PROMPT
from transformers import PreTrainedTokenizerBase

class DataPreprocessor:
    """Handles loading, preprocessing, and tokenizing the dataset."""
    def __init__(self, config: TrainingConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def _format_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a single data example into a structured string."""
        example["text"] = PROMPT.format(
                                    prompt=example[self.config.prompt_text_column], 
                                    objective=example["objective"],
                                    thought=example["chain_of_thought"],
                                    syntax=example["syntax"],
                                    command=example["command"])
        return example

    def _tokenize(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        """Tokenizes a formatted text example."""
        formatted_example = self._format_text(example)
        return self.tokenizer(
            formatted_example["text"],
            # text_target=example["response"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    def prepare_datasets(self) -> Tuple["Dataset", "Dataset"]:
        """Loads, splits, and tokenizes the dataset."""
        logging.info(f"Loading dataset from {self.config.train_dataset_path}")
        df = pd.read_parquet(self.config.train_dataset_path)
        if self.config.test_run:
            df = df.head(20)

        dataset = Dataset.from_pandas(df)
        
        # Stratified split and tokenization
        split_dataset = dataset.train_test_split(test_size=0.2, seed=self.config.seed)
        
        logging.info("Tokenizing datasets...")
        train_dataset = split_dataset["train"].map(self._tokenize, batched=False)
        val_dataset = split_dataset["test"].map(self._tokenize, batched=False)

        return train_dataset, val_dataset