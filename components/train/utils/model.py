import logging
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from utils.config import TrainingConfig


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
            self.config.model_name, #if self.config.resume_from_checkpoint is None else self.config.resume_from_checkpoint,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        
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