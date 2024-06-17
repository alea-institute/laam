"""Train a Llama2 model from Linux sources."""

# imports
import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

# set the environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# packages
import accelerate
import torch.utils.data
import wandb
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# laam imports
from laam.model.llama_model import init_model, reload_model
from laam.model.tokenizer import get_tokenizer, DEFAULT_TOKENIZER_PATH
from laam.data.kernel_source import get_linux_sources, KernelSourceDataset

# setup default logger with timestamps and console output
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

# init the accelerator
accelerator = accelerate.Accelerator(log_with="wandb")
accelerator.init_trackers(
    project_name="laam",
)

if __name__ == "__main__":
    # set up arg parser for these:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", type=str, default="v1.0/1.0")
    arg_parser.add_argument("--batch_size", type=int, default=1)
    arg_parser.add_argument("--hidden_size", type=int, default=128)
    arg_parser.add_argument("--intermediate_size", type=int, default=256)
    arg_parser.add_argument("--num_hidden_layers", type=int, default=4)
    arg_parser.add_argument("--num_attention_heads", type=int, default=4)
    arg_parser.add_argument("--num_key_value_heads", type=int, default=4)
    arg_parser.add_argument("--max_position_embeddings", type=int, default=1024)
    arg_parser.add_argument("--precision", type=str, default="float32")
    arg_parser.add_argument("--learning_rate", type=float, default=1e-3)
    arg_parser.add_argument("--decay_rate", type=float, default=0.1)
    arg_parser.add_argument("--warmup_steps", type=int, default=100)
    arg_parser.add_argument("--logging_steps", type=int, default=10)
    arg_parser.add_argument("--epochs", type=int, default=3)
    arg_parser.add_argument(
        "--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH
    )
    arg_parser.add_argument("--output_path", type=str, default="models/llama2")
    arg_parser.add_argument("--reload", action="store_true")
    args = arg_parser.parse_args()

    # get the output path as path
    output_path = Path(args.output_path)

    # handle precision
    if args.precision == "float32":
        precision = torch.float32
    elif args.precision == "float16":
        precision = torch.float16
    elif args.precision == "bfloat16":
        precision = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")

    if args.reload:
        LOGGER.info("Reloading tokenizer...")
        tokenizer = get_tokenizer(tokenizer_path=args.output_path)

        LOGGER.info("Reloading model...")
        model = reload_model(args.output_path)
    else:
        # get the tokenizer
        LOGGER.info("Initializing tokenizer...")
        tokenizer = get_tokenizer(tokenizer_path=args.tokenizer_path)

        # get a model
        LOGGER.info("Initializing model...")
        model = init_model(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_position_embeddings,
            precision=precision,
            tokenizer=tokenizer,
        )

    # get model size in millions as a string with one digit precision
    model_size = sum(l.numel() for l in model.parameters())
    model_size_string = f"{model_size / 1e6:.1f}M"
    LOGGER.info(f"Model: {model_size_string}")

    # setup the dataset
    LOGGER.info("Initializing dataset...")
    kernel_dataset = KernelSourceDataset(
        version=args.version,
        tokenizer=tokenizer,
        max_tokens=model.config.max_position_embeddings
    )
    LOGGER.info(f"Dataset size: tokens={kernel_dataset.num_tokens}, records={kernel_dataset.num_records}")

    # create a torch dataloader
    LOGGER.info("Initializing dataloader...")
    kernel_dataloader = torch.utils.data.DataLoader(
        kernel_dataset, batch_size=args.batch_size
    )

    # get the training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path.resolve().absolute()),
        learning_rate=args.learning_rate,
        weight_decay=args.decay_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        save_steps=1_000,
        save_total_limit=3,
        bf16=precision == torch.bfloat16,
        fp16=precision == torch.float16,
    )

    # get the trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=kernel_dataset,
        )

        # train the model
        LOGGER.info("Training model...")
        trainer.train()

        # save the model
        LOGGER.info("Saving model...")
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)

        LOGGER.info("Training complete.")
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted.")
    except Exception as e:
        LOGGER.error(f"Training failed: {e}")
    finally:
        # finish wandb
        accelerator.end_training()
