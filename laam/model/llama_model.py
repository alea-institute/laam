"""Llama model class for training and inference on LaaM data."""

# imports

# packages
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

# laam imports
from laam.model.tokenizer import get_tokenizer, PAD_TOKEN
from laam.data.kernel_source import get_linux_sources, START_FILE_TOKEN, END_FILE_TOKEN


def init_model(
    hidden_size: int = 128,
    intermediate_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 1024,
    tie_word_embeddings: bool = False,
    precision: torch.dtype = torch.float32,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> LlamaForCausalLM:
    """Initialize a new LLaMA model.

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 128.
        intermediate_size (int, optional): The intermediate size of the model. Defaults to 256.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 4.
        num_attention_heads (int, optional): The number of attention heads in the model. Defaults to 4.
        num_key_value_heads (int, optional): The number of key value heads in the model. Defaults to 4.
        max_position_embeddings (int, optional): The maximum position embeddings in the model. Defaults to 1024.
        tie_word_embeddings (bool, optional): Whether to tie the word embeddings. Defaults to True.
        precision (torch.dtype, optional): The precision of the model. Defaults to torch.float32.
        tokenizer (PreTrainedTokenizerFast, optional): The tokenizer to use. Defaults to get_tokenizer().

    Returns:
        LlamaForCausalLM: The initialized LLaMA model.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # bos token = START_FILE_TOKEN
    tokenizer.bos_token = START_FILE_TOKEN
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(START_FILE_TOKEN)

    # eos token = END_FILE_TOKEN
    tokenizer.eos_token = END_FILE_TOKEN
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(END_FILE_TOKEN)

    # pad token = PAD_TOKEN
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

    # get vocab size directly
    vocab_size = len(tokenizer.get_vocab())

    # initialize the configuration
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=precision,
    )

    # set the attention implementation
    if torch.cuda.is_available():
        config._attn_implementation = "flash_attention_2"

    # initialize the model
    print("Model precision:", precision)
    model = LlamaForCausalLM(
        config,
    ).to(precision)

    return model


def reload_model(model_path: str, precision: torch.dtype = torch.float32) -> LlamaForCausalLM:
    """Reload a LLaMA model.

    Args:
        model_path (str): The path to the model.
        precision (torch.dtype, optional): The precision of the model. Defaults to torch.float32.

    Returns:
        LlamaForCausalLM: The reloaded LLaMA model.
    """
    # load the model
    model = LlamaForCausalLM.from_pretrained(model_path)

    # set the precision
    model = model.to(precision)

    return model
