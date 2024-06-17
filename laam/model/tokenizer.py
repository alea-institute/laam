"""Tokenizer module for LaaM model."""

# imports
from pathlib import Path

# packages
import tokenizers
import transformers

# constants
DEFAULT_TOKENIZER_PATH = Path("tokenizer/")
PAD_TOKEN = "<|pad|>"


def init_tokenizer() -> tokenizers.Tokenizer:
    """Initialize a new tokenizer.

    Returns:
        tokenizers.Tokenizer: The initialized tokenizer.
    """
    # initialize the tokenizer
    tokenizer_instance = tokenizers.ByteLevelBPETokenizer(
        add_prefix_space=True, lowercase=False, trim_offsets=False
    )

    # return the tokenizer
    return tokenizer_instance


def get_tokenizer(
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> transformers.PreTrainedTokenizerFast:
    """Get tokenizer for LaaM model.

    Args:
        tokenizer_path (str | Path | None, optional): Path to tokenizer. Defaults to None.

    Returns:
        transformers.PreTrainedTokenizerFast: Tokenizer for LaaM model.
    """
    # make sure we have an absolute path
    if isinstance(tokenizer_path, str):
        tokenizer_path = Path(tokenizer_path)

    # load it
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path.resolve().absolute()
    )

    return tokenizer
