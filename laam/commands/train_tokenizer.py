"""Train a tokenizer from the kernel sources."""

# imports
import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable


# set the environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# packages
import tokenizers

# laam imports
from laam.model.tokenizer import init_tokenizer, PAD_TOKEN
from laam.data.kernel_source import (
    get_linux_sources,
    START_FILE_TOKEN,
    END_FILE_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
    SPECIAL_TOKENS,
)


if __name__ == "__main__":
    # setup argparse: train_tokenizer.py [--version kernel_version] [--output_path tokenizer/]
    parser = argparse.ArgumentParser(
        description="Train a tokenizer from the kernel sources."
    )

    # add arguments
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0/1.0",
        help="The version of the linux kernel to download.",
    )

    # quiet for prog bars
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to suppress progress bars.",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("tokenizer/"),
        help="The output path for the trained tokenizer.",
    )

    # optional limiter
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2**16,
        help="The size of the vocabulary to train the tokenizer with.",
    )

    # format; either hf or tokenizers
    parser.add_argument(
        "--format",
        type=str,
        default="tokenizers",
        help="The format to save the tokenizer in.",
    )

    # parse the arguments
    args = parser.parse_args()

    def source_string_iterator() -> Iterable[str]:
        for name, source in get_linux_sources(linux_version=args.version):
            # check if it's safe as a utf-8 string
            try:
                try:
                    yield source.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        yield source.decode("latin-1")
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue

    # initialize the tokenizer
    t = init_tokenizer()

    # get special tokens including tabs, double spaces, quad spaces
    tokenizer_special_tokens = [
        START_FILE_TOKEN,
        END_FILE_TOKEN,
        START_HEADER_TOKEN,
        END_HEADER_TOKEN,
        PAD_TOKEN,
        "\t",
        " ",
        "  ",
        "    ",
    ]

    # train the tokenizer
    t.train_from_iterator(
        source_string_iterator(),
        min_frequency=1,
        vocab_size=args.vocab_size,
        special_tokens=tokenizer_special_tokens,
        show_progress=not args.quiet,
    )

    # set bos, eos, and pad tokens
    t.bos_token = START_FILE_TOKEN
    t.eos_token = END_FILE_TOKEN
    t.enable_padding(direction="left", pad_id=tokenizer_special_tokens.index(PAD_TOKEN))

    # now pad to the vocab size
    pre_vocab_size = len(t.get_vocab())
    if pre_vocab_size < args.vocab_size:
        pad_tokens = [f"<extra_id_{i}>" for i in range(pre_vocab_size, args.vocab_size)]
        t.add_special_tokens(pad_tokens)
        num_pad_tokens = len(pad_tokens)
    else:
        num_pad_tokens = 0

    # output post-train info
    num_tokens = len(t.get_vocab())
    num_special_tokens = len(tokenizer_special_tokens)

    # get relative path as absolute and ensure it exists
    args.output_path = Path(args.output_path).resolve()
    args.output_path.mkdir(parents=True, exist_ok=True)

    # save the tokenizer *file* in output path
    if args.format == "tokenizers":
        tokenizer_file_path = args.output_path / "tokenizer.json"
        t.save(str(tokenizer_file_path))
        output_path = tokenizer_file_path
        output_format = "tokenizers"
    else:
        tokenizer_model_path = args.output_path
        t.save_model(str(tokenizer_model_path))
        output_path = tokenizer_model_path
        output_format = "hf"

    # output all data as pretty json
    print(
        json.dumps(
            {
                "num_tokens": num_tokens,
                "num_special_tokens": num_special_tokens,
                "num_pad_tokens": num_pad_tokens,
                "pre_vocab_size": pre_vocab_size,
                "vocab_size": args.vocab_size,
                "output_path": str(output_path),
                "output_format": output_format,
            },
            indent=4,
        )
    )
