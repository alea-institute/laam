
"""laam.data - Data loader and preprocessing functionality, including:
 - Retrieving the linux source code tars from official mirror
 - Iterating over source objects within the tars
 - Segmenting and tokenizing the source objects
"""
# imports
import random
import tarfile
from pathlib import Path
from typing import Callable, Iterator, Tuple

# packages
import httpx
import tqdm
import torch.utils.data
from transformers import PreTrainedTokenizerFast

# laam imports
from laam.model.tokenizer import get_tokenizer, PAD_TOKEN

# constants
DATA_PATH = Path(__file__).parent.parent.parent / "data"
DEFAULT_LINUX_MIRROR_URI = "https://mirrors.edge.kernel.org/pub/linux/kernel/"
DEFAULT_LINUX_VERSION = "v6.x/6.9.4"
DEFAULT_USER_AGENT = "laam/0.1.0 <https://github.com/mjbommar/linux-as-a-model>"

# special tokens
START_FILE_TOKEN = "<|start_file|>"
END_FILE_TOKEN = "<|end_file|>"
START_HEADER_TOKEN = "<|start_header|>"
END_HEADER_TOKEN = "<|end_header|>"

SPECIAL_TOKENS = [
    START_FILE_TOKEN,
    END_FILE_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
]


def is_source_file(name: str | Path) -> bool:
    """
    Check if the file is a source file in standard kernel file extensions.

    Args:
        name (str | Path): The name of the file to check.

    Returns:
        bool: Whether the file is a source file.
    """
    if isinstance(name, Path):
        name = name.name

    # check if the file is a source file
    return any(
        name.lower().endswith(extension)
        for extension in [".c", ".cc", ".cpp", ".cxx", ".c++",
                            ".h", ".hh", ".hpp", ".hxx", ".h++",
                          ".s", ".rs", ".txt",
        ]
    )


def download_linux_source(
    linux_version: str = DEFAULT_LINUX_VERSION,
    mirror_uri: str = DEFAULT_LINUX_MIRROR_URI,
) -> Path:
    """Download the linux source tarball from the official mirror.

    Args:
        linux_version (str): The linux version to download.
        mirror_uri (str): The base URI for the linux mirror.

    Returns:
        Path: The path to the downloaded tarball.
    """
    # create the data directory if it doesn't exist
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    # create the path for the tarball
    major_family = linux_version.split("/")[0]
    kernel_version = linux_version.split("/")[1]
    tarball_path = DATA_PATH / f"linux-{kernel_version}.tar.xz"

    # download the tarball if it doesn't exist
    if not tarball_path.exists():
        # create a client and stream the response
        tarball_url = f"{mirror_uri.rstrip('/')}/{major_family}/linux-{kernel_version}.tar.xz"

        # set up a nice client
        with httpx.Client(
                http2=True,
                follow_redirects=True,
                headers={"User-Agent": DEFAULT_USER_AGENT}
        ) as client:
            with client.stream("GET", tarball_url) as response:
                # set up a progress bar
                prog_bar = tqdm.tqdm(
                    total=int(response.headers.get("Content-Length", 0)),
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {tarball_url}",
                )
                # write the response to the tarball file
                try:
                    with tarball_path.open("wb") as tarball_file:
                        for chunk in response.iter_bytes():
                            tarball_file.write(chunk)
                            prog_bar.update(len(chunk))
                except Exception as e:
                    # unlink it
                    tarball_path.unlink()

                    # raise the exception
                    raise e

                # close the progress bar
                prog_bar.close()

    return tarball_path


def get_linux_sources(
    linux_version: str = DEFAULT_LINUX_VERSION,
    mirror_uri: str = DEFAULT_LINUX_MIRROR_URI,
    add_file_tokens: bool = True,
    filter_method: Callable[[str], bool] = is_source_file,
    limit: int | None = None,
) -> Iterator[Tuple[str, bytes]]:
    """Iterate over the linux source files in the tarball.

    Args:
        linux_version (str): The linux version to download.
        mirror_uri (str): The base URI for the linux mirror.
        add_file_tokens (bool): Whether to add file tokens to the source.
        filter_method (Callable[[Path], bool]): A method to filter the files.
        limit (int | None): The number of files to limit the iteration to.

    Yields:
        Tuple[str, bytes]: The name and contents of the linux source file.
    """
    # download the tarball
    tarball_path = download_linux_source(linux_version, mirror_uri)

    # switch opener mode based on extension
    if tarball_path.suffix == ".xz":
        opener = tarfile.open
        mode = "r:xz"
    elif tarball_path.suffix == ".gz":
        opener = tarfile.open
        mode = "r:gz"
    elif tarball_path.suffix == ".bz2":
        opener = tarfile.open
        mode = "r:bz2"
    else:
        raise ValueError(f"Unsupported extension: {tarball_path.suffix}")

    # open the tarball
    with opener(tarball_path, mode) as tarball:
        # iterate over the members
        for member_number, member in enumerate(tarball.getmembers()):
            if limit is not None and member_number >= limit:
                break

            # check if the member is a file
            if member.isfile() and filter_method(Path(member.name)):
                # extract the member
                with tarball.extractfile(member) as member_file:
                    file_name = member.name
                    if not filter_method(file_name):
                        continue

                    file_source = member_file.read()
                    if add_file_tokens:
                        output_source = (
                                                f"{START_FILE_TOKEN}{START_HEADER_TOKEN}".encode("utf-8") +
                                                file_name.encode("utf-8") +
                                                f"{END_HEADER_TOKEN}".encode("utf-8") +
                                                file_source +
                                                f"{END_FILE_TOKEN}".encode("utf-8")
                        )
                    else:
                        output_source = file_source

                    yield file_name, output_source



class KernelSourceDataset(torch.utils.data.Dataset):
    # constructor with version and tokenizer
    def __init__(self, version: str = "v1.0/1.0", tokenizer: PreTrainedTokenizerFast | None = None, max_tokens: int = 1024, shuffle: bool = True):
        """
        Initialize the dataset.

        Args:
            version (str): The version of the linux kernel to download.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
            max_tokens (int): The maximum number of tokens to use.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            None
        """
        # set the version
        self.version = version

        # set the tokenizer and params
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

        # set shuffle param
        self.shuffle = shuffle
        self.unseen_indices = []

        # set up the linux source iterator
        self.source_iterator = get_linux_sources(version)
        self.samples: list[dict[str, list[int]]] = []

        # store stats
        self.num_tokens = 0
        self.num_records = 0

        # materialize the records
        self.load_records()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, list[int]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            dict[str, list[int]]: The item from the dataset.
        """
        # get a random index
        if self.shuffle:
            # pop a random index from the unseen indices
            if len(self.unseen_indices) == 0:
                self.unseen_indices = list(range(len(self.samples)))
                random.shuffle(self.unseen_indices)
            random_index = self.unseen_indices.pop()
            return self.samples[random_index]
        else:
            return self.samples[idx]

    def load_records(self):
        """
        Load the records from the source iterator.
        """
        # iterate over the linux sources
        for source_name, source_text in self.source_iterator:
            # get the raw tokens without truncation
            tokens = self.tokenizer.encode(
                text=source_text.decode("utf-8"),
                add_special_tokens=False,
                truncation=False
            )
            self.num_tokens += len(tokens)
            special_tokens_mask = [
                1 if token in (START_FILE_TOKEN, END_FILE_TOKEN, PAD_TOKEN) else 0
                for token in tokens
            ]

            if len(tokens) <= self.max_tokens:
                pad_token_count = self.max_tokens - len(tokens)
                record = {
                    "input_ids": [self.tokenizer.pad_token_id] * pad_token_count + tokens,
                    "attention_mask": [0] * pad_token_count + [1] * len(tokens),
                    "special_tokens_mask": [1] * pad_token_count + special_tokens_mask,
                    "labels": [self.tokenizer.pad_token_id] * pad_token_count + tokens,
                }
                self.samples.append(record)
                self.num_records += 1
            else:
                for i in range(0, len(tokens), self.max_tokens):
                    segment_length = min(self.max_tokens, len(tokens) - i)
                    pad_token_count = self.max_tokens - segment_length
                    record = {
                        "input_ids": [self.tokenizer.pad_token_id] * pad_token_count + tokens[i:i + segment_length],
                        "attention_mask": [0] * pad_token_count + [1] * segment_length,
                        "special_tokens_mask": [1] * pad_token_count + special_tokens_mask[i:i + segment_length],
                        "labels": [self.tokenizer.pad_token_id] * pad_token_count + tokens[i:i + segment_length],
                    }
                    self.samples.append(record)
                    self.num_records += 1

        # set up unseen indices
        self.unseen_indices = list(range(len(self.samples)))
        random.shuffle(self.unseen_indices)
