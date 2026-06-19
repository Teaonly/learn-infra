from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer from a local model path.

    `local_files_only=True` disables any network fetch — `model_path` must
    point at a directory already on disk.
    """
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
