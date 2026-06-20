from __future__ import annotations

import argparse
import asyncio
import os
import random
import urllib.request
from pathlib import Path
from typing import List

from client import (
    BenchmarkTrace,
    benchmark_trace,
    init_logger,
    process_benchmark_results,
    scale_traces,
)
from openai import AsyncOpenAI as OpenAI
from pydantic import BaseModel

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"

# Char-based prompt generation — no tokenizer needed. See bench_simple.py.
_CHARS_PER_TOKEN = 4
_PROMPT_BASE = "The quick brown fox jumps over the lazy dog. "


def make_prompt(approx_tokens: int) -> str:
    target_chars = max(1, approx_tokens * _CHARS_PER_TOKEN)
    return (_PROMPT_BASE * (target_chars // len(_PROMPT_BASE) + 1))[:target_chars]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "model",
        help="Model name sent in chat-completion requests; must match the frontend's /v1/models.",
    )
    return p.parse_args()


def download_qwen_trace(url: str) -> str:
    dir = Path(__file__).resolve().parent
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


def read_qwen_trace(file_path: str, n: int | None = None) -> List[BenchmarkTrace]:
    """Parse a Qwen trace file. Prompts are char-approximated from input_length
    (~4 chars/token), so actual token counts will vary slightly from the trace.
    Timestamps and output_lengths are preserved exactly."""
    class JSONInput(BaseModel):
        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path) as f:
        lines = f.readlines()
    if n is not None:
        lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=make_prompt(obj.input_length),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


async def main():
    args = parse_args()
    MODEL = args.model
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        TRACES = read_qwen_trace(download_qwen_trace(URL), n=N)
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
