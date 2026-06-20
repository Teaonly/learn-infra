import argparse
import asyncio
import random
import sys
from typing import List

from client import (
    benchmark_one,
    benchmark_one_batch,
    init_logger,
    process_benchmark_results,
)
from openai import AsyncOpenAI as OpenAI

logger = init_logger(__name__)

# Char-based prompt generation — no tokenizer needed. ~4 chars/token for
# English text, so this approximates the requested token count within a
# few percent. Good enough for benchmarking; the server has its own
# tokenizer and never sees this approximation.
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


async def main():
    args = parse_args()
    MODEL = args.model
    try:
        random.seed(42)  # reproducibility

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of prompts with random input lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, MAX_INPUT)
                result.append(make_prompt(length))
                await asyncio.sleep(0)
            return result

        TEST_BS = [64]
        PORT = 1919
        MAX_INPUT = 8192  # approximate tokens
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
            logger.info(f"Using model: {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                test_msg = make_prompt(100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                raise e from e

            msgs = await gen_task
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            for batch_size in TEST_BS:
                try:
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL
                    )
                    process_benchmark_results(results)
                except Exception as e:
                    logger.info(f"Error with batch size {batch_size}: {e}")
                    continue
            logger.info("Benchmark completed.")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
