from __future__ import annotations

import multiprocessing as mp
import sys
from typing import TYPE_CHECKING

from frontend.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs

logger = init_logger(__name__, "initializer")


def _spawn_tokenizer(server_args: "ServerArgs", ack_queue: "mp.Queue[str]") -> None:
    """Launch the detokenizer worker plus N tokenizer workers.

    Topology (dedicated-only): the backend owns _B2Dt (PUSH bind); the
    detokenizer connects PULL to it. The frontend owns _F2T (HTTP PUSH bind);
    each tokenizer connects PULL to it. All workers PUSH-connect to _T2B
    (backend owns PULL bind) and PUSH-connect to _Dt2F (HTTP owns PULL bind).
    """
    from frontend.tokenizer import tokenize_worker

    num_tokenizers = server_args.num_tokenizer

    mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": server_args.model_path,
            "addr": server_args.zmq_detokenizer_addr,
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "create": False,  # backend binds _B2Dt
            "tokenizer_id": num_tokenizers,
            "ack_queue": ack_queue,
        },
        daemon=True,
        name="learninfra-detokenizer-0",
    ).start()

    for i in range(num_tokenizers):
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_tokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": False,  # HTTP binds _F2T
                "tokenizer_id": i,
                "ack_queue": ack_queue,
            },
            daemon=True,
            name=f"learninfra-tokenizer-{i}",
        ).start()

    for _ in range(num_tokenizers + 1):
        logger.info(ack_queue.get())


def launch_server(run_shell: bool = False) -> None:
    """Entry point: parse CLI args, spawn tokenizer subprocesses, run the API server.

    The GPU backend is not yet wired up — ZMQ PUSH to `zmq_backend_addr` will simply
    buffer locally until a backend binds that socket. Tokenizer/detokenizer workers
    still run, so HTTP routing and tokenization can be validated standalone.
    """
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)

    logger.info("ZMQ IPC base: %s", server_args.socket_base)
    logger.info("  HTTP API → tokenizer: %s", server_args.zmq_tokenizer_addr)
    logger.info("  tokenizer → backend : %s", server_args.zmq_backend_addr)
    logger.info("  backend   → detok   : %s", server_args.zmq_detokenizer_addr)
    logger.info("  detok     → HTTP API: %s", server_args.zmq_frontend_addr)

    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)
        ack_queue: mp.Queue[str] = mp.Queue()
        _spawn_tokenizer(server_args, ack_queue)

    run_api_server(server_args, start_subprocess, run_shell=run_shell)
