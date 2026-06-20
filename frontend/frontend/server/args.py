from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class ServerArgs:
    """Configuration for the frontend (HTTP API + tokenizer/detokenizer) only.

    The GPU backend is expected to live in another language/process and talk to
    the frontend over ZMQ. The addresses below name those IPC links.
    """

    model_path: str
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

    # Base path for ZMQ IPC files. When set, both frontend and backend can be
    # launched independently with the same value to rendezvous. When None,
    # a PID-suffixed path under /tmp is used so concurrent frontend instances
    # on the same host don't collide.
    socket_path: str | None = None

    @property
    def socket_base(self) -> str:
        if self.socket_path is not None:
            return self.socket_path
        return f"/tmp/learninfra{_get_pid_suffix()}"

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    # IPC slot numbering (mirrors SGLang's convention):
    #   _0  tokenizer → backend
    #   _1  backend   → detokenizer
    #   _2  RESERVED  backend → detokenizer control channel (abort / state sync).
    #                 Topology (decided): detokenizer binds PULL, backend
    #                 connects PUSH — same shape as _1 in shared mode.
    #                 Detokenizer owns the file lifecycle; when implemented,
    #                 add _2 to frontend's rm list in run.sh and bind it in
    #                 launch.py alongside _1. Today: unused, file never created.
    #   _3  detokenizer → HTTP API
    #   _4  HTTP API  → tokenizer (only when num_tokenizer > 0)

    @property
    def zmq_backend_addr(self) -> str:
        # Tokenizer → backend (scheduler) link
        return f"ipc://{self.socket_base}_0"

    @property
    def zmq_detokenizer_addr(self) -> str:
        # Backend → detokenizer link (also the shared tokenizer addr when num_tokenizer=0)
        return f"ipc://{self.socket_base}_1"

    @property
    def zmq_frontend_addr(self) -> str:
        # Detokenizer → HTTP API link
        return f"ipc://{self.socket_base}_3"

    @property
    def zmq_tokenizer_addr(self) -> str:
        # HTTP API → tokenizer link (distinct from detokenizer when num_tokenizer > 0)
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = f"ipc://{self.socket_base}_4"
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """Parse frontend command-line arguments.

    Args:
        args: Command line arguments (e.g. sys.argv[1:])
        run_shell: Force shell mode (also set by --shell-mode).

    Returns:
        (ServerArgs, run_shell) tuple.
    """
    parser = argparse.ArgumentParser(description="learn-infra Frontend Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. Must be an existing local folder.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help=(
            "Base path for ZMQ IPC files. The frontend creates four sockets: "
            "{socket_path}_0 (tokenizer→backend), _1 (backend→detokenizer), "
            "_3 (detokenizer→HTTP), _4 (HTTP→tokenizer when num_tokenizer>0). "
            "Pass the same value to a separately-launched backend so it can connect. "
            "Defaults to a PID-suffixed /tmp/learninfra* path so concurrent instances "
            "don't collide."
        ),
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    kwargs = parser.parse_args(args).__dict__.copy()

    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    result = ServerArgs(**kwargs)
    return result, run_shell
