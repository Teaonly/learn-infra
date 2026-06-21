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
    num_tokenizer: int = 1
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

    # IPC slot naming (directional, e.g. _B2Dt = "backend to detokenizer"):
    #   _T2B   tokenizer  → backend     (backend binds PULL)
    #   _B2Dt  backend    → detokenizer (backend binds PUSH, detok connects PULL)
    #   _Dt2F  detokenizer → HTTP API   (HTTP binds PULL, detok connects PUSH)
    #   _F2T   HTTP API   → tokenizer   (HTTP binds PUSH, tok connects PULL)
    # Backend owns _T2B/_B2Dt file lifecycle; frontend owns _Dt2F/_F2T.
    # run.sh cleanup mirrors that split so the two processes can start in any order.

    @property
    def zmq_backend_addr(self) -> str:
        # Tokenizer → backend link
        return f"ipc://{self.socket_base}_T2B"

    @property
    def zmq_detokenizer_addr(self) -> str:
        # Backend → detokenizer link
        return f"ipc://{self.socket_base}_B2Dt"

    @property
    def zmq_frontend_addr(self) -> str:
        # Detokenizer → HTTP API link
        return f"ipc://{self.socket_base}_Dt2F"

    @property
    def zmq_tokenizer_addr(self) -> str:
        # HTTP API → tokenizer link
        return f"ipc://{self.socket_base}_F2T"


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
        help="The number of tokenizer worker processes to launch (>=1). The detokenizer always runs as a separate process.",
    )

    parser.add_argument(
        "--socket-path",
        type=str,
        default=None,
        help=(
            "Base path for ZMQ IPC files. The frontend creates four sockets: "
            "{socket_path}_T2B (tokenizer→backend), _B2Dt (backend→detokenizer), "
            "_Dt2F (detokenizer→HTTP), _F2T (HTTP→tokenizer). "
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
