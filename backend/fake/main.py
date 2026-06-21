"""Debug-only fake backend for IPC smoke testing.

Not a real backend. Reads `UserMsg` from `ipc://{base}_T2B` (PULL/bind),
ignores the input, and streams the fixed `REPLAY_TOKENS` back as one
`DetokenizeMsg` per token to `ipc://{base}_B2Dt` (PUSH/bind). The
frontend's detokenizer worker turns those tokens into text, so the
end-to-end effect visible through HTTP is a fixed, deterministic stream
— useful for exercising the full HTTP → tokenizer → backend →
detokenizer → HTTP pipeline and verifying streaming chunks + the
`finished=True` end signal without a GPU or real model weights.

Usage:
    python main.py <socket-base> [model-path]

Topology: the backend always owns both bind sides:
    ipc://{base}_T2B   PULL bind    (receives UserMsg from tokenizer)
    ipc://{base}_B2Dt  PUSH bind    (sends DetokenizeMsg to detokenizer)

Start the frontend first so the detokenizer is connecting to `_B2Dt`
before we start emitting; otherwise the first few PUSH frames buffer in
the local socket until a peer connects.

`model-path` is currently unused — this backend emits a fixed token
sequence, so no tokenizer is needed. It's accepted only to match the
run.sh CLI contract for a future real backend.

Wire format: msgpack + `__type__` discriminator, mirroring the frontend
serializer (`serialize_type` / `deserialize_type` in
frontend/frontend/message/utils.py). torch.Tensor is decoded to a 1D
numpy array; the wire bytes are unchanged, so this stays interoperable
with the frontend.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import msgpack
import numpy as np
import zmq

TOKEN_IM_START = 151644
TOKEN_IM_END = 151645

REPLAY_TOKENS = [14990, 14991, 14992, 14993, TOKEN_IM_END]

STREAM_DELAY_S = 0.01


# ---- Wire-format definitions (mirror frontend subset) -----------------------

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024


@dataclass
class UserMsg:
    uid: int
    input_ids: np.ndarray  # 1D int32, decoded from the torch.Tensor wire format
    sampling_params: SamplingParams
    prompt_tokens: int = 0  # filled by the tokenizer; we forward it on finish


@dataclass
class DetokenizeMsg:
    uid: int
    next_token: int
    finished: bool
    prompt_tokens: int = 0  # only meaningful on the finished frame


# ---- (De)serialization -------------------------------------------------------

_TYPE_REGISTRY = {
    "SamplingParams": SamplingParams,
    "UserMsg": UserMsg,
}


def _deserialize_any(data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return _deserialize_typed(data)
        return {k: _deserialize_any(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_deserialize_any(v) for v in data]
    return data


def _deserialize_typed(data: Dict) -> Any:
    type_name = data["__type__"]

    # 1D torch.Tensor wire format: {"buffer": bytes, "dtype": "torch.<dtype>"}
    if type_name == "Tensor":
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        return np.frombuffer(data["buffer"], dtype=np_dtype).copy()

    # BatchBackendMsg unwraps to a list of underlying messages.
    if type_name == "BatchBackendMsg":
        return [_deserialize_any(item) for item in data["data"]]

    cls = _TYPE_REGISTRY[type_name]
    kwargs = {k: _deserialize_any(v) for k, v in data.items() if k != "__type__"}
    return cls(**kwargs)


def unpack(raw: bytes) -> Any:
    """Decode one msgpack frame from the wire into typed Python objects."""
    return _deserialize_any(msgpack.unpackb(raw, raw=False))


def pack_detokenize(msg: DetokenizeMsg) -> Any:
    """Encode a DetokenizeMsg to the frontend's wire format."""
    packed: Any = msgpack.packb(
        {
            "__type__": "DetokenizeMsg",
            "uid": msg.uid,
            "next_token": int(msg.next_token),
            "finished": msg.finished,
            "prompt_tokens": int(msg.prompt_tokens),
        },
        use_bin_type=True,
    )
    return packed


# ---- Main loop ---------------------------------------------------------------

def main(socket_base: str, _model_path: str | None = None) -> None:
    ctx = zmq.Context()

    recv = ctx.socket(zmq.PULL)
    recv.bind(f"ipc://{socket_base}_T2B")

    send = ctx.socket(zmq.PUSH)
    send.bind(f"ipc://{socket_base}_B2Dt")

    print(f"[fake-backend] pull: ipc://{socket_base}_T2B", flush=True)
    print(f"[fake-backend] push: ipc://{socket_base}_B2Dt", flush=True)

    while True:
        raw = recv.recv()
        msgs = unpack(raw)
        if not isinstance(msgs, list):
            msgs = [msgs]
        for msg in msgs:
            if isinstance(msg, UserMsg):
                _handle(msg, send)


def _handle(msg: UserMsg, send: zmq.Socket) -> None:
    # Stream the fixed REPLAY_TOKENS regardless of input — exercises the
    # frontend's streaming path and the finished=True end signal without
    # depending on actual input content. The final frame carries
    # prompt_tokens back to the detokenizer so HTTP usage stats are right.
    last = len(REPLAY_TOKENS) - 1
    for i, tok in enumerate(REPLAY_TOKENS):
        send.send(pack_detokenize(
            DetokenizeMsg(
                uid=msg.uid,
                next_token=int(tok),
                finished=i == last,
                prompt_tokens=msg.prompt_tokens if i == last else 0,
            )
        ))
        time.sleep(STREAM_DELAY_S)
    print(
        f"[fake-backend] uid={msg.uid} prompt_tokens={msg.prompt_tokens} -> "
        f"{len(REPLAY_TOKENS)} fixed tokens",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
