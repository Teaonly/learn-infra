from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from frontend.sampling import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> "BaseBackendMsg":
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Set by the tokenizer worker (it knows input_ids.shape[0]). The backend
    # forwards it on the final DetokenizeMsg so the detokenizer can fill in
    # `UserReply.prompt_tokens` without sharing state with the tokenizer.
    prompt_tokens: int = 0


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
