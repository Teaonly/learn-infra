from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from frontend.sampling import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: "BaseTokenizerMsg") -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> "BaseTokenizerMsg":
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    # Meaningful only when finished=True: the backend forwards the prompt
    # token count it received on the matching UserMsg so the detokenizer can
    # populate UserReply.prompt_tokens without tokenizer-side state.
    prompt_tokens: int = 0


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
