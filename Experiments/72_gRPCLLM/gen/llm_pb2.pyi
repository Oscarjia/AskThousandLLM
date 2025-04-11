from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("prompt", "max_tokens", "temperature", "history", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    max_tokens: int
    temperature: float
    history: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., history: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("reply", "is_final")
    REPLY_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    reply: str
    is_final: bool
    def __init__(self, reply: _Optional[str] = ..., is_final: bool = ...) -> None: ...
