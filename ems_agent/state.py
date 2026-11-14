import json
from dataclasses import dataclass, field
from typing import Any

from .config import FIELDS


@dataclass
class CallState:
    values: dict[str, Any] = field(default_factory=lambda: {name: None for name in FIELDS})
    retries: dict[str, int] = field(default_factory=lambda: {name: 0 for name in FIELDS})
    declined: set[str] = field(default_factory=set)
    introduced: bool = False
    hold_acknowledged: bool = False
    on_hold: bool = False
    readback_done: bool = False
    active_field: str | None = None
    field_buffers: dict[str, str] = field(default_factory=dict)
    correction_field: str | None = None
    awaiting_correction: bool = False
    correction_attempts: int = 0
    ended: bool = False
    transcript: list[tuple[str, str]] = field(default_factory=list)

    def missing_fields(self) -> list[str]:
        return [
            name for name, value in self.values.items()
            if value is None and name not in self.declined
        ]

    def collection_complete(self) -> bool:
        return all(value is not None or name in self.declined for name, value in self.values.items())

    def all_done(self) -> bool:
        return not self.missing_fields()

    def to_json(self) -> str:
        return json.dumps(self.values, ensure_ascii=True)