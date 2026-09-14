from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .column import Column


@dataclass(frozen=True, kw_only=True)
class Row:
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def value_for(self, column: Column) -> Any:
        if not hasattr(self, column.name):
            raise ValueError(f"{type(self).__name__} is missing column {column.name}")
        return column.to_sql_value(getattr(self, column.name))

    def insert_params(self, columns: tuple[Column, ...]) -> tuple[Any, ...]:
        return tuple(self.value_for(column) for column in columns)
