"""Pydantic request bodies. Responses are plain dicts to keep the JSON shape
identical to the original C++ server (camelCase keys)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class InsertVectorRequest(BaseModel):
    metadata: str = Field(min_length=1)
    category: str = ""
    embedding: list[float] = Field(min_length=1)


class DocInsertRequest(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=20)
