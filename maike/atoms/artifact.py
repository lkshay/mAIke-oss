"""Artifact models and metadata."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from maike.utils import utcnow


class ArtifactKind(str, Enum):
    STAGE = "stage"
    FILE = "file"


class ArtifactType(str, Enum):
    SPEC = "spec"
    PLAN = "plan"
    ARCHITECTURE = "architecture"
    CODE = "code"
    TEST = "test"
    DOCS = "docs"
    REVIEW = "review"
    ANALYSIS = "analysis"
    DIAGNOSIS = "diagnosis"
    RESULT = "result"


class ArtifactStatus(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    OBSERVED = "observed"
    INVALIDATED = "invalidated"


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    kind: ArtifactKind = ArtifactKind.STAGE
    type: ArtifactType
    logical_name: str
    path: str | None = None
    content: str
    content_hash: str = ""
    produced_by: str
    stage_name: str
    version: int = 1
    status: ArtifactStatus = ArtifactStatus.DRAFT
    invalidated: bool = False
    depends_on: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def set_hash(self) -> Self:
        if not self.content_hash:
            self.content_hash = sha256(self.content.encode("utf-8")).hexdigest()
        return self
