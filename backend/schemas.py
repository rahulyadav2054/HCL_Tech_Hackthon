"""Request bodies for the API (response bodies are the analyzer's JSON, documented in the README)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(..., examples=["analyst1"])
    password: str = Field(..., examples=["your-password"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="seconds until the token expires")
    user_id: str
    role: str


class Filters(BaseModel):
    application: list[str] | str | None = None
    category: list[str] | str | None = None
    date_from: str | None = Field(None, examples=["2025-01-01"])
    date_to: str | None = None
    status: list[str] | str | None = Field(None, examples=[["Resolved", "Closed"]])
    assignment_group: list[str] | str | None = None
    environment: list[str] | str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "", [])}


class AnalyzeRequest(BaseModel):
    title: str = Field(..., examples=["VPN not connecting after password change"])
    description: str = Field(..., examples=["Since this morning I get error 809 on GlobalProtect from home. "
                                            "Changed my password yesterday."])
    application: str | None = Field("unknown", examples=["GlobalProtect VPN"])
    environment: str | None = Field("unknown", examples=["Production"])
    filters: Filters | None = None
    use_llm: bool = True

    @field_validator("title", "description")
    @classmethod
    def not_blank(cls, v):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., examples=["error 809 vpn"])
    filters: Filters | None = None
    top_k: int = Field(10, ge=1, le=50)
    include_kb: bool = True


class FeedbackRequest(BaseModel):
    analysis_id: str
    target_type: Literal["match", "resolution_path", "classification"]
    target_id: str = Field(..., examples=["INC0010455"])
    verdict: str = Field(..., examples=["incorrect"], description="useful | incorrect | unresolved")
    corrected_label: str | None = None
    comment: str | None = None


class DecisionRequest(BaseModel):
    decision: Literal["Accepted", "Edited", "Rejected", "Escalated"]
    comment: str | None = None
