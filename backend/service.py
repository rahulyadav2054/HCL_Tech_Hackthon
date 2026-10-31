"""Backend service layer: user -> role, AI analysis, audit persistence.

The AI layer (ai.analyzer) never touches the database; this module wires it to the audit log:
  * looks up the user's role (users table, seeded from data/users.csv; the API authenticates with JWT)
  * passes analyst-feedback signals into the analysis (hidden matches, ranking adjustments)
  * writes the audit row BEFORE returning the response (FR-08)
"""
from __future__ import annotations

from ai.analyzer import get_analyzer
from backend import audit


class UnknownUser(PermissionError):
    pass


class Service:
    def __init__(self):
        audit.init_db()
        self.analyzer = get_analyzer()

    @property
    def retriever(self):
        return self.analyzer.retriever

    def role_of(self, user_id: str) -> str:
        role = audit.get_role(user_id)
        if role is None:
            raise UnknownUser(f"unknown user '{user_id}'")
        return role

    def analyze(self, ticket: dict, user_id: str, use_llm: bool = True, write_audit: bool = True) -> dict:
        role = self.role_of(user_id)
        response, record = self.analyzer.analyze(ticket, role, user_id=user_id, use_llm=use_llm,
                                                 hidden_for=audit.hidden_matches,
                                                 feedback_adj=audit.feedback_adjustments())
        if write_audit:
            audit.write_analysis(record)
        return response

    def search(self, query: str, user_id: str, filters: dict | None = None, top_k: int = 10,
               include_kb: bool = True) -> dict:
        return self.analyzer.search(query, self.role_of(user_id), filters, top_k, include_kb)

    def health(self) -> dict:
        # public endpoint: versions, filter catalog and aggregate metrics only (no user list, no ticket text)
        return {"status": "ok", **self.analyzer.info(), "metrics": audit.metrics()}


_service = None


def get_service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service
