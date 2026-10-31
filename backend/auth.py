"""JWT authentication for the backend.

Flow: POST /auth/login {username, password} -> the password is checked against a salted PBKDF2-SHA256
hash in the users table -> a signed JWT (HS256) is returned -> every protected endpoint reads
`Authorization: Bearer <token>`, verifies signature, expiry and issuer, and then loads the user's role
from the database. The role inside the token is informational only: a role change or a deleted user
takes effect on the next request, not when the token expires.

Secrets and passwords come from .env (JWT_SECRET, DEMO_PASSWORD) or this CLI, never from code:
  python -m backend.auth --generate-secret          print a random JWT_SECRET for .env
  python -m backend.auth --set-password analyst1    set or change a user's password (prompted)
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai import config
from backend import audit

ALGORITHM = "HS256"
ISSUER = "it-ticket-assistant"
PBKDF2_ITERATIONS = 310_000
MIN_PASSWORD_LENGTH = 8

_bearer = HTTPBearer(auto_error=False, description="JWT from POST /auth/login")
_ephemeral_secret = secrets.token_urlsafe(48)
_warned = False


def _secret() -> str:
    global _warned
    if config.JWT_SECRET:
        return config.JWT_SECRET
    if not _warned:
        print("[auth] JWT_SECRET is not set: using a random per-process secret; tokens end at restart",
              file=sys.stderr)
        _warned = True
    return _ephemeral_secret


# ----------------------------------------------------------------------------- passwords
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Salted PBKDF2-SHA256, stored as pbkdf2_sha256$<iterations>$<salt>$<hash>."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------------- login attempts
class LoginThrottle:
    """Locks a username after LOGIN_MAX_FAILURES failed logins within LOGIN_LOCKOUT_SECONDS (in memory)."""

    def __init__(self):
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, username: str) -> deque:
        q = self._fails[username]
        while q and time.time() - q[0] > config.LOGIN_LOCKOUT_SECONDS:
            q.popleft()
        return q

    def locked_for(self, username: str) -> int:
        """Seconds until the lockout ends, 0 when the user may try."""
        with self._lock:
            q = self._recent(username)
            if len(q) < config.LOGIN_MAX_FAILURES:
                return 0
            return max(1, int(config.LOGIN_LOCKOUT_SECONDS - (time.time() - q[0])))

    def failed(self, username: str) -> None:
        with self._lock:
            self._recent(username).append(time.time())

    def succeeded(self, username: str) -> None:
        with self._lock:
            self._fails.pop(username, None)


THROTTLE = LoginThrottle()


def authenticate(username: str, password: str) -> str | None:
    """Return the user's role when the password matches, else None (same answer for unknown users)."""
    stored = audit.get_password_hash(username)
    if stored is None:
        verify_password(password, hash_password("timing-equalizer", iterations=1000))  # similar work either way
        return None
    return audit.get_role(username) if verify_password(password, stored) else None


# ----------------------------------------------------------------------------- tokens
def create_access_token(user_id: str, role: str, minutes: int | None = None) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    ttl = 60 * (minutes or config.JWT_EXPIRE_MINUTES)
    now = int(time.time())
    payload = {"sub": user_id, "role": role, "iat": now, "nbf": now, "exp": now + ttl,
               "iss": ISSUER, "jti": secrets.token_hex(8)}
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM), ttl


def decode_token(token: str) -> dict:
    """Verify signature, algorithm, expiry and issuer. Raises jwt.InvalidTokenError subclasses."""
    return jwt.decode(token, _secret(), algorithms=[ALGORITHM], issuer=ISSUER,
                      options={"require": ["sub", "exp", "iat", "iss"]})


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """FastAPI dependency: the authenticated user and their CURRENT role from the database."""
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise _unauthorized("not authenticated: send Authorization: Bearer <token> from POST /auth/login")
    try:
        claims = decode_token(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("token expired: log in again")
    except jwt.InvalidTokenError:
        raise _unauthorized("invalid token")
    role = audit.get_role(claims["sub"])
    if role is None:
        raise _unauthorized("user no longer exists")
    return {"user_id": claims["sub"], "role": role, "token_expires_at": claims["exp"]}


# ----------------------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate-secret", action="store_true", help="print a random JWT_SECRET for .env")
    ap.add_argument("--set-password", metavar="USER", help="set a user's password (prompted, not echoed)")
    a = ap.parse_args()
    if a.generate_secret:
        print(f"JWT_SECRET={secrets.token_urlsafe(48)}")
    if a.set_password:
        audit.init_db()
        if audit.get_role(a.set_password) is None:
            sys.exit(f"unknown user '{a.set_password}' (users come from data/users.csv)")
        pw = getpass.getpass(f"New password for {a.set_password}: ")
        if len(pw) < MIN_PASSWORD_LENGTH:
            sys.exit(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
        if pw != getpass.getpass("Repeat password: "):
            sys.exit("passwords do not match")
        audit.set_password_hash(a.set_password, hash_password(pw))
        print(f"password updated for {a.set_password}")
    if not (a.generate_secret or a.set_password):
        ap.print_help()


if __name__ == "__main__":
    _cli()
