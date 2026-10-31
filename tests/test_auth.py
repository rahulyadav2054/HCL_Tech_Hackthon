import time

import jwt
import pytest
from fastapi.testclient import TestClient

from backend import audit, auth
from backend.main import app

PASSWORD = "analyst-pass-123"


@pytest.fixture(scope="module")
def client():
    audit.init_db()
    audit.set_password_hash("analyst1", auth.hash_password(PASSWORD))
    audit.set_password_hash("admin1", auth.hash_password("admin-pass-456"))
    # no `with`: the lifespan (model loading) does not run; auth endpoints do not need the models
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_throttle():
    auth.THROTTLE._fails.clear()
    yield
    auth.THROTTLE._fails.clear()


def login(client, user="analyst1", pw=PASSWORD):
    return client.post("/auth/login", json={"username": user, "password": pw})


def test_password_hash_roundtrip():
    stored = auth.hash_password("s3cret-pass")
    assert stored.startswith("pbkdf2_sha256$") and "s3cret-pass" not in stored
    assert auth.verify_password("s3cret-pass", stored)
    assert not auth.verify_password("wrong", stored)
    assert not auth.verify_password("s3cret-pass", "garbage") and not auth.verify_password("x", None)
    assert auth.hash_password("same") != auth.hash_password("same")  # salted


def test_login_returns_jwt_and_me_works(client):
    r = login(client)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["role"] == "L1" and body["expires_in"] > 0
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["user_id"] == "analyst1" and me.json()["role"] == "L1"


def test_wrong_password_and_unknown_user_are_rejected_alike(client):
    a, b = login(client, pw="nope"), login(client, user="mallory", pw="nope")
    assert a.status_code == b.status_code == 401
    assert a.json()["detail"] == b.json()["detail"]


def test_lockout_after_repeated_failures(client):
    for _ in range(5):
        assert login(client, pw="bad").status_code == 401
    r = login(client)  # even the right password is refused while locked
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_protected_endpoints_need_a_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.post("/search", json={"query": "vpn"}).status_code == 401
    assert client.get("/audit/export").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


def test_tampered_expired_and_foreign_tokens_are_rejected(client):
    token = login(client).json()["access_token"]
    head, payload, sig = token.split(".")
    tampered = f"{head}.{payload}.{'A' if sig[0] != 'A' else 'B'}{sig[1:]}"
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401

    now = int(time.time())
    expired = jwt.encode({"sub": "analyst1", "role": "L1", "iat": now - 7200, "exp": now - 3600, "iss": auth.ISSUER},
                         auth._secret(), algorithm=auth.ALGORITHM)
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]

    foreign = jwt.encode({"sub": "admin1", "role": "L2_Admin", "iat": now, "exp": now + 600, "iss": auth.ISSUER},
                         "some-other-secret", algorithm=auth.ALGORITHM)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {foreign}"}).status_code == 401


def test_role_comes_from_database_not_token(client):
    token, _ = auth.create_access_token("analyst1", "L2_Admin")  # claims admin, but analyst1 is L1
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["role"] == "L1"
