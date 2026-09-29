"""
passwords.py

Password hashing for admin accounts: PBKDF2-HMAC-SHA256 from the standard
library (no new native dependency), with the iteration count stored in the
hash so it can be raised later without invalidating existing passwords.

Format: pbkdf2_sha256$<iterations>$<salt hex>$<digest hex>
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 600_000  # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256
_SALT_BYTES = 16

# Verified against when the account doesn't exist, so an unknown email takes
# as long to reject as a wrong password (no user-enumeration timing oracle).
_DUMMY_HASH = None


def hash_password(password: str, iterations: int | None = None) -> str:
    if not isinstance(password, str) or password == "":
        raise ValueError("password must be a non-empty string")
    iterations = iterations or ITERATIONS
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, AttributeError, TypeError):
        return False
    return hmac.compare_digest(candidate.hex(), digest_hex)


def burn_time(password: str) -> None:
    """Spend the same work as a real verification, for unknown accounts."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("dummy-password-for-timing")
    verify_password(password or "x", _DUMMY_HASH)


def needs_rehash(stored: str) -> bool:
    try:
        algorithm, iterations, _, _ = stored.split("$")
        return algorithm != ALGORITHM or int(iterations) < ITERATIONS
    except ValueError:
        return True
