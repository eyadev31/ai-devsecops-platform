"""
Auth Routes
===========
User registration and JWT login for the Portfolio System API.
Users are persisted to a JSON file so they survive server restarts.
"""

import json
import logging
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional

from api.services.auth_service import (
    hash_password, verify_password, create_access_token, decode_token
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ── Persistent user store (JSON file) ────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _load_users() -> dict[str, dict]:
    """Load users from disk. Seed default admin if file doesn't exist."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Corrupted users.json — recreating with defaults")

    # Seed default admin account
    default_users = {
        "admin@gov.dz": {
            "email": "admin@gov.dz",
            "name": "Government Admin",
            "password_hash": hash_password("admin123"),
            "created_at": datetime.utcnow().isoformat(),
            "binance_connected": False,
            "binance_api_key": None,
            "binance_api_secret": None,
            "daq_sessions": [],
        }
    }
    _save_users(default_users)
    return default_users


def _save_users(users: dict[str, dict]):
    """Write users to disk atomically."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_file = USERS_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, default=str)
    # Atomic rename (safe on Windows with same volume)
    if os.path.exists(USERS_FILE):
        os.remove(USERS_FILE)
    os.rename(tmp_file, USERS_FILE)


# Load users once at module startup
_users_db: dict[str, dict] = _load_users()


# ── Models ───────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: str = Field(default="Investor", description="Display name")


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ── Dependency: Get current user from JWT ────────────────────
async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.split(" ")[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    email = payload.get("sub")
    user = _users_db.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


# ── Routes ───────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Register a new user account."""
    if req.email in _users_db:
        # If user exists, just log them in instead of erroring
        user = _users_db[req.email]
        if not verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="User already exists with different password")
        token = create_access_token({"sub": req.email})
        logger.info(f"User re-registered (auto-login): {req.email}")
        return TokenResponse(
            access_token=token,
            user={"email": user["email"], "name": user["name"], "binance_connected": user.get("binance_connected", False)}
        )
    
    user = {
        "email": req.email,
        "name": req.name,
        "password_hash": hash_password(req.password),
        "created_at": datetime.utcnow().isoformat(),
        "binance_connected": False,
        "binance_api_key": None,
        "binance_api_secret": None,
        "daq_sessions": [],
    }
    _users_db[req.email] = user
    _save_users(_users_db)  # Persist to disk
    
    token = create_access_token({"sub": req.email})
    logger.info(f"User registered: {req.email}")
    
    return TokenResponse(
        access_token=token,
        user={"email": user["email"], "name": user["name"], "binance_connected": False}
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Login with email and password."""
    user = _users_db.get(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": req.email})
    logger.info(f"User logged in: {req.email}")
    
    return TokenResponse(
        access_token=token,
        user={
            "email": user["email"],
            "name": user["name"],
            "binance_connected": user["binance_connected"],
        }
    )


@router.get("/me")
async def get_profile(user: dict = Depends(get_current_user)):
    """Get the current user's profile."""
    return {
        "email": user["email"],
        "name": user["name"],
        "binance_connected": user["binance_connected"],
        "created_at": user["created_at"],
        "total_daq_sessions": len(user.get("daq_sessions", [])),
    }
