"""
Portfolio Routes
================
Binance account connection, balance queries, and trade execution.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from api.routes.auth import get_current_user, _users_db
from api.services.binance_service import BinanceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["Portfolio & Binance"])


# ── Models ───────────────────────────────────────────────────
class ConnectBinanceRequest(BaseModel):
    api_key: str = Field(..., description="Binance API key")
    api_secret: str = Field(..., description="Binance API secret")
    testnet: bool = Field(default=False, description="Use Binance testnet")


class TradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair (e.g., BTCUSDT)")
    side: str = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    order_type: str = Field(default="MARKET", description="MARKET or LIMIT")
    price: Optional[float] = Field(None, description="Limit price (required for LIMIT)")


# ── Routes ───────────────────────────────────────────────────
@router.post("/connect-binance")
async def connect_binance(req: ConnectBinanceRequest, user: dict = Depends(get_current_user)):
    """Connect user's Binance account by validating and storing API keys."""
    
    # Validate keys first
    validation = BinanceService.validate_keys(req.api_key, req.api_secret, req.testnet)
    
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid Binance keys: {validation.get('error')}")
    
    # Store keys (encrypted in production)
    user["binance_api_key"] = req.api_key
    user["binance_api_secret"] = req.api_secret
    user["binance_testnet"] = req.testnet
    user["binance_connected"] = True
    _users_db[user["email"]] = user
    
    logger.info(f"Binance connected for user: {user['email']} (can_trade={validation['can_trade']})")
    
    return {
        "connected": True,
        "can_trade": validation["can_trade"],
        "account_type": validation["account_type"],
    }


@router.get("/balances")
async def get_balances(user: dict = Depends(get_current_user)):
    """Get user's Binance account balances."""
    if not user.get("binance_connected"):
        raise HTTPException(status_code=400, detail="Binance not connected. Use /connect-binance first.")
    
    svc = BinanceService(
        user["binance_api_key"], 
        user["binance_api_secret"],
        user.get("binance_testnet", False)
    )
    
    try:
        return svc.get_portfolio_value()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(user: dict = Depends(get_current_user)):
    """Get detailed portfolio positions with weights and values."""
    if not user.get("binance_connected"):
        raise HTTPException(status_code=400, detail="Binance not connected")
    
    svc = BinanceService(
        user["binance_api_key"],
        user["binance_api_secret"],
        user.get("binance_testnet", False)
    )
    
    try:
        portfolio = svc.get_portfolio_value()
        account = svc.get_account_info()
        return {
            **portfolio,
            "can_trade": account["can_trade"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trade")
async def execute_trade(req: TradeRequest, user: dict = Depends(get_current_user)):
    """Execute a trade on the user's Binance account."""
    if not user.get("binance_connected"):
        raise HTTPException(status_code=400, detail="Binance not connected")
    
    svc = BinanceService(
        user["binance_api_key"],
        user["binance_api_secret"],
        user.get("binance_testnet", False)
    )
    
    try:
        if req.order_type == "MARKET":
            result = svc.place_market_order(req.symbol, req.side, req.quantity)
        elif req.order_type == "LIMIT":
            if not req.price:
                raise HTTPException(status_code=400, detail="Price required for LIMIT orders")
            result = svc.place_limit_order(req.symbol, req.side, req.quantity, req.price)
        else:
            raise HTTPException(status_code=400, detail="Order type must be MARKET or LIMIT")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_open_orders(
    symbol: Optional[str] = None, user: dict = Depends(get_current_user)
):
    """Get all open orders."""
    if not user.get("binance_connected"):
        raise HTTPException(status_code=400, detail="Binance not connected")
    
    svc = BinanceService(
        user["binance_api_key"],
        user["binance_api_secret"],
        user.get("binance_testnet", False)
    )
    
    return svc.get_open_orders(symbol)


@router.delete("/orders/{symbol}/{order_id}")
async def cancel_order(symbol: str, order_id: int, user: dict = Depends(get_current_user)):
    """Cancel an open order."""
    if not user.get("binance_connected"):
        raise HTTPException(status_code=400, detail="Binance not connected")
    
    svc = BinanceService(
        user["binance_api_key"],
        user["binance_api_secret"],
        user.get("binance_testnet", False)
    )
    
    try:
        return svc.cancel_order(symbol, order_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
