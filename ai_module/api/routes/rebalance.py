"""
Auto-Rebalance API Route
POST /api/portfolio/rebalance — Execute trades to match Agent 3/4 recommended allocation
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from api.routes.auth import get_current_user
from api.services.binance_service import BinanceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["rebalance"])

# In-memory store for latest recommendations (in production, use DB)
_latest_recommendations: dict = {}


class RebalanceRequest(BaseModel):
    session_id: Optional[str] = None
    confirm: bool = True


class TradeOrder(BaseModel):
    asset: str
    side: str  # BUY or SELL
    quantity: float
    symbol: str
    estimated_usd: float


class RebalanceResponse(BaseModel):
    success: bool
    trades_executed: List[TradeOrder]
    message: str
    new_portfolio_value: Optional[float] = None


def store_recommendation(user_email: str, allocation: list):
    """Called by DAQ route after pipeline completion to store latest allocation."""
    _latest_recommendations[user_email] = allocation


@router.post("/rebalance", response_model=RebalanceResponse)
async def rebalance_portfolio(
    request: RebalanceRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Auto-rebalance: compare current Binance holdings with AI-recommended
    allocation and execute the necessary trades.
    """
    user_email = current_user.get("email", "")

    # 1. Get Binance service
    binance_keys = current_user.get("binance_keys", {})
    if not binance_keys:
        raise HTTPException(status_code=400, detail="Binance not connected")

    binance = BinanceService(
        api_key=binance_keys["api_key"],
        api_secret=binance_keys["api_secret"],
        testnet=binance_keys.get("testnet", True),
    )

    # 2. Get latest recommended allocation
    recommendation = _latest_recommendations.get(user_email)
    if not recommendation:
        raise HTTPException(
            status_code=400,
            detail="No AI recommendation found. Run the AI Advisor first."
        )

    # 3. Get current portfolio
    try:
        account_info = binance.get_account_info()
        portfolio_value = binance.get_portfolio_value()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance data: {str(e)}")

    total_value = portfolio_value.get("total_value_usdt", 0)
    current_assets = {
        b["asset"]: float(b["free"])
        for b in account_info.get("balances", [])
        if float(b["free"]) > 0
    }

    # 4. Calculate required trades
    trades_to_execute: List[TradeOrder] = []

    for alloc in recommendation:
        asset = alloc.get("asset", "")
        target_weight = alloc.get("weight", 0)
        target_usd = total_value * target_weight

        # Get current value of this asset
        current_qty = current_assets.get(asset, 0)
        # Simplified: assume we have price data
        # In production, fetch live price from Binance
        symbol = f"{asset}USDT" if asset != "USDT" else None

        if symbol and target_weight > 0:
            try:
                # Get current price
                ticker = binance.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker["price"])
                current_usd = current_qty * current_price
                delta_usd = target_usd - current_usd

                if abs(delta_usd) > 10:  # Min $10 trade
                    side = "BUY" if delta_usd > 0 else "SELL"
                    quantity = abs(delta_usd) / current_price

                    trades_to_execute.append(TradeOrder(
                        asset=asset,
                        side=side,
                        quantity=round(quantity, 6),
                        symbol=symbol,
                        estimated_usd=abs(delta_usd),
                    ))
            except Exception as e:
                logger.warning(f"Skipping {asset}: {str(e)}")

    # 5. Execute trades (if confirmed)
    executed_trades = []
    if request.confirm and trades_to_execute:
        for trade in trades_to_execute:
            try:
                result = binance.place_order(
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    order_type="MARKET",
                )
                executed_trades.append(trade)
                logger.info(f"Executed: {trade.side} {trade.quantity} {trade.asset}")
            except Exception as e:
                logger.error(f"Failed to execute {trade.asset}: {str(e)}")

    # 6. Get updated portfolio value
    new_value = None
    try:
        new_portfolio = binance.get_portfolio_value()
        new_value = new_portfolio.get("total_value_usdt")
    except Exception:
        pass

    return RebalanceResponse(
        success=len(executed_trades) > 0 or len(trades_to_execute) == 0,
        trades_executed=executed_trades,
        message=f"Executed {len(executed_trades)}/{len(trades_to_execute)} trades",
        new_portfolio_value=new_value,
    )
