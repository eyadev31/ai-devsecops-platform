"""
Binance Service
===============
Full Binance integration: read balances, positions, and execute trades.
Uses the python-binance SDK for authenticated API calls.
"""

import logging
import os
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceService:
    """
    Manages Binance API operations for a user's account.
    Supports: balance queries, position tracking, and order execution.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
        else:
            self.client = Client(api_key, api_secret)
        
        logger.info(f"Binance client initialized (testnet={testnet})")

    def get_account_info(self) -> dict:
        """Get full account information including all balances."""
        try:
            account = self.client.get_account()
            return {
                "can_trade": account.get("canTrade", False),
                "can_withdraw": account.get("canWithdraw", False),
                "account_type": account.get("accountType", "SPOT"),
                "balances": [
                    {
                        "asset": b["asset"],
                        "free": float(b["free"]),
                        "locked": float(b["locked"]),
                        "total": float(b["free"]) + float(b["locked"]),
                    }
                    for b in account.get("balances", [])
                    if float(b["free"]) > 0 or float(b["locked"]) > 0
                ],
            }
        except BinanceAPIException as e:
            logger.error(f"Binance account error: {e}")
            raise

    def get_portfolio_value(self) -> dict:
        """Get total portfolio value in USDT with per-asset breakdown."""
        try:
            account = self.get_account_info()
            balances = account["balances"]
            
            portfolio = []
            total_value = 0.0

            for bal in balances:
                asset = bal["asset"]
                total_qty = bal["total"]
                
                if asset in ("USDT", "BUSD", "USD"):
                    price_usdt = 1.0
                elif total_qty > 0:
                    try:
                        ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price_usdt = float(ticker["price"])
                    except Exception:
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}BUSD")
                            price_usdt = float(ticker["price"])
                        except Exception:
                            price_usdt = 0.0
                else:
                    price_usdt = 0.0

                value = total_qty * price_usdt
                total_value += value
                
                portfolio.append({
                    "asset": asset,
                    "quantity": round(total_qty, 8),
                    "price_usdt": round(price_usdt, 4),
                    "value_usdt": round(value, 2),
                })

            # Sort by value
            portfolio.sort(key=lambda x: x["value_usdt"], reverse=True)
            
            # Compute weights
            for p in portfolio:
                p["weight"] = round(p["value_usdt"] / total_value, 4) if total_value > 0 else 0

            return {
                "total_value_usdt": round(total_value, 2),
                "num_assets": len(portfolio),
                "positions": portfolio,
            }
        except BinanceAPIException as e:
            logger.error(f"Binance portfolio error: {e}")
            raise

    def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 10) -> list:
        """Get recent trades for a symbol."""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            return [
                {
                    "symbol": t["symbol"],
                    "side": "BUY" if t["isBuyer"] else "SELL",
                    "price": float(t["price"]),
                    "qty": float(t["qty"]),
                    "commission": float(t["commission"]),
                    "time": t["time"],
                }
                for t in trades
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance trades error: {e}")
            return []

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Amount to trade
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="MARKET",
                quantity=quantity,
            )
            logger.info(f"Order placed: {side} {quantity} {symbol}")
            return {
                "order_id": order["orderId"],
                "symbol": order["symbol"],
                "side": order["side"],
                "status": order["status"],
                "executed_qty": float(order["executedQty"]),
                "fills": order.get("fills", []),
            }
        except BinanceAPIException as e:
            logger.error(f"Binance order error: {e}")
            raise

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Place a limit order."""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="LIMIT",
                timeInForce="GTC",
                quantity=quantity,
                price=str(price),
            )
            logger.info(f"Limit order placed: {side} {quantity} {symbol} @ {price}")
            return {
                "order_id": order["orderId"],
                "symbol": order["symbol"],
                "side": order["side"],
                "status": order["status"],
                "price": float(order["price"]),
                "quantity": float(order["origQty"]),
            }
        except BinanceAPIException as e:
            logger.error(f"Binance limit order error: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get all open orders."""
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            return [
                {
                    "order_id": o["orderId"],
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "type": o["type"],
                    "price": float(o["price"]),
                    "quantity": float(o["origQty"]),
                    "status": o["status"],
                }
                for o in orders
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance open orders error: {e}")
            return []

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            return {"status": "cancelled", "order_id": order_id}
        except BinanceAPIException as e:
            logger.error(f"Binance cancel error: {e}")
            raise

    @staticmethod
    def validate_keys(api_key: str, api_secret: str, testnet: bool = False) -> dict:
        """Validate Binance API keys without storing them."""
        try:
            client = Client(api_key, api_secret, testnet=testnet)
            account = client.get_account()
            return {
                "valid": True,
                "can_trade": account.get("canTrade", False),
                "account_type": account.get("accountType", "SPOT"),
            }
        except BinanceAPIException as e:
            return {"valid": False, "error": str(e)}
        except Exception as e:
            return {"valid": False, "error": str(e)}
