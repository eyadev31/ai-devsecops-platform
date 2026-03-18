"""
Dashboard Routes
================
Portfolio analytics, market regime, and performance data for the frontend.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from api.routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard & Analytics"])

# Import DAQ sessions store
from api.routes.daq import _daq_sessions


@router.get("/summary")
async def get_summary(user: dict = Depends(get_current_user)):
    """Get the latest portfolio recommendation summary."""
    # Find the most recent completed session
    sessions = user.get("daq_sessions", [])
    
    latest = None
    for sid in reversed(sessions):
        session = _daq_sessions.get(sid)
        if session and session["status"] == "completed":
            latest = session
            break
    
    if not latest:
        return {
            "has_recommendation": False,
            "message": "No portfolio recommendation yet. Complete a DAQ session first.",
        }
    
    a2 = latest.get("agent2_output", {})
    a3 = latest.get("agent3_output", {})
    a4 = latest.get("agent4_output", {})
    a1 = latest.get("agent1_output", {})
    a5 = latest.get("agent5_output", {})
    
    # Extract news intelligence
    news_intelligence = None
    if a5:
        news_intelligence = {
            "signal_type": a5.get("market_signal", {}).get("signal_type", "neutral"),
            "confidence": a5.get("market_signal", {}).get("confidence", 0),
            "signal_strength": a5.get("market_signal", {}).get("signal_strength", "weak"),
            "narrative": a5.get("market_signal", {}).get("narrative", ""),
            "primary_driver": a5.get("market_signal", {}).get("primary_driver", ""),
            "events": a5.get("event_detection", {}).get("events_detected", []),
            # Pass all returned articles (up to the limit processed, usually representing the last week)
            "articles": a5.get("articles", []),
            "temporal_sentiment": a5.get("temporal_sentiment", {}),
        }

    return {
        "has_recommendation": True,
        "session_id": latest["session_id"],
        "completed_at": latest.get("completed_at"),
        "market_regime": a1.get("market_regime", {}).get("primary_regime", "unknown"),
        "regime_confidence": a1.get("market_regime", {}).get("confidence", 0),
        "news_intelligence": news_intelligence,
        "risk_profile": {
            "risk_score": a2.get("risk_classification", {}).get("risk_score", 0),
            "behavioral_type": a2.get("risk_classification", {}).get("behavioral_type", "unknown"),
            "investor_personality": a2.get("risk_classification", {}).get("investor_personality", "unknown"),
        },
        "allocation": [{"asset": item.get("ticker", ""), **item} for item in a3.get("allocation", [])],
        "strategy": a3.get("strategy_rationale", {}).get("selected_strategy", "unknown"),
        "portfolio_metrics": {
            "expected_return": a3.get("portfolio_metrics", {}).get("expected_annual_return", 0),
            "volatility": a3.get("portfolio_metrics", {}).get("expected_annual_volatility", 0),
            "sharpe_ratio": a3.get("portfolio_metrics", {}).get("sharpe_ratio", 0),
            "cvar_95": a3.get("portfolio_metrics", {}).get("cvar_95", 0),
        },
        "monte_carlo": {
            "median_return": a3.get("monte_carlo", {}).get("median_annual_return", 0),
            "prob_loss": a3.get("monte_carlo", {}).get("probability_of_loss", 0),
            "max_drawdown": a3.get("monte_carlo", {}).get("median_max_drawdown", 0),
            "var_95": a3.get("monte_carlo", {}).get("var_95", 0),
        },
        "risk_verdict": a4.get("final_verdict", {}),
    }


@router.get("/regime")
async def get_current_regime(user: dict = Depends(get_current_user)):
    """Run Agent 1 to get current market regime (lightweight)."""
    from agents.agent1_macro import Agent1MacroIntelligence
    
    try:
        agent1 = Agent1MacroIntelligence()
        output = agent1.run(mock=True)  # Fast mock for regime check
        
        return {
            "regime": output.get("market_regime", {}),
            "volatility": output.get("volatility_state", {}),
            "systemic_risk": output.get("systemic_risk", {}),
            "macro": output.get("macro_environment", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics(user: dict = Depends(get_current_user)):
    """Get performance analytics specifically for the user's portfolio."""
    from backtest.backtest_data import TimelineGenerator
    import numpy as np

    # Find the user's latest completed session
    sessions = user.get("daq_sessions", [])
    latest = None
    for sid in reversed(sessions):
        session = _daq_sessions.get(sid)
        if session and session["status"] == "completed":
            latest = session
            break

    if not latest:
        raise HTTPException(status_code=400, detail="No portfolio to analyze. Run DAQ first.")

    try:
        a3 = latest.get("agent3_output", {})
        allocation = latest.get("final_portfolio") or a3.get("allocation", [])
        strategy_name = a3.get("strategy_rationale", {}).get("selected_strategy", "Custom User Portfolio")

        # Create a weights dictionary
        weights = {item["ticker"]: item.get("adjusted_weight", item.get("weight", 0)) for item in allocation}
        
        # Get historical windows (last 24 periods)
        windows = TimelineGenerator.generate_full_timeline()[-24:]

        returns = []
        equity_curve = [1.0]
        allocation_history = []
        regime_returns = {}

        for w in windows:
            # Calculate portfolio return for this month
            fwd = w.get("forward_returns", {})
            month_return = sum(weights.get(tk, 0) * fwd.get(tk, 0) for tk in weights)
            returns.append(month_return)
            equity_curve.append(equity_curve[-1] * (1 + month_return))

            allocation_history.append({
                "date": w.get("period_description", "Unknown Month"),
                "strategy": strategy_name
            })
            
            regime = w.get("regime_label", "Unknown")
            if regime not in regime_returns:
                regime_returns[regime] = []
            regime_returns[regime].append(month_return)

        # Compute metrics
        returns_np = np.array(returns)
        total_return = equity_curve[-1] - 1.0
        
        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown = np.max(drawdowns)

        # Sharpe (annualized)
        monthly_rf = 0.04 / 12
        excess_returns = returns_np - monthly_rf
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(12) if np.std(excess_returns) > 0 else 0

        win_rate = np.mean(returns_np > 0)

        # Regime adaptation: calculate average return per regime as a float
        regime_adaptation = {
            regime: float(np.mean(rets)) for regime, rets in regime_returns.items() if len(rets) > 0
        }

        return {
            "summary": {
                "total_return": float(total_return),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "sharpe_ratio": float(sharpe_ratio),
            },
            "equity_curve": [float(x) for x in equity_curve],
            "allocation_history": allocation_history,
            "regime_adaptation": regime_adaptation,
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
