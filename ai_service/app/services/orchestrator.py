from app.agents.profiling_agent import run_profile_analysis
from app.agents.portfolio_agent import run_portfolio_analysis
from app.agents.recommendation_agent import run_recommendation
from app.agents.risk_agent import run_risk_analysis
from app.agents.explanation_agent import run_explanation


def run_all_agents(profile: dict, portfolio: dict) -> dict:
    profile_result = run_profile_analysis(profile)
    portfolio_result = run_portfolio_analysis(portfolio)
    recommendation_result = run_recommendation(profile_result, portfolio)
    risk_result = run_risk_analysis(portfolio)

    explanation_result = run_explanation(
        investor_type=profile_result["investor_type"],
        diversification_score=portfolio_result["diversification_score"],
        risk_score=risk_result["global_risk_score"],
        recommendations=recommendation_result["recommendations"],
    )

    confidence_score = round(
        (portfolio_result["diversification_score"] + 10 - risk_result["global_risk_score"]) / 2,
        1,
    )

    return {
        "user_id": profile["id"],
        "investor_type": profile_result["investor_type"],
        "diversification_score": portfolio_result["diversification_score"],
        "global_risk_score": risk_result["global_risk_score"],
        "confidence_score": confidence_score,
        "recommendations": recommendation_result["recommendations"],
        "alerts": risk_result["alerts"],
        "explanation": explanation_result["explanation"],
        "agents": [
            profile_result,
            portfolio_result,
            recommendation_result,
            risk_result,
            explanation_result,
        ],
    }
