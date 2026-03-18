from pydantic import BaseModel
from typing import List


class UserProfile(BaseModel):
    id: int
    name: str
    risk_level: str
    horizon: str
    objective: str


class Asset(BaseModel):
    name: str
    percentage: float


class Portfolio(BaseModel):
    user_id: int
    assets: List[Asset]


class AgentResult(BaseModel):
    agent_name: str
    score: float
    summary: str
    details: List[str]


class RecommendationResponse(BaseModel):
    user_id: int
    investor_type: str
    diversification_score: float
    global_risk_score: float
    confidence_score: float
    recommendations: List[str]
    alerts: List[str]
    explanation: str
    agents: List[AgentResult]
