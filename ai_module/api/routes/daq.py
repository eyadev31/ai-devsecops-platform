"""
DAQ Routes
==========
Dynamic Assessment Questionnaire — the interactive behavioral profiling flow.
Exposes Agent 1 (macro context) + Agent 2 (question generation + answer processing).
"""

import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from api.routes.auth import get_current_user, _users_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/daq", tags=["DAQ Questionnaire"])

# ── Session store ────────────────────────────────────────────
_daq_sessions: dict[str, dict] = {}


# ── Models ───────────────────────────────────────────────────
class StartDAQRequest(BaseModel):
    mock: bool = Field(default=False, description="Use mock data (no API keys needed)")


class SubmitAnswersRequest(BaseModel):
    session_id: str = Field(..., description="DAQ session ID from /start")
    answers: list[dict] = Field(..., description="User answers array")


# ── Routes ───────────────────────────────────────────────────
@router.post("/start")
async def start_daq(req: StartDAQRequest, user: dict = Depends(get_current_user)):
    """
    Start a DAQ session:
    1. Run Agent 1 (Macro Intelligence) to get market context
    2. Generate behavioral questions from Agent 2 Phase 1
    
    Returns the questions for the frontend to display.
    """
    from agents.agent1_macro import Agent1MacroIntelligence
    from agents.agent2_daq import Agent2BehavioralIntelligence

    session_id = f"daq_{uuid.uuid4().hex[:8]}"
    
    try:
        # ── Step 1: Run Agent 1 ──────────────────────────
        logger.info(f"[{session_id}] Starting Agent 1 (Macro Intelligence)...")
        agent1 = Agent1MacroIntelligence()
        agent1_output = agent1.run(mock=req.mock)
        
        # ── Step 2: Generate Questions ───────────────────
        logger.info(f"[{session_id}] Generating DAQ questions (Agent 2 Phase 1)...")
        agent2 = Agent2BehavioralIntelligence()
        question_output = agent2.generate_questions(
            agent1_output=agent1_output,
            num_questions=4,
        )
        
        # ── Store session ────────────────────────────────
        _daq_sessions[session_id] = {
            "session_id": session_id,
            "user_email": user["email"],
            "agent1_output": agent1_output,
            "question_output": question_output,
            "questions": question_output.get("questions", []),
            "started_at": datetime.utcnow().isoformat(),
            "status": "awaiting_answers",
        }
        
        # Extract regime info for the frontend
        regime = agent1_output.get("market_regime", {})
        
        # ── Normalize questions for mobile frontend ──────
        # LLM outputs: question_id, question_text, options[].text
        # Frontend expects: id, question, options[].label
        raw_questions = question_output.get("questions", [])
        normalized_questions = []
        for i, q in enumerate(raw_questions):
            normalized_options = []
            for opt in (q.get("options") or q.get("choices") or []):
                normalized_options.append({
                    "label": opt.get("label") or opt.get("text") or opt.get("option", f"Option"),
                    "value": opt.get("value") or chr(65 + len(normalized_options)),  # A, B, C, D
                    "score": opt.get("risk_signal") or opt.get("score") or 0.5,
                })
            
            normalized_questions.append({
                "id": q.get("id") or q.get("question_id") or f"q_{i+1}",
                "scenario": q.get("scenario", ""),
                "question": q.get("question") or q.get("question_text") or "",
                "options": normalized_options,
                "category": q.get("category") or "behavioral_assessment",
            })
        
        logger.info(f"[{session_id}] Normalized {len(normalized_questions)} questions for frontend")
        
        return {
            "session_id": session_id,
            "questions": normalized_questions,
            "market_context": {
                "regime": regime.get("primary_regime", "unknown"),
                "confidence": regime.get("confidence", 0),
                "stress_level": question_output.get("market_calibration", {}).get("stress_multiplier", 0),
            },
            "generation_method": question_output.get("generation_method", "unknown"),
        }
        
    except Exception as e:
        logger.error(f"[{session_id}] DAQ start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit")
async def submit_answers(req: SubmitAnswersRequest, user: dict = Depends(get_current_user)):
    """
    Submit user answers and run the full pipeline:
    1. Agent 2 Phase 2 (Behavioral Profiling)
    2. Agent 3 (Portfolio Optimization)
    3. Agent 4 (Risk Oversight CRO)
    
    Returns the complete portfolio recommendation.
    """
    from agents.agent2_daq import Agent2BehavioralIntelligence
    from agents.agent3_strategist import Agent3PortfolioStrategist
    from agents.agent4_supervisor import Agent4RiskSupervisor
    from agents.agent5_news import Agent5NewsIntelligence
    from schemas.agent2_output import Agent2Output
    from schemas.agent3_output import Agent3Output
    from schemas.agent4_output import Agent4Output
    import numpy as np
    import pandas as pd
    from datetime import datetime

    def aggressive_clean(obj):
        """Recursively convert NumPy/Pandas objects to standard Python types."""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {str(k): aggressive_clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [aggressive_clean(v) for v in obj]
        elif hasattr(obj, "model_dump"):
            return aggressive_clean(obj.model_dump(mode="json"))
        return obj

    session = _daq_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="DAQ session not found. Call /start first.")
    
    if session["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Not your session")
    
    try:
        agent1_output = aggressive_clean(session["agent1_output"])
        questions = session["questions"]
        question_output = session["question_output"]
        stress = question_output.get("market_calibration", {}).get("stress_multiplier", 0.5)
        
        # ── Agent 2 Phase 2: Process Answers ─────────────
        logger.info(f"[{req.session_id}] Processing answers (Agent 2 — Investor Profiling AI)...")
        agent2 = Agent2BehavioralIntelligence()
        agent2_dict = agent2.process_answers(
            questions=questions,
            answers=req.answers,
            agent1_output=agent1_output,
            session_id=req.session_id,
            market_stress=stress,
        )
        agent2_dict = aggressive_clean(agent2_dict)
        agent2_validated = Agent2Output.model_validate(agent2_dict).model_dump(mode="json")
        
        # ── Agent 5: News Intelligence AI ────────────────
        logger.info(f"[{req.session_id}] Running news sentiment analysis (Agent 5 — News Intelligence AI)...")
        try:
            agent5 = Agent5NewsIntelligence()
            agent5_output = agent5.run(mock=False, agent1_output=agent1_output)
            agent5_output = aggressive_clean(agent5_output)
            logger.info(
                f"[{req.session_id}] Agent 5 complete: "
                f"signal={agent5_output.get('market_signal', {}).get('signal_type', 'N/A')}, "
                f"articles={agent5_output.get('article_count', 0)}"
            )
        except Exception as e5:
            logger.warning(f"[{req.session_id}] Agent 5 failed (non-blocking): {e5}")
            agent5_output = {}
        
        # ── Agent 3: Portfolio Optimization ──────────────
        logger.info(f"[{req.session_id}] Running portfolio optimization (Agent 3 — Portfolio Optimization AI)...")
        agent3 = Agent3PortfolioStrategist()
        agent3_dict = agent3.run(
            agent1_output=agent1_output,
            agent2_output=agent2_validated,
        )
        agent3_dict = aggressive_clean(agent3_dict)
        agent3_validated = Agent3Output.model_validate(agent3_dict).model_dump(mode="json")
        
        # ── Agent 4: Risk Oversight ──────────────────────
        logger.info(f"[{req.session_id}] Running risk oversight (Agent 4 — Risk Control AI)...")
        agent4 = Agent4RiskSupervisor()
        agent4_dict = agent4.run(
            agent1_output=agent1_output,
            agent2_output=agent2_validated,
            agent3_output=agent3_validated,
        )
        agent4_dict = aggressive_clean(agent4_dict)
        agent4_validated = Agent4Output.model_validate(agent4_dict).model_dump(mode="json")
        
        # ── Update session with ALL 5 agents ─────────────
        session["status"] = "completed"
        session["completed_at"] = datetime.utcnow().isoformat()
        session["agent2_output"] = agent2_validated
        session["agent3_output"] = agent3_validated
        session["agent4_output"] = agent4_validated
        session["agent5_output"] = agent5_output  # ← CRITICAL: stored for dashboard
        _daq_sessions[req.session_id] = session
        
        # Track in user profile
        user.setdefault("daq_sessions", []).append(req.session_id)
        _users_db[user["email"]] = user
        
        return {
            "session_id": req.session_id,
            "status": "completed",
            "agent1_output": agent1_output,
            "agent2_output": agent2_validated,
            "agent3_output": agent3_validated,
            "agent4_output": agent4_validated,
            "agent5_output": agent5_output,
            "final_portfolio": agent3_validated.get("allocation", []),
        }
        
    except Exception as e:
        logger.error(f"[{req.session_id}] DAQ submit failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-mock")
async def submit_mock_answers(req: SubmitAnswersRequest, user: dict = Depends(get_current_user)):
    """
    Auto-generate simulated answers for the questions and run the full pipeline.
    Useful for testing the system end-to-end without manually answering questions.
    """
    from agents.agent2_daq import Agent2BehavioralIntelligence
    session = _daq_sessions.get(req.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="DAQ session not found.")
        
    questions = session["questions"]
    
    # Simulate realistic user answers based on a profile
    mock_answers = Agent2BehavioralIntelligence._simulate_answers(questions, profile="growth_seeker")
    
    req.answers = mock_answers
    return await submit_answers(req, user)


@router.get("/session/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    """Get a DAQ session's results."""
    session = _daq_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["user_email"] != user["email"]:
        raise HTTPException(status_code=403, detail="Not your session")
    
    # Return safe subset
    result = {
        "session_id": session["session_id"],
        "status": session["status"],
        "started_at": session["started_at"],
    }
    
    if session["status"] == "completed":
        a3 = session.get("agent3_output", {})
        a4 = session.get("agent4_output", {})
        result["allocation"] = a3.get("allocation", [])
        result["portfolio_metrics"] = a3.get("portfolio_metrics", {})
        result["risk_verdict"] = a4.get("final_verdict", {})
        result["completed_at"] = session.get("completed_at")
    
    return result


@router.get("/history")
async def get_history(user: dict = Depends(get_current_user)):
    """Get all DAQ sessions for the current user."""
    sessions = []
    for sid in user.get("daq_sessions", []):
        session = _daq_sessions.get(sid)
        if session:
            sessions.append({
                "session_id": session["session_id"],
                "status": session["status"],
                "started_at": session["started_at"],
                "completed_at": session.get("completed_at"),
            })
    return {"sessions": sessions}
