"""
Hybrid Intelligence Portfolio System — Main Entry Point
=========================================================
CLI entry point for running agents.

Usage:
  python main.py --agent 1              # Run Agent 1 with live data
  python main.py --agent 1 --mock       # Run Agent 1 with mock data
  python main.py --agent 5 --mock       # Run Agent 5 news sentiment
  python main.py --agent 1 --validate   # Validate API keys only
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config.settings import APIKeys, SystemMeta


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the system."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers = [handler]


def run_agent1(mock: bool = False) -> dict:
    """Run Agent 1 -- Macro & Market Intelligence System."""
    from agents.agent1_macro import Agent1MacroIntelligence

    agent = Agent1MacroIntelligence()
    return agent.run(mock=mock)


def run_agent2(mock: bool = False, agent1_json: str = None) -> dict:
    """Run Agent 2 -- Cognitive & Behavioral Profiling Agent (DAQ)."""
    from agents.agent2_daq import Agent2BehavioralIntelligence

    agent = Agent2BehavioralIntelligence()

    # Load Agent 1 output for context
    agent1_output = None
    if agent1_json:
        try:
            with open(agent1_json, "r", encoding="utf-8") as f:
                agent1_output = json.load(f)
            logging.info(f"Loaded Agent 1 context from: {agent1_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 1 context: {e}")

    if mock:
        return agent.run_mock(agent1_output=agent1_output)
    else:
        # Live mode: run Agent 1 first, then Agent 2
        if agent1_output is None:
            logging.info("No Agent 1 context provided. Running Agent 1 first...")
            agent1_output = run_agent1(mock=False)

        combined = agent.run_mock(agent1_output=agent1_output)
        return combined


def run_agent3(
    mock: bool = False,
    agent1_json: str = None,
    agent2_json: str = None,
) -> dict:
    """Run Agent 3 -- Strategic Allocation & Optimization Agent."""
    from agents.agent3_strategist import Agent3PortfolioStrategist

    agent = Agent3PortfolioStrategist()

    # Load upstream outputs
    agent1_output = None
    agent2_output = None

    if agent1_json:
        try:
            with open(agent1_json, "r", encoding="utf-8") as f:
                agent1_output = json.load(f)
            logging.info(f"Loaded Agent 1 context from: {agent1_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 1 context: {e}")

    if agent2_json:
        try:
            with open(agent2_json, "r", encoding="utf-8") as f:
                agent2_output = json.load(f)
            logging.info(f"Loaded Agent 2 profile from: {agent2_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 2 profile: {e}")

    return agent.run_mock(
        agent1_output=agent1_output,
        agent2_output=agent2_output,
    )


def run_agent4(
    mock: bool = False,
    agent1_json: str = None,
    agent2_json: str = None,
    agent3_json: str = None,
) -> dict:
    """Run Agent 4 -- Meta-Risk & Supervision Agent."""
    from agents.agent4_supervisor import Agent4RiskSupervisor

    agent = Agent4RiskSupervisor()

    agent1_output = None
    agent2_output = None
    agent3_output = None

    if agent1_json:
        try:
            with open(agent1_json, "r", encoding="utf-8") as f:
                agent1_output = json.load(f)
            logging.info(f"Loaded Agent 1 context from: {agent1_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 1 context: {e}")

    if agent2_json:
        try:
            with open(agent2_json, "r", encoding="utf-8") as f:
                agent2_output = json.load(f)
            logging.info(f"Loaded Agent 2 profile from: {agent2_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 2 profile: {e}")

    if agent3_json:
        try:
            with open(agent3_json, "r", encoding="utf-8") as f:
                agent3_output = json.load(f)
            logging.info(f"Loaded Agent 3 allocation from: {agent3_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 3 allocation: {e}")

    return agent.run_mock(
        agent1_output=agent1_output,
        agent2_output=agent2_output,
        agent3_output=agent3_output,
    )


def run_agent5(mock: bool = False, agent1_json: str = None) -> dict:
    """Run Agent 5 -- News Sentiment Intelligence Agent."""
    from agents.agent5_news import Agent5NewsIntelligence

    agent = Agent5NewsIntelligence()

    # Load Agent 1 output for market context
    agent1_output = None
    if agent1_json:
        try:
            with open(agent1_json, "r", encoding="utf-8") as f:
                agent1_output = json.load(f)
            logging.info(f"Loaded Agent 1 context from: {agent1_json}")
        except Exception as e:
            logging.warning(f"Could not load Agent 1 context: {e}")

    return agent.run(mock=mock, agent1_output=agent1_output)


def validate_keys():
    """Validate API key configuration."""
    print("\n+==========================================+")
    print("|    API KEY VALIDATION                    |")
    print("+==========================================+\n")

    status = APIKeys.validate()
    for key, configured in status.items():
        icon = "[OK]" if configured else "[X]"
        status_text = "Configured" if configured else "NOT CONFIGURED"
        print(f"  {icon} {key.upper():<15s} -- {status_text}")

    print()
    all_good = all(status.values())
    if all_good:
        print("  [OK] All API keys configured. System ready for live data.\n")
    else:
        missing = [k for k, v in status.items() if not v]
        print(f"  [!] Missing keys: {', '.join(missing)}")
        print("  -> Copy .env.example to .env and fill in your keys.\n")
        print("  You can still run with --mock to test without API keys.\n")

    return all_good


def main():
    parser = argparse.ArgumentParser(
        description=f"{SystemMeta.SYSTEM_NAME} v{SystemMeta.VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --agent 1              Run Agent 1 with live data
  python main.py --agent 1 --mock       Run Agent 1 with synthetic data
  python main.py --agent 4 --mock       Run Agent 4 risk oversight
  python main.py --agent 4 --mock --persist  Run + save to database
  python main.py --init-db              Initialize database tables + seed assets
        """,
    )

    parser.add_argument(
        "--agent", type=int, choices=[1, 2, 3, 4, 5],
        help="Agent number to run (1=Macro, 2=DAQ, 3=Allocation, 4=Risk, 5=News)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run with synthetic mock data (no API keys required)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate API key configuration only"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save output to JSON file"
    )
    parser.add_argument(
        "--agent1-json", type=str, default=None,
        help="Path to Agent 1 JSON output (for Agent 2/3/4 context)"
    )
    parser.add_argument(
        "--agent2-json", type=str, default=None,
        help="Path to Agent 2 JSON output (for Agent 3/4 context)"
    )
    parser.add_argument(
        "--agent3-json", type=str, default=None,
        help="Path to Agent 3 JSON output (for Agent 4 context)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Persist agent output to database"
    )
    parser.add_argument(
        "--init-db", action="store_true",
        help="Initialize database tables and seed assets"
    )
    parser.add_argument(
        "--user-email", type=str, default="investor@demo.com",
        help="User email for database persistence (default: investor@demo.com)"
    )
    
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Run the complete end-to-end Intelligent Orchestrator pipeline"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the 10-year rolling historical backtest engine (Step 5.2)"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run backtest + generate Performance Analytics Dashboard (HTML report)"
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting {SystemMeta.SYSTEM_NAME} v{SystemMeta.VERSION}")
    
    # ── Dashboard Mode ─────────────────────────────────
    if args.dashboard:
        from backtest.backtest_engine import BacktestEngine
        from dashboard.dashboard_generator import DashboardGenerator
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE ANALYTICS DASHBOARD — Generating...")
        logger.info("=" * 60)
        
        engine = BacktestEngine()
        
        try:
            metrics = engine.run_backtest()
            engine.print_summary(metrics)
            
            # Generate the dashboard
            output_file = args.output or "analytics_dashboard.html"
            gen = DashboardGenerator(metrics, engine.results)
            dashboard_path = gen.generate(output_file)
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"DASHBOARD GENERATED SUCCESSFULLY")
            logger.info(f"  Open in browser: {dashboard_path}")
            logger.info(f"{'=' * 60}")
            
            sys.exit(0)
        except Exception as e:
            logger.error(f"DASHBOARD GENERATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # ── Backtest Mode ───────────────────────────────────
    if args.backtest:
        from backtest.backtest_engine import BacktestEngine
        
        logger.info("Initializing 10-Year Rolling Historical Backtest...")
        engine = BacktestEngine()
        
        try:
            metrics = engine.run_backtest()
            engine.print_summary(metrics)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Backtest metrics saved to: {args.output}")
                
            sys.exit(0)
        except Exception as e:
            logger.error(f"BACKTEST ABORTED: {e}")
            sys.exit(1)
    
    # ── Orchestrator Mode (Brain Stem) ──────────────────
    if args.orchestrate:
        if not args.mock and not validate_keys():
            sys.exit(1)
            
        from orchestrator import IntelligenceOrchestrator, OrchestrationError
        orch = IntelligenceOrchestrator()
        
        try:
            results = orch.run_pipeline(mock=args.mock)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Full orchestration output saved to: {args.output}")
                
            if args.persist:
                logger.info("Persisting full session to database...")
                from data.dal import PortfolioDAL
                dal = PortfolioDAL()
                
                # Assuming dal handles an end-to-end generic save, or we save individually
                # To be precise, we'll save each component if persistence is configured
                dal.save_market_state(results["agent1_output"])
                dal.save_user_profile(args.user_email, results["agent2_output"])
                p_id = dal.save_portfolio(args.user_email, results["agent3_output"])
                if p_id:
                    dal.save_risk_verdict(p_id, results["agent4_output"])
                    
                logger.info("Database persistence complete.")
                
            sys.exit(0)
        except OrchestrationError as e:
            logger.error(f"PIPELINE ABORTED: {e}")
            sys.exit(1)

    # ── Single Agent Mode ──────────────────────────────
    # Handle --init-db
    if getattr(args, 'init_db', False):
        from data.database import init_db, seed_assets
        print("\n  Initializing database...")
        init_db()
        seed_assets()
        print("  [OK] Database initialized and assets seeded!")
        return

    if args.validate:
        validate_keys()
        return

    if not args.agent:
        print("\n  Error: --agent is required (unless using --init-db)")
        sys.exit(1)

    try:
        if args.agent == 1:
            output = run_agent1(mock=args.mock)
            title = "AGENT 1 OUTPUT -- Macro & Market Intelligence Context"

        elif args.agent == 2:
            output = run_agent2(mock=args.mock, agent1_json=args.agent1_json)
            title = "AGENT 2 OUTPUT -- Cognitive & Behavioral Profile"

        elif args.agent == 3:
            output = run_agent3(
                mock=args.mock,
                agent1_json=args.agent1_json,
                agent2_json=getattr(args, 'agent2_json', None),
            )
            title = "AGENT 3 OUTPUT -- Strategic Portfolio Allocation"

        elif args.agent == 4:
            output = run_agent4(
                mock=args.mock,
                agent1_json=args.agent1_json,
                agent2_json=getattr(args, 'agent2_json', None),
                agent3_json=getattr(args, 'agent3_json', None),
            )
            title = "AGENT 4 OUTPUT -- Meta-Risk & Supervision Verdict"

        elif args.agent == 5:
            output = run_agent5(
                mock=args.mock,
                agent1_json=args.agent1_json,
            )
            title = "AGENT 5 OUTPUT -- News Sentiment Intelligence"

        else:
            print(f"\n  Agent {args.agent} is not yet implemented.")
            print("  Available agents: 1 (Macro), 2 (DAQ), 3 (Portfolio), 4 (Risk), 5 (News)")
            sys.exit(1)

        # Print output
        print("\n" + "=" * 62)
        print(f"  {title}")
        print("=" * 62 + "\n")
        print(json.dumps(output, indent=2, default=str))

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n  [OK] Output saved to: {args.output}")

        # Persist to database if requested
        if getattr(args, 'persist', False):
            from data.database import init_db
            from data.dal import PipelineDAL
            init_db()  # Creates tables if not exists
            agent_outputs = {f"agent{args.agent}_output": output}
            # Load upstream outputs from files if provided
            for flag, key in [("agent1_json", "agent1_output"), ("agent2_json", "agent2_output"), ("agent3_json", "agent3_output")]:
                path = getattr(args, flag.replace('-', '_'), None)
                if path:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            agent_outputs[key] = json.load(f)
                    except Exception:
                        pass
            summary = PipelineDAL.persist_full_pipeline(
                email=args.user_email,
                **agent_outputs,
            )
            print(f"\n  [DB] Persisted to database: {json.dumps(summary, indent=2)}")

    except Exception as e:
        logging.error(f"Agent {args.agent} execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
