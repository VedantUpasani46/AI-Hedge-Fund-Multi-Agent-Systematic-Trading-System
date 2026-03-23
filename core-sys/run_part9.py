"""
AI Hedge Fund — Part 9: Cloud Production
==========================================
run_part9.py — Main Entry Point

Usage:
    # Docker operations (local):
    python run_part9.py --docker-build
    python run_part9.py --docker-run
    python run_part9.py --docker-compose-up
    python run_part9.py --docker-compose-down

    # AWS deployment:
    python run_part9.py --deploy                 # Full build → push → deploy
    python run_part9.py --aws-status             # ECS service status
    python run_part9.py --aws-logs               # Tail CloudWatch logs
    python run_part9.py --aws-costs              # Estimate monthly costs
    python run_part9.py --aws-alarms             # Create production alarms

    # Health checks:
    python run_part9.py --health                 # Local health check
    python run_part9.py --health-deep            # Full dependency check

    # Monitoring:
    python run_part9.py --llm-costs              # Show LLM API cost summary
    python run_part9.py --structured-log-test    # Test JSON logging

    # Process management:
    python run_part9.py --start-risk-monitor     # Start risk monitor process
    python run_part9.py --start-strategy-engine  # Start strategy engine process

    # Full demo:
    python run_part9.py --demo
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import cfg, setup_logging

logger = setup_logging()


def validate_environment() -> bool:
    print("\n" + "═" * 65)
    print("  Part 9: Cloud Production — Environment Check")
    print("═" * 65)

    required = [
        ("numpy",     "pip install numpy"),
        ("pandas",    "pip install pandas"),
        ("yfinance",  "pip install yfinance"),
    ]
    deploy = [
        ("boto3",     "pip install boto3       # AWS SDK"),
        ("docker",    "pip install docker      # Docker Python SDK"),
    ]
    infra = []

    # Check Docker
    docker_ok = subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0

    # Check AWS CLI
    aws_ok = subprocess.run(
        ["aws", "--version"], capture_output=True
    ).returncode == 0

    all_ok = True
    print("\n  Python packages (core):")
    for pkg, install in required:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg} — {install}")
            all_ok = False

    print("\n  Python packages (deployment):")
    for pkg, install in deploy:
        try:
            __import__(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ○ {pkg} — {install}")

    print("\n  System tools:")
    print(f"    {'✓' if docker_ok else '○'} Docker {'(running)' if docker_ok else '(not running — needed for container builds)'}")
    print(f"    {'✓' if aws_ok else '○'} AWS CLI {'(configured)' if aws_ok else '(not installed — needed for AWS deployment)'}")

    aws_id = os.getenv("AWS_ACCOUNT_ID", "")
    has_aws = bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))
    print(f"\n  AWS credentials: {'✓ configured' if has_aws else '○ not set (needed for cloud deployment)'}")
    if aws_id:
        print(f"  AWS Account ID:  {aws_id}")

    return all_ok


def docker_build():
    print(f"\n{'═'*65}")
    print("  Docker Build")
    print(f"{'═'*65}")

    result = subprocess.run(
        ["docker", "build", "-t", "hedgefund-api:latest", "-f", "Dockerfile", "."],
        cwd = str(Path(__file__).parent),
    )
    if result.returncode == 0:
        print("\n  ✓ Build successful")
        print("  Image: hedgefund-api:latest")
        print("  Run with: python run_part9.py --docker-run")
    else:
        print("\n  ✗ Build failed")


def docker_run():
    print(f"\n{'═'*65}")
    print("  Docker Run (local)")
    print(f"{'═'*65}")

    env_file = Path(__file__).parent / ".env"
    env_args  = ["--env-file", str(env_file)] if env_file.exists() else [
        "-e", "ENV=development",
        "-e", "API_KEYS=dev-key-1",
    ]

    cmd = [
        "docker", "run",
        "--rm", "-it",
        "-p", "8000:8000",
        "-v", f"{Path(__file__).parent}/db:/app/db",
        "-v", f"{Path(__file__).parent}/logs:/app/logs",
        *env_args,
        "hedgefund-api:latest",
    ]

    print(f"\n  Running: hedgefund-api:latest")
    print(f"  API:     http://localhost:8000")
    print(f"  Docs:    http://localhost:8000/docs")
    print(f"  Ctrl+C to stop\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  Container stopped.")


def docker_compose_up():
    print(f"\n{'═'*65}")
    print("  Docker Compose — Full Stack")
    print(f"{'═'*65}")
    print("\n  Starting: api, monitor, strategy, redis...")
    print("  Logs: docker-compose logs -f api\n")

    subprocess.run(
        ["docker-compose", "up", "-d", "--build"],
        cwd=str(Path(__file__).parent),
    )
    print("\n  Services running:")
    subprocess.run(
        ["docker-compose", "ps"],
        cwd=str(Path(__file__).parent),
    )


def docker_compose_down():
    subprocess.run(
        ["docker-compose", "down"],
        cwd=str(Path(__file__).parent),
    )
    print("  Stack stopped.")


def aws_deploy():
    from src.deploy.aws_deploy import AWSConfig, full_deploy
    cfg_aws = AWSConfig.from_env()
    if not cfg_aws.account_id:
        print("\n  ✗ AWS_ACCOUNT_ID not set. Configure with:")
        print("    export AWS_ACCOUNT_ID=123456789012")
        print("    export AWS_REGION=us-east-1")
        print("    export ECR_REPO=hedgefund-api")
        print("    export ECS_CLUSTER=hedgefund-cluster")
        print("    export ECS_SERVICE=hedgefund-api")
        return
    full_deploy(cfg_aws)


def aws_status():
    print(f"\n{'═'*65}")
    print("  AWS ECS Service Status")
    print(f"{'═'*65}")
    try:
        from src.deploy.aws_deploy import AWSConfig, ECSManager
        ecs    = ECSManager(AWSConfig.from_env())
        status = ecs.get_service_status()
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(f"  Error: {e}")
        print("  Ensure boto3 is installed and AWS credentials are configured.")


def aws_logs():
    print(f"\n{'═'*65}")
    print("  CloudWatch Log Streaming")
    print(f"{'═'*65}")
    try:
        from src.deploy.aws_deploy import AWSConfig, CloudWatchManager
        cw = CloudWatchManager(AWSConfig.from_env())
        cw.tail_logs(follow=True, since_minutes=30)
    except Exception as e:
        print(f"  Error: {e}")


def aws_costs():
    from src.deploy.aws_deploy import estimate_monthly_costs
    print(f"\n{'═'*65}")
    print("  Monthly AWS Cost Estimate (us-east-1, t3.medium)")
    print(f"{'═'*65}")
    costs = estimate_monthly_costs()
    print(f"\n  {'Component':<30} {'Cost/month':>12}")
    print(f"  {'─'*44}")
    for item, cost in costs.items():
        if item != "total":
            label = item.replace("_", " ").title()
            print(f"  {label:<30} ${cost:>10.2f}")
    print(f"  {'─'*44}")
    print(f"  {'TOTAL':<30} ${costs['total']:>10.2f}")
    print(f"\n  Note: Excludes LLM API costs (~$0.50-2.00/day for Claude Sonnet)")
    print(f"  Upgrade path: t3.medium → t3.large at ~$50M AUM")


def aws_alarms():
    try:
        from src.deploy.aws_deploy import AWSConfig, CloudWatchManager
        cw      = CloudWatchManager(AWSConfig.from_env())
        results = cw.setup_fund_alarms()
        print(f"\n{'═'*65}")
        print("  CloudWatch Alarms Created")
        print(f"{'═'*65}")
        for alarm, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {alarm}")
    except Exception as e:
        print(f"  Error: {e}")


def health_check():
    from src.monitoring.monitoring import HealthChecker
    checker = HealthChecker()

    print(f"\n{'═'*65}")
    print("  System Health Check")
    print(f"{'═'*65}")

    print("\n  Database:")
    for name, status in checker.check_database().items():
        icon = "✓" if status.get("ok") else "✗"
        print(f"    {icon} {name}: {status['status']}" +
              (f" ({status.get('size_mb')}MB)" if status.get("size_mb") else ""))

    print("\n  Market data:")
    md = checker.check_market_data()
    print(f"    {'✓' if md['ok'] else '✗'} {md['status']}" +
          (f" (SPY=${md.get('last_price')}, {md.get('latency_ms')}ms)" if md["ok"] else ""))

    print("\n  Disk space:")
    disk = checker.check_disk_space()
    print(f"    {'✓' if disk['ok'] else '⚠'} {disk.get('pct_used')}% used, "
          f"{disk.get('free_gb')}GB free")

    print("\n  Risk monitor:")
    rm = checker.check_risk_monitor()
    print(f"    {'✓' if rm['ok'] else '○'} {rm['status']}" +
          (f" (age={rm.get('age_seconds')}s)" if rm.get("age_seconds") else ""))


def show_llm_costs():
    from src.monitoring.monitoring import LLMCostTracker
    tracker = LLMCostTracker()
    summary = tracker.cost_summary()
    by_agent= tracker.cost_by_agent()

    print(f"\n{'═'*65}")
    print("  LLM API Cost Summary")
    print(f"{'═'*65}")
    print(f"\n  Period       Cost")
    print(f"  {'─'*30}")
    for period, cost in summary.items():
        print(f"  {period:<15} ${cost:>8.5f}")

    if by_agent:
        print(f"\n  By Agent (7 days):")
        for agent, cost in by_agent.items():
            print(f"  {agent:<20} ${cost:.4f}")
    else:
        print(f"\n  No LLM calls recorded yet.")


def test_structured_logging():
    from src.monitoring.monitoring import get_monitor_logger, timer
    import time as _time

    log = get_monitor_logger("test_service")
    print(f"\n  Writing structured logs to: {log._log_file}")
    print()

    log.info("system_start", message="Part 9 monitoring test", version="1.0.0")
    log.metric("portfolio_nav", 1_042_500.00, portfolio_id="FUND_001")
    log.metric("var_95_pct",    0.0142,        portfolio_id="FUND_001")
    log.warning("var_warning",  message="VaR approaching limit", value=0.0142, limit=0.02)

    with timer("test_operation", log=log):
        _time.sleep(0.01)

    print("  Sample log entries (JSON lines):")
    if log._log_file and log._log_file.exists():
        lines = log._log_file.read_text().strip().split("\n")
        for line in lines[-4:]:
            print(f"  {line}")
    print("\n  ✓ Structured logging working")


def start_risk_monitor():
    print("  Starting risk monitor process (Ctrl+C to stop)...")
    from src.deploy.process_manager import RiskMonitorProcess
    RiskMonitorProcess().run()


def start_strategy_engine():
    print("  Starting strategy engine process (Ctrl+C to stop)...")
    from src.deploy.process_manager import StrategyEngineProcess
    StrategyEngineProcess().run()


def run_demo():
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'+'  PART 9: CLOUD PRODUCTION DEMO'.center(58)+'║'}")
    print(f"{'╚'+'═'*58+'╝'}")

    print("\n[1/5] Health checks...")
    health_check()

    print("\n[2/5] Structured logging...")
    test_structured_logging()

    print("\n[3/5] LLM cost tracking...")
    show_llm_costs()

    print("\n[4/5] AWS cost estimate...")
    aws_costs()

    print("\n[5/5] Deployment commands available:")
    print("  docker build -t hedgefund-api .")
    print("  docker-compose up -d")
    print("  python run_part9.py --deploy  (requires AWS credentials)")

    print(f"\n  Full production stack summary:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Service          Port   Description                │")
    print(f"  │  ─────────────────────────────────────────────────  │")
    print(f"  │  FastAPI API      8000   REST + WebSocket           │")
    print(f"  │  Risk Monitor     -      Background thread          │")
    print(f"  │  Strategy Engine  -      AgentCoordinator loop      │")
    print(f"  │  Redis            6379   MessageBus backend         │")
    print(f"  │  Nginx            80/443 Reverse proxy + TLS        │")
    print(f"  │  CloudWatch       -      Logs + metrics + alarms    │")
    print(f"  └─────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(
        description="AI Hedge Fund — Part 9: Cloud Production"
    )
    parser.add_argument("--docker-build",       action="store_true")
    parser.add_argument("--docker-run",         action="store_true")
    parser.add_argument("--docker-compose-up",  action="store_true")
    parser.add_argument("--docker-compose-down",action="store_true")
    parser.add_argument("--deploy",             action="store_true")
    parser.add_argument("--aws-status",         action="store_true")
    parser.add_argument("--aws-logs",           action="store_true")
    parser.add_argument("--aws-costs",          action="store_true")
    parser.add_argument("--aws-alarms",         action="store_true")
    parser.add_argument("--health",             action="store_true")
    parser.add_argument("--llm-costs",          action="store_true")
    parser.add_argument("--structured-log-test",action="store_true")
    parser.add_argument("--start-risk-monitor", action="store_true")
    parser.add_argument("--start-strategy-engine", action="store_true")
    parser.add_argument("--demo",               action="store_true")
    parser.add_argument("--validate",           action="store_true")
    args = parser.parse_args()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  AI HEDGE FUND — PART 9: CLOUD PRODUCTION".center(58) + "║")
    print("║" + f"  {datetime.now():%Y-%m-%d %H:%M:%S}".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    validate_environment()

    if args.validate:                     return
    elif args.docker_build:               docker_build()
    elif args.docker_run:                 docker_run()
    elif getattr(args,"docker_compose_up",False):   docker_compose_up()
    elif getattr(args,"docker_compose_down",False):  docker_compose_down()
    elif args.deploy:                     aws_deploy()
    elif getattr(args,"aws_status",False):aws_status()
    elif getattr(args,"aws_logs",False):  aws_logs()
    elif getattr(args,"aws_costs",False): aws_costs()
    elif getattr(args,"aws_alarms",False):aws_alarms()
    elif args.health:                     health_check()
    elif getattr(args,"llm_costs",False): show_llm_costs()
    elif getattr(args,"structured_log_test",False): test_structured_logging()
    elif getattr(args,"start_risk_monitor",False):  start_risk_monitor()
    elif getattr(args,"start_strategy_engine",False): start_strategy_engine()
    elif args.demo:                       run_demo()
    else:
        print("\n  No command — running demo")
        run_demo()

    print("\n✅ Part 9 complete.")
    print("   Next: Part 10 — Fund Operations & Compliance")


if __name__ == "__main__":
    main()
