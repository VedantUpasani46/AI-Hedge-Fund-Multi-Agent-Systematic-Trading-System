"""
Basic health check tests — run by CI/CD on every push.
No LLM calls, no broker connections.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))


def test_monitoring_imports():
    from src.monitoring.monitoring import HealthChecker, LLMCostTracker, get_monitor_logger
    assert HealthChecker is not None
    assert LLMCostTracker is not None


def test_structured_logger():
    from src.monitoring.monitoring import get_monitor_logger
    log = get_monitor_logger("test")
    # Should not raise
    log.info("test_event", message="test", value=42)


def test_health_checker_disk():
    from src.monitoring.monitoring import HealthChecker
    checker = HealthChecker()
    result  = checker.check_disk_space()
    assert "ok" in result
    assert "pct_used" in result


def test_llm_cost_tracker():
    from src.monitoring.monitoring import LLMCostTracker
    tracker = LLMCostTracker()
    summary = tracker.cost_summary()
    assert "last_hour" in summary
    assert "last_day" in summary


def test_aws_config():
    from src.deploy.aws_deploy import AWSConfig, estimate_monthly_costs
    cfg = AWSConfig()
    assert cfg.region == "us-east-1"

    costs = estimate_monthly_costs()
    assert "total" in costs
    assert costs["total"] > 0


def test_process_manager_imports():
    from src.deploy.process_manager import RiskMonitorProcess, StrategyEngineProcess
    assert RiskMonitorProcess is not None
    assert StrategyEngineProcess is not None
