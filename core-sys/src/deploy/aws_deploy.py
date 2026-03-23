"""
AI Hedge Fund — Part 9: Cloud Production
==========================================
aws_deploy.py — AWS Infrastructure & Deployment Utilities

Manages all AWS resources for the hedge fund system.

Architecture (cost-optimised for a sub-$100M fund):
    ECR:            Container registry (push Docker images here)
    EC2 t3.medium:  Single instance running all containers ($30/month)
    EBS gp3 50GB:   Persistent storage for SQLite DBs ($4/month)
    CloudWatch:     Logs, metrics, alarms ($5-15/month)
    Route53:        DNS for api.yourfund.com ($1/month)
    ACM:            Free TLS certificate
    Total:          ~$40-60/month for the full production stack

Upgrade path (when AUM grows):
    $10M-50M:   Single t3.medium → t3.large (still single instance)
    $50M-100M:  ECS Fargate (auto-scaling, pay-per-task)
    $100M+:     Multi-region with RDS PostgreSQL, ElasticCache Redis

Functions:
    aws_setup()         — First-time infrastructure provisioning
    deploy()            — Build + push + update running container
    check_health()      — Verify all services are healthy on AWS
    get_costs()         — Estimate monthly AWS costs
    tail_logs()         — Stream CloudWatch logs to terminal
    create_alarm()      — Set up CloudWatch alarm for monitoring
    rotate_api_keys()   — Rotate Secrets Manager API keys

Prerequisites:
    pip install boto3
    aws configure  (or use IAM role on EC2 — preferred for production)

IAM permissions required:
    ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability,
    ecr:GetDownloadUrlForLayer, ecr:BatchGetImage, ecr:PutImage,
    ecr:InitiateLayerUpload, ecr:UploadLayerPart, ecr:CompleteLayerUpload,
    ecs:UpdateService, ecs:DescribeServices,
    logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents,
    logs:GetLogEvents, logs:FilterLogEvents,
    cloudwatch:PutMetricAlarm, cloudwatch:PutMetricData,
    secretsmanager:GetSecretValue, secretsmanager:PutSecretValue
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.aws")


# ─────────────────────────────────────────────────────────────────────────────
# AWS configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AWSConfig:
    """AWS deployment configuration."""
    region:          str = "us-east-1"
    account_id:      str = ""
    ecr_repo:        str = "hedgefund-api"
    ecs_cluster:     str = "hedgefund-cluster"
    ecs_service:     str = "hedgefund-api"
    ec2_instance_id: str = ""
    log_group:       str = "/hedgefund/app"
    secret_name:     str = "hedgefund/api-keys"
    image_tag:       str = "latest"

    @classmethod
    def from_env(cls) -> "AWSConfig":
        return cls(
            region          = os.getenv("AWS_REGION",       "us-east-1"),
            account_id      = os.getenv("AWS_ACCOUNT_ID",   ""),
            ecr_repo        = os.getenv("ECR_REPO",         "hedgefund-api"),
            ecs_cluster     = os.getenv("ECS_CLUSTER",      "hedgefund-cluster"),
            ecs_service     = os.getenv("ECS_SERVICE",      "hedgefund-api"),
            ec2_instance_id = os.getenv("EC2_INSTANCE_ID",  ""),
            log_group       = os.getenv("LOG_GROUP",        "/hedgefund/app"),
            image_tag       = os.getenv("IMAGE_TAG",        "latest"),
        )

    @property
    def ecr_uri(self) -> str:
        return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{self.ecr_repo}"


# ─────────────────────────────────────────────────────────────────────────────
# ECR — Container registry
# ─────────────────────────────────────────────────────────────────────────────

class ECRManager:
    """
    Manages container images in AWS Elastic Container Registry.

    Build → Tag → Push → Deploy workflow:
        1. docker build -t hedgefund-api:local .
        2. docker tag hedgefund-api:local {ecr_uri}:{tag}
        3. docker push {ecr_uri}:{tag}
        4. Update ECS service to pull new image
    """

    def __init__(self, config: AWSConfig):
        self.cfg = config
        self._boto3_ecr = None

    def _ecr(self):
        if not self._boto3_ecr:
            try:
                import boto3
                self._boto3_ecr = boto3.client("ecr", region_name=self.cfg.region)
            except ImportError:
                raise ImportError("boto3 not installed. Run: pip install boto3")
        return self._boto3_ecr

    def authenticate(self) -> bool:
        """Log Docker into ECR."""
        try:
            token  = self._ecr().get_authorization_token()
            auth   = token["authorizationData"][0]
            endpoint = auth["proxyEndpoint"]

            cmd = f"aws ecr get-login-password --region {self.cfg.region} | docker login --username AWS --password-stdin {endpoint}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("ECR authentication successful")
                return True
            else:
                logger.error(f"ECR auth failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"ECR authenticate failed: {e}")
            return False

    def build_and_push(
        self,
        tag:         str = "latest",
        dockerfile:  str = "Dockerfile",
        context:     str = ".",
        also_tag_sha: bool = True,
    ) -> Optional[str]:
        """
        Build Docker image and push to ECR.

        Returns the full ECR URI with tag, or None on failure.
        """
        image_uri = f"{self.cfg.ecr_uri}:{tag}"

        print(f"\n  Building image: hedgefund-api:{tag}")
        print(f"  Context: {context}")

        # Build
        build_cmd = [
            "docker", "build",
            "-t", f"hedgefund-api:{tag}",
            "-f", dockerfile,
            "--label", f"build-time={datetime.now().isoformat()}",
            "--label", f"git-commit={self._get_git_sha()}",
            context,
        ]
        result = subprocess.run(build_cmd, capture_output=False)
        if result.returncode != 0:
            logger.error("Docker build failed")
            return None

        print(f"  ✓ Build complete")

        # Authenticate
        if not self.authenticate():
            return None

        # Tag
        subprocess.run(
            ["docker", "tag", f"hedgefund-api:{tag}", image_uri],
            check=True,
        )

        # Also tag with git SHA for auditability
        sha = self._get_git_sha()
        if also_tag_sha and sha:
            sha_uri = f"{self.cfg.ecr_uri}:{sha[:8]}"
            subprocess.run(
                ["docker", "tag", f"hedgefund-api:{tag}", sha_uri],
                check=True,
            )

        # Push
        print(f"  Pushing to {image_uri}...")
        result = subprocess.run(
            ["docker", "push", image_uri],
            capture_output=False,
        )
        if result.returncode != 0:
            logger.error("Docker push failed")
            return None

        if also_tag_sha and sha:
            subprocess.run(["docker", "push", sha_uri], capture_output=False)

        print(f"  ✓ Pushed: {image_uri}")
        return image_uri

    def list_images(self) -> List[Dict]:
        """List images in the ECR repository."""
        try:
            result = self._ecr().list_images(repositoryName=self.cfg.ecr_repo)
            return result.get("imageIds", [])
        except Exception as e:
            logger.error(f"list_images failed: {e}")
            return []

    def get_latest_digest(self) -> Optional[str]:
        """Get the digest of the latest pushed image."""
        try:
            result = self._ecr().describe_images(
                repositoryName = self.cfg.ecr_repo,
                imageIds       = [{"imageTag": self.cfg.image_tag}],
            )
            images = result.get("imageDetails", [])
            return images[0].get("imageDigest") if images else None
        except Exception as e:
            logger.debug(f"get_latest_digest: {e}")
            return None

    def _get_git_sha(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True,
            )
            return result.stdout.strip()[:12] if result.returncode == 0 else ""
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# ECS — Container orchestration
# ─────────────────────────────────────────────────────────────────────────────

class ECSManager:
    """
    Manages ECS services for zero-downtime deployments.

    Zero-downtime deployment flow:
        1. Push new image to ECR
        2. Update task definition with new image URI
        3. Update ECS service to use new task definition
        4. ECS performs rolling replacement (new containers up before old ones stop)
        5. Health check passes → old containers terminated
    """

    def __init__(self, config: AWSConfig):
        self.cfg = config
        self._ecs = None

    def _client(self):
        if not self._ecs:
            import boto3
            self._ecs = boto3.client("ecs", region_name=self.cfg.region)
        return self._ecs

    def deploy(self, image_uri: str) -> bool:
        """
        Deploy a new container image to ECS.

        Performs a zero-downtime rolling deployment.
        Returns True if deployment initiated successfully.
        """
        print(f"\n  Deploying {image_uri} to ECS...")
        try:
            # Get current task definition
            svc = self._client().describe_services(
                cluster  = self.cfg.ecs_cluster,
                services = [self.cfg.ecs_service],
            )["services"][0]

            task_def_arn = svc["taskDefinition"]
            task_def     = self._client().describe_task_definition(
                taskDefinition = task_def_arn
            )["taskDefinition"]

            # Update container image in task definition
            containers = task_def["containerDefinitions"]
            for container in containers:
                if "hedgefund" in container["name"].lower():
                    container["image"] = image_uri

            # Register new task definition revision
            new_task_def = self._client().register_task_definition(
                family               = task_def["family"],
                containerDefinitions = containers,
                taskRoleArn          = task_def.get("taskRoleArn", ""),
                executionRoleArn     = task_def.get("executionRoleArn", ""),
                networkMode          = task_def.get("networkMode", "awsvpc"),
                volumes              = task_def.get("volumes", []),
                requiresCompatibilities = task_def.get("requiresCompatibilities", ["FARGATE"]),
                cpu                  = task_def.get("cpu", "512"),
                memory               = task_def.get("memory", "1024"),
            )["taskDefinition"]["taskDefinitionArn"]

            # Update service with new task definition
            self._client().update_service(
                cluster        = self.cfg.ecs_cluster,
                service        = self.cfg.ecs_service,
                taskDefinition = new_task_def,
                forceNewDeployment = True,
            )

            print(f"  ✓ ECS service update initiated")
            print(f"  ✓ New task definition: {new_task_def.split('/')[-1]}")
            return True

        except Exception as e:
            logger.error(f"ECS deploy failed: {e}")
            return False

    def wait_for_stable(self, timeout_seconds: int = 300) -> bool:
        """
        Wait for ECS service to reach stable state after deployment.
        Polls every 15 seconds up to timeout.
        """
        print(f"  Waiting for stable (max {timeout_seconds}s)...")
        start = time.time()

        while time.time() - start < timeout_seconds:
            try:
                svc = self._client().describe_services(
                    cluster  = self.cfg.ecs_cluster,
                    services = [self.cfg.ecs_service],
                )["services"][0]

                deployments  = svc.get("deployments", [])
                running_count= svc.get("runningCount", 0)
                desired_count= svc.get("desiredCount", 0)

                # Stable = exactly one deployment, running == desired
                if (len(deployments) == 1 and running_count == desired_count):
                    elapsed = time.time() - start
                    print(f"  ✓ Service stable in {elapsed:.0f}s "
                          f"({running_count}/{desired_count} tasks running)")
                    return True

                print(
                    f"  ... {running_count}/{desired_count} running | "
                    f"{len(deployments)} deployment(s) active | "
                    f"{int(time.time()-start)}s elapsed"
                )
                time.sleep(15)

            except Exception as e:
                logger.error(f"wait_for_stable poll failed: {e}")
                time.sleep(15)

        logger.error(f"Service did not stabilise within {timeout_seconds}s")
        return False

    def get_service_status(self) -> Dict:
        """Get current ECS service status."""
        try:
            svc = self._client().describe_services(
                cluster  = self.cfg.ecs_cluster,
                services = [self.cfg.ecs_service],
            )["services"][0]
            return {
                "service":       self.cfg.ecs_service,
                "status":        svc.get("status"),
                "desired_count": svc.get("desiredCount"),
                "running_count": svc.get("runningCount"),
                "pending_count": svc.get("pendingCount"),
                "deployments":   len(svc.get("deployments", [])),
                "last_event":    svc.get("events", [{}])[0].get("message", "") if svc.get("events") else "",
            }
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CloudWatch — Logging and monitoring
# ─────────────────────────────────────────────────────────────────────────────

class CloudWatchManager:
    """
    Manages CloudWatch logs and metrics.

    In production, every log line from every container goes to
    CloudWatch via the awslogs Docker log driver.
    This class provides:
        - Log streaming (tail -f style)
        - Custom metric publishing
        - Alarm creation and management
    """

    def __init__(self, config: AWSConfig):
        self.cfg  = config
        self._logs = None
        self._cw   = None

    def _logs_client(self):
        if not self._logs:
            import boto3
            self._logs = boto3.client("logs", region_name=self.cfg.region)
        return self._logs

    def _cw_client(self):
        if not self._cw:
            import boto3
            self._cw = boto3.client("cloudwatch", region_name=self.cfg.region)
        return self._cw

    def tail_logs(
        self,
        log_stream_prefix: str = "",
        since_minutes:     int = 10,
        follow:            bool = True,
    ) -> None:
        """
        Stream recent CloudWatch log events to stdout.
        Similar to: aws logs tail --follow
        """
        start_time = int((datetime.now() - timedelta(minutes=since_minutes)).timestamp() * 1000)

        print(f"\n  Streaming logs from {self.cfg.log_group} (last {since_minutes}min)...")
        print(f"  {'─' * 65}")

        seen_events = set()
        next_token  = None

        try:
            while True:
                kwargs = {
                    "logGroupName":        self.cfg.log_group,
                    "startTime":           start_time,
                    "interleaved":         True,
                }
                if log_stream_prefix:
                    kwargs["logStreamNamePrefix"] = log_stream_prefix
                if next_token:
                    kwargs["nextToken"] = next_token

                try:
                    resp    = self._logs_client().filter_log_events(**kwargs)
                    events  = resp.get("events", [])
                    next_token = resp.get("nextToken")

                    for event in events:
                        event_id = event.get("eventId", "")
                        if event_id not in seen_events:
                            seen_events.add(event_id)
                            ts  = datetime.fromtimestamp(event["timestamp"] / 1000)
                            msg = event.get("message", "").rstrip()
                            print(f"  {ts:%H:%M:%S} {msg}")

                    if not follow:
                        break

                    if not next_token:
                        time.sleep(2)
                        start_time = int(
                            (datetime.now() - timedelta(seconds=5)).timestamp() * 1000
                        )

                except KeyboardInterrupt:
                    break

        except Exception as e:
            logger.error(f"Log streaming failed: {e}")

    def put_metric(
        self,
        metric_name:  str,
        value:        float,
        unit:         str = "None",
        dimensions:   Optional[List[Dict]] = None,
        namespace:    str = "HedgeFund",
    ) -> None:
        """Publish a custom metric to CloudWatch."""
        try:
            self._cw_client().put_metric_data(
                Namespace  = namespace,
                MetricData = [{
                    "MetricName": metric_name,
                    "Value":      value,
                    "Unit":       unit,
                    "Timestamp":  datetime.now(),
                    "Dimensions": dimensions or [],
                }]
            )
        except Exception as e:
            logger.debug(f"put_metric failed: {e}")

    def create_alarm(
        self,
        alarm_name:       str,
        metric_name:      str,
        threshold:        float,
        comparison:       str = "GreaterThanThreshold",
        evaluation_periods: int = 2,
        period_seconds:   int = 300,
        alarm_actions:    Optional[List[str]] = None,    # SNS ARNs
        namespace:        str = "HedgeFund",
    ) -> bool:
        """Create a CloudWatch alarm."""
        try:
            self._cw_client().put_metric_alarm(
                AlarmName          = alarm_name,
                MetricName         = metric_name,
                Namespace          = namespace,
                Statistic          = "Average",
                Period             = period_seconds,
                EvaluationPeriods  = evaluation_periods,
                Threshold          = threshold,
                ComparisonOperator = comparison,
                AlarmActions       = alarm_actions or [],
                OKActions          = alarm_actions or [],
                TreatMissingData   = "notBreaching",
            )
            logger.info(f"Alarm created: {alarm_name}")
            return True
        except Exception as e:
            logger.error(f"create_alarm failed: {e}")
            return False

    def setup_fund_alarms(
        self,
        sns_topic_arn: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Create the standard set of production alarms.

        These fire to SNS (→ email / PagerDuty / Slack) when triggered.
        """
        actions = [sns_topic_arn] if sns_topic_arn else []
        results = {}

        alarms = [
            # Business alarms
            ("HF-DailyLossBreached",  "daily_loss_pct",   0.02,  "GreaterThanThreshold"),
            ("HF-VaRBreached",        "var_95_pct",        0.02,  "GreaterThanThreshold"),
            ("HF-DrawdownWarning",    "intraday_drawdown", 0.01,  "GreaterThanThreshold"),
            # Operational alarms
            ("HF-APIHighLatency",     "api_latency_p99",   2000,  "GreaterThanThreshold"),   # ms
            ("HF-APIErrorRate",       "api_error_rate",    0.05,  "GreaterThanThreshold"),   # 5%
            ("HF-LLMCostSpike",       "llm_cost_hourly",   5.0,   "GreaterThanThreshold"),   # $5/hr
        ]

        for name, metric, threshold, comparison in alarms:
            results[name] = self.create_alarm(
                alarm_name    = name,
                metric_name   = metric,
                threshold     = threshold,
                comparison    = comparison,
                alarm_actions = actions,
            )

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Secrets Manager
# ─────────────────────────────────────────────────────────────────────────────

class SecretsManager:
    """
    Manages API keys and secrets in AWS Secrets Manager.

    In production:
        - API keys are stored in Secrets Manager (not in .env files)
        - Containers fetch secrets at startup via IAM role
        - Key rotation doesn't require container restart

    Secret structure:
        {
          "anthropic_api_key": "sk-ant-...",
          "api_keys":          "prod-key-1,investor-key-2",
          "ib_account":        "DU12345678",
        }
    """

    def __init__(self, config: AWSConfig):
        self.cfg = config
        self._sm = None

    def _client(self):
        if not self._sm:
            import boto3
            self._sm = boto3.client("secretsmanager", region_name=self.cfg.region)
        return self._sm

    def get_secrets(self) -> Dict[str, str]:
        """Fetch all secrets from Secrets Manager."""
        try:
            result = self._client().get_secret_value(SecretId=self.cfg.secret_name)
            return json.loads(result["SecretString"])
        except Exception as e:
            logger.error(f"get_secrets failed: {e}")
            return {}

    def rotate_api_keys(self, new_keys: List[str]) -> bool:
        """Rotate API keys without downtime."""
        try:
            secrets = self.get_secrets()
            secrets["api_keys"] = ",".join(new_keys)
            self._client().put_secret_value(
                SecretId     = self.cfg.secret_name,
                SecretString = json.dumps(secrets),
            )
            logger.info(f"Rotated API keys: {len(new_keys)} keys active")
            return True
        except Exception as e:
            logger.error(f"rotate_api_keys failed: {e}")
            return False

    def inject_to_environment(self) -> None:
        """Load secrets into os.environ at container startup."""
        secrets = self.get_secrets()
        for key, value in secrets.items():
            env_key = key.upper()
            if env_key not in os.environ:
                os.environ[env_key] = str(value)
        logger.info(f"Injected {len(secrets)} secrets from Secrets Manager")


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_monthly_costs(
    instance_type: str = "t3.medium",
    region:        str = "us-east-1",
    data_transfer_gb: float = 10.0,
) -> Dict[str, float]:
    """
    Estimate monthly AWS costs for the hedge fund infrastructure.

    Prices as of 2024 (us-east-1). Verify at aws.amazon.com/pricing.
    """
    ec2_hourly = {
        "t3.micro":   0.0104,
        "t3.small":   0.0208,
        "t3.medium":  0.0416,
        "t3.large":   0.0832,
        "t3.xlarge":  0.1664,
        "c5.xlarge":  0.17,
        "c5.2xlarge": 0.34,
    }

    prices = {
        "ec2":             ec2_hourly.get(instance_type, 0.0416) * 730,  # 730hrs/month
        "ebs_50gb_gp3":    50 * 0.08,
        "cloudwatch_logs": 5.0,       # ~5GB logs/month
        "data_transfer":   data_transfer_gb * 0.09,
        "ecr_storage":     1.0,       # ~10GB images
        "route53":         0.50,
    }

    prices["total"] = sum(prices.values())
    return {k: round(v, 2) for k, v in prices.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Full deployment pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_deploy(
    config:        Optional[AWSConfig] = None,
    tag:           str = "latest",
    wait_for_stable: bool = True,
) -> bool:
    """
    Full deployment pipeline: build → push → deploy → verify.

    This is what CI/CD calls on every merge to main.
    """
    cfg = config or AWSConfig.from_env()

    print("\n" + "═" * 55)
    print("  AI Hedge Fund — Production Deployment")
    print(f"  Target: {cfg.ecr_uri}:{tag}")
    print("═" * 55)

    ecr = ECRManager(cfg)
    ecs = ECSManager(cfg)
    cw  = CloudWatchManager(cfg)

    # Step 1: Build and push
    print("\n  Step 1/3: Build and push Docker image...")
    image_uri = ecr.build_and_push(tag=tag)
    if not image_uri:
        print("  ✗ Build/push failed. Deployment aborted.")
        return False

    # Step 2: Deploy to ECS
    print("\n  Step 2/3: Deploy to ECS...")
    if not ecs.deploy(image_uri):
        print("  ✗ ECS deploy failed.")
        return False

    # Step 3: Wait for stable
    if wait_for_stable:
        print("\n  Step 3/3: Waiting for service to stabilise...")
        stable = ecs.wait_for_stable(timeout_seconds=300)
        if not stable:
            print("  ✗ Service did not stabilise. Check ECS console.")
            return False

    # Publish deployment metric
    cw.put_metric("deployment_success", 1.0, dimensions=[
        {"Name": "Service", "Value": cfg.ecs_service}
    ])

    print("\n  ✓ Deployment complete!")
    print(f"  Image: {image_uri}")
    print(f"  API:   https://api.yourfund.com/portfolio")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="AWS Deployment Utilities")
    sub    = parser.add_subparsers(dest="command")

    sub.add_parser("deploy",  help="Full build + push + deploy pipeline")
    sub.add_parser("status",  help="Check ECS service status")
    sub.add_parser("logs",    help="Tail CloudWatch logs")
    sub.add_parser("costs",   help="Estimate monthly AWS costs")
    sub.add_parser("alarms",  help="Create production CloudWatch alarms")

    args = parser.parse_args()
    cfg  = AWSConfig.from_env()

    if args.command == "deploy":
        full_deploy(cfg)

    elif args.command == "status":
        ecs = ECSManager(cfg)
        status = ecs.get_service_status()
        print(json.dumps(status, indent=2))

    elif args.command == "logs":
        cw = CloudWatchManager(cfg)
        cw.tail_logs(follow=True)

    elif args.command == "costs":
        costs = estimate_monthly_costs()
        print("\n  Monthly AWS Cost Estimate (us-east-1, t3.medium):")
        print(f"  {'─'*40}")
        for item, cost in costs.items():
            if item != "total":
                print(f"  {item:<25}: ${cost:>7.2f}")
        print(f"  {'─'*40}")
        print(f"  {'TOTAL':<25}: ${costs['total']:>7.2f}/month")

    elif args.command == "alarms":
        cw      = CloudWatchManager(cfg)
        results = cw.setup_fund_alarms()
        for alarm, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {alarm}")

    else:
        parser.print_help()
