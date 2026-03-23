"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
base_agent.py — Abstract Base for All Agents

Every agent in the hedge fund system inherits from this class.
It provides:
  - LLM client (Claude or GPT-4) with cost tracking
  - Message bus integration (publish/consume/reply)
  - Tool registry (functions the LLM can call)
  - Processing loop (run once or continuously)
  - Performance metrics (decisions made, accuracy over time)
  - Heartbeat (liveness signalling to the coordinator)

Agent lifecycle:
    __init__() → register tools → start processing loop
    loop: consume messages → process each → reply/alert → sleep

Adding a new agent:
    1. Subclass BaseAgent
    2. Implement _get_system_prompt() — the agent's role and rules
    3. Implement _get_tools() — functions the agent can call
    4. Implement handle_message() — what to do with each message type
    5. Done. The base class handles everything else.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("hedge_fund.base_agent")


# ─────────────────────────────────────────────────────────────────────────────
# Tool definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    """
    A function that an LLM agent can call.

    Tools connect the LLM to real data and calculations.
    The description is what the LLM reads to decide when to use it.
    Write descriptions precisely — vague descriptions = wrong tool calls.
    """
    name:          str
    func:          Callable
    description:   str
    param_schema:  Optional[Dict] = None   # JSON schema for parameters
    is_async:      bool = False

    def call(self, **kwargs) -> Any:
        """Execute the tool function."""
        try:
            result = self.func(**kwargs)
            return result
        except Exception as e:
            return f"Tool '{self.name}' error: {str(e)}"

    def to_anthropic_format(self) -> Dict:
        """Convert to Anthropic tool-use format."""
        schema = self.param_schema or {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input to the tool"}
            },
            "required": []
        }
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for any agent in the system."""
    name:             str
    model:            str = "claude-sonnet-4-6"
    temperature:      float = 0.1
    max_tokens:       int = 4096
    poll_interval_s:  float = 2.0        # How often to check for messages
    max_iterations:   int = 10           # Max tool-call loops per request
    heartbeat_s:      float = 30.0       # How often to send heartbeat
    use_cache:        bool = False        # Never cache risk/live decisions
    verbose:          bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Agent metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    """Runtime performance metrics for an agent."""
    agent_name:        str
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed:   int = 0
    llm_calls:         int = 0
    total_llm_cost:    float = 0.0
    avg_latency_ms:    float = 0.0
    started_at:        datetime = field(default_factory=datetime.now)
    last_active:       datetime = field(default_factory=datetime.now)

    def record_llm_call(self, cost: float, latency_ms: float) -> None:
        n = self.llm_calls
        self.llm_calls += 1
        self.total_llm_cost += cost
        # Running average of latency
        self.avg_latency_ms = (self.avg_latency_ms * n + latency_ms) / (n + 1)
        self.last_active = datetime.now()

    def uptime_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            f"Agent: {self.agent_name} | "
            f"Processed: {self.messages_processed} | "
            f"Failed: {self.messages_failed} | "
            f"LLM calls: {self.llm_calls} | "
            f"LLM cost: ${self.total_llm_cost:.4f} | "
            f"Avg latency: {self.avg_latency_ms:.0f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all hedge fund agents.

    Subclass this and implement:
        _get_system_prompt() → str
        _get_tools()         → List[Tool]
        handle_message()     → Optional[Dict]

    Everything else (LLM calling, message routing, metrics,
    heartbeat, tool execution loop) is handled here.
    """

    def __init__(self, config: AgentConfig):
        self.config  = config
        self.name    = config.name
        self.metrics = AgentMetrics(agent_name=config.name)

        # Set up LLM client
        from src.agents.llm_client import LLMClient
        self.llm = LLMClient(
            model      = config.model,
            agent_name = config.name,
            use_cache  = config.use_cache,
        )

        # Set up message bus connection
        from src.comms.message_bus import get_bus
        self.bus = get_bus()

        # Register tools
        self._tools: Dict[str, Tool] = {}
        for tool in self._get_tools():
            self._tools[tool.name] = tool

        # Threading
        self._running       = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event    = threading.Event()

        logger.info(
            f"Agent '{self.name}' initialised | "
            f"model={config.model} | "
            f"tools={list(self._tools.keys())}"
        )

    # ── Abstract methods (must implement in subclass) ─────────────────────────

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Define this agent's role, capabilities, and decision-making rules.

        This is the most important method — it determines how the agent thinks.
        Be specific about:
          - What this agent's job is
          - What tools it has and when to use them
          - What its output format must be
          - What its hard limits are
        """
        ...

    @abstractmethod
    def _get_tools(self) -> List[Tool]:
        """
        Return the list of tools this agent can use.

        Tools should cover everything the agent needs to do its job.
        Rule of thumb: if you'd ask a human analyst to look something up,
        it should be a tool.
        """
        ...

    @abstractmethod
    def handle_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Process an incoming message and return a response payload.

        Args:
            message: Message object from the bus

        Returns:
            Dict to send back as response, or None for no response.

        The base class handles routing the response back through the bus.
        """
        ...

    # ── LLM interaction ───────────────────────────────────────────────────────

    def think(
        self,
        user_message:  str,
        conversation:  Optional[List[Dict]] = None,
        use_tools:     bool = True,
        max_iterations: Optional[int] = None,
        purpose:       str = "",
    ) -> Tuple[str, List[Dict]]:
        """
        Run the agent's reasoning loop.

        Sends the user_message to the LLM.
        If the LLM calls a tool, executes it and sends the result back.
        Repeats until LLM produces a final text response or max_iterations.

        Args:
            user_message  : The task or question
            conversation  : Prior conversation history
            use_tools     : Whether to allow tool calls
            max_iterations: Override config max_iterations
            purpose       : Label for cost tracking

        Returns:
            (final_text_response, tool_call_log)
        """
        system        = self._get_system_prompt()
        history       = list(conversation or [])
        tool_call_log = []
        max_iter      = max_iterations or self.config.max_iterations
        start_time    = datetime.now()

        # Add user message to history
        history.append({"role": "user", "content": user_message})

        for iteration in range(max_iter):
            # Build tools for this call
            tools_list = None
            if use_tools and self._tools:
                tools_list = [t.to_anthropic_format() for t in self._tools.values()]

            # Call LLM
            response = self._call_llm_with_tools(
                system     = system,
                messages   = history,
                tools_list = tools_list,
                purpose    = purpose,
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.record_llm_call(response.get("cost", 0), latency_ms)

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # LLM finished — no more tool calls
                logger.debug(
                    f"{self.name}: think() complete after {iteration+1} iterations"
                )
                return content, tool_call_log

            # Execute each tool call
            tool_results = []
            for tc in tool_calls:
                tool_name  = tc.get("name", "")
                tool_input = tc.get("input", {})
                tool_id    = tc.get("id", f"tool_{uuid.uuid4().hex[:8]}")

                result = self._execute_tool(tool_name, tool_input)
                tool_call_log.append({
                    "tool":   tool_name,
                    "input":  tool_input,
                    "result": str(result)[:500],  # Truncate for log
                })

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     str(result),
                })

            # Add assistant tool call turn + tool results to history
            history.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc.get("input", {})}
                    for tc in tool_calls
                ]
            })
            history.append({
                "role": "user",
                "content": tool_results,
            })

        logger.warning(
            f"{self.name}: hit max_iterations ({max_iter}) — "
            "returning last response"
        )
        return content, tool_call_log

    def _call_llm_with_tools(
        self,
        system:     str,
        messages:   List[Dict],
        tools_list: Optional[List[Dict]],
        purpose:    str = "",
    ) -> Dict:
        """
        Direct Anthropic API call with tool-use support.

        Returns dict with keys: content, tool_calls, cost
        """
        import os
        try:
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", "")
            )

            kwargs = {
                "model":      self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system":     system,
                "messages":   messages,
            }
            if tools_list:
                kwargs["tools"] = tools_list

            response = client.messages.create(**kwargs)

            # Parse response
            text_content = ""
            tool_calls   = []

            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id":    block.id,
                        "name":  block.name,
                        "input": block.input,
                    })

            # Cost calculation
            in_tok  = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            cost    = (in_tok * 0.003 + out_tok * 0.015) / 1000

            return {
                "content":    text_content,
                "tool_calls": tool_calls,
                "cost":       cost,
                "in_tokens":  in_tok,
                "out_tokens": out_tok,
            }

        except Exception as e:
            logger.error(f"LLM call failed in {self.name}: {e}")
            return {"content": f"Error: {e}", "tool_calls": [], "cost": 0}

    def _execute_tool(self, name: str, input_data: Any) -> Any:
        """Execute a named tool with given input."""
        if name not in self._tools:
            return f"Unknown tool: {name}. Available: {list(self._tools.keys())}"

        tool = self._tools[name]
        try:
            # Handle both dict input and string input
            if isinstance(input_data, dict):
                result = tool.func(**input_data)
            elif isinstance(input_data, str):
                result = tool.func(input_data)
            else:
                result = tool.func(input_data)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            return f"Tool error: {e}"

    # ── Message processing loop ───────────────────────────────────────────────

    def process_once(self) -> int:
        """
        Process one batch of pending messages.

        Returns the number of messages processed.
        Call this manually for testing or step-by-step operation.
        """
        from src.comms.message_bus import MessageType, Priority

        messages = self.bus.consume(self.name, limit=10)
        processed = 0

        for msg in messages:
            self.metrics.messages_received += 1
            start = datetime.now()

            try:
                # Skip expired
                if msg.is_expired:
                    self.bus.nack(msg.message_id)
                    continue

                # Handle heartbeat check (respond to coordinator pings)
                if msg.message_type == MessageType.HEARTBEAT:
                    self.bus.reply(
                        original_message = msg,
                        sender           = self.name,
                        payload          = {
                            "status": "alive",
                            "metrics": self.metrics.__dict__,
                        }
                    )
                    self.bus.ack(msg.message_id)
                    continue

                # Dispatch to subclass handler
                response_payload = self.handle_message(msg)

                # Send response if handler returned one
                if response_payload is not None:
                    self.bus.reply(
                        original_message = msg,
                        sender           = self.name,
                        payload          = response_payload,
                    )

                self.bus.ack(msg.message_id)
                self.metrics.messages_processed += 1
                processed += 1

            except Exception as e:
                logger.error(f"{self.name} failed processing {msg.message_id}: {e}")
                self.metrics.messages_failed += 1
                self.bus.nack(msg.message_id)

        return processed

    def start(self) -> None:
        """Start the agent's background processing loop."""
        if self._running:
            logger.warning(f"{self.name} is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target = self._run_loop,
            name   = f"agent-{self.name}",
            daemon = True,
        )
        self._thread.start()
        logger.info(f"Agent '{self.name}' started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent's processing loop gracefully."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(f"Agent '{self.name}' stopped")

    def _run_loop(self) -> None:
        """Background processing loop."""
        last_heartbeat = time.time()

        while not self._stop_event.is_set():
            try:
                self.process_once()

                # Send heartbeat
                if time.time() - last_heartbeat > self.config.heartbeat_s:
                    self._send_heartbeat()
                    last_heartbeat = time.time()

            except Exception as e:
                logger.error(f"{self.name} loop error: {e}")

            self._stop_event.wait(timeout=self.config.poll_interval_s)

    def _send_heartbeat(self) -> None:
        from src.comms.message_bus import Message, MessageType, Priority
        hb = Message.create(
            sender       = self.name,
            recipient    = "AgentCoordinator",
            subject      = "heartbeat",
            payload      = {
                "agent":      self.name,
                "status":     "alive",
                "timestamp":  datetime.now().isoformat(),
                "metrics":    {
                    "messages_processed": self.metrics.messages_processed,
                    "llm_calls":          self.metrics.llm_calls,
                    "total_cost":         self.metrics.total_llm_cost,
                }
            },
            message_type = MessageType.HEARTBEAT,
            priority     = Priority.LOW,
            ttl_seconds  = 60,
        )
        self.bus.publish(hb)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def ask_agent(
        self,
        recipient: str,
        subject:   str,
        payload:   Dict[str, Any],
        timeout:   float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask another agent a question and wait for the answer.

        Returns the response payload dict or None if no response.
        """
        response = self.bus.request(
            sender    = self.name,
            recipient = recipient,
            subject   = subject,
            payload   = payload,
            timeout   = timeout,
        )
        return response.payload if response else None

    def alert(
        self,
        subject:  str,
        payload:  Dict[str, Any],
    ) -> str:
        """Broadcast an alert to all agents."""
        return self.bus.broadcast_alert(
            sender  = self.name,
            subject = subject,
            payload = payload,
        )

    def get_metrics(self) -> str:
        return self.metrics.summary()

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.config.model}')"
