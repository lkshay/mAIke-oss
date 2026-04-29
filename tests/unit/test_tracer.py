from maike.atoms.context import AgentContext
from maike.atoms.llm import LLMCallRecord
from maike.atoms.tool import ToolResult
from maike.observability.tracer import Tracer
from maike.tools.context import CURRENT_AGENT_CONTEXT


def make_record() -> LLMCallRecord:
    return LLMCallRecord(
        provider="anthropic",
        model="claude-opus-4-20250514",
        input_tokens=120,
        output_tokens=30,
        cost_usd=0.42,
        latency_ms=18,
        stop_reason="end_turn",
    )


def test_tracer_enriches_events_from_current_agent_context():
    tracer = Tracer()
    ctx = AgentContext(
        role="coder",
        task="write tests",
        stage_name="coding",
        tool_profile="coding",
        metadata={"session_id": "session-123"},
    )
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        tracer.log_llm_call(make_record())
        tracer.log_tool_result(
            ToolResult(
                tool_name="read_file",
                success=True,
                raw_output="hello",
                output="hello",
                execution_ms=7,
            )
        )
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    llm_event = tracer.events[0]
    tool_event = tracer.events[1]

    assert llm_event["kind"] == "llm_call"
    assert llm_event["input_tokens"] == 120
    assert llm_event["output_tokens"] == 30
    assert llm_event["total_tokens"] == 150
    assert llm_event["latency_ms"] == 18
    assert llm_event["session_id"] == "session-123"
    assert llm_event["agent_id"] == ctx.agent_id
    assert llm_event["stage_name"] == "coding"
    assert llm_event["tool_profile"] == "coding"

    assert tool_event["kind"] == "tool_result"
    assert tool_event["tool_name"] == "read_file"
    assert tool_event["session_id"] == "session-123"
    assert tool_event["agent_id"] == ctx.agent_id
    assert tool_event["stage_name"] == "coding"
    assert tool_event["tool_profile"] == "coding"


def test_tracer_leaves_agent_metadata_empty_without_context():
    tracer = Tracer()

    tracer.log_llm_call(make_record())

    event = tracer.events[0]
    assert event["session_id"] is None
    assert event["agent_id"] is None
    assert event["stage_name"] is None
    assert event["tool_profile"] is None


def test_tracer_records_lifecycle_events_with_payloads():
    tracer = Tracer()
    ctx = AgentContext(
        role="tester",
        task="run checks",
        stage_name="testing",
        tool_profile="testing",
        metadata={"session_id": "session-456"},
    )
    token = CURRENT_AGENT_CONTEXT.set(ctx)
    try:
        tracer.log_stage_start("testing", payload={"attempt": 1})
        tracer.log_tool_start("execute_bash", payload={"input": {"cmd": "pytest -q"}})
        tracer.log_event("spawn_request", request={"role": "debugger"})
    finally:
        CURRENT_AGENT_CONTEXT.reset(token)

    tracer.log_agent_spawn(
        session_id="session-456",
        agent_id="agent-child",
        agent_role="debugger",
        stage_name="testing",
        tool_profile="debugging",
        payload={"spawn_reason": "specialist_needed"},
    )
    tracer.log_stage_complete("testing", success=True, payload={"attempt": 1})

    stage_start, tool_start, spawn_request, agent_spawn, stage_complete = tracer.events

    assert stage_start["kind"] == "stage_start"
    assert stage_start["stage_name"] == "testing"
    assert stage_start["payload"]["attempt"] == 1
    assert stage_start["agent_role"] == "tester"

    assert tool_start["kind"] == "tool_start"
    assert tool_start["tool_name"] == "execute_bash"
    assert tool_start["payload"]["input"]["cmd"] == "pytest -q"
    assert tool_start["session_id"] == "session-456"

    assert spawn_request["kind"] == "spawn_request"
    assert spawn_request["payload"]["request"]["role"] == "debugger"
    assert spawn_request["agent_id"] == ctx.agent_id

    assert agent_spawn["kind"] == "agent_spawn"
    assert agent_spawn["agent_id"] == "agent-child"
    assert agent_spawn["agent_role"] == "debugger"
    assert agent_spawn["payload"]["spawn_reason"] == "specialist_needed"

    assert stage_complete["kind"] == "stage_complete"
    assert stage_complete["success"] is True
