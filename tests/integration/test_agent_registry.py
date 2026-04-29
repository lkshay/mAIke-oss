import asyncio

from maike.atoms.agent import AgentResult
from maike.atoms.blueprint import AgentBlueprint, SpawnReason
from maike.constants import MAX_CONCURRENT_AGENTS
from maike.memory.session import SessionStore
from maike.orchestrator.registry import AgentRegistry, SpawnLimitError
from maike.orchestrator.session import OrchestratorSession


class FakeAgentCore:
    async def run(self, ctx, messages):
        return AgentResult(
            agent_id=ctx.agent_id,
            role=ctx.role,
            stage_name=ctx.stage_name,
            output="ok",
            messages=messages,
        )


def test_agent_registry_enforces_concurrent_limit(tmp_path):
    async def scenario():
        store = SessionStore(tmp_path)
        await store.initialize()
        session_id = await store.create_session("task", tmp_path)
        session = OrchestratorSession(store, session_id, "task", tmp_path)
        registry = AgentRegistry(session, FakeAgentCore())
        blueprints = [
            AgentBlueprint(
                role="coder",
                task=f"task {index}",
                stage_name="coding",
                tool_profile="coding",
                spawn_reason=SpawnReason.PARALLEL_PARTITION,
            )
            for index in range(MAX_CONCURRENT_AGENTS + 1)
        ]
        live_agents = []
        for blueprint in blueprints[:MAX_CONCURRENT_AGENTS]:
            live_agents.append(await registry.spawn(blueprint))
        try:
            await registry.spawn(blueprints[-1])
        except SpawnLimitError:
            return
        raise AssertionError("Spawn limit was not enforced")

    asyncio.run(scenario())


