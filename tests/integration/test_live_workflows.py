import os
import shutil

import pytest

from maike.smoke.workflows import (
    WORKFLOW_CASES,
    default_live_provider,
    provider_has_key,
    run_workflow_case,
    select_workflow_names,
)


@pytest.mark.parametrize("workflow_name", sorted(WORKFLOW_CASES))
def test_live_workflows_against_real_provider(workflow_name):
    if os.getenv("MAIKE_RUN_LIVE_SMOKE") != "1":
        pytest.skip("Set MAIKE_RUN_LIVE_SMOKE=1 to enable live workflow smoke tests.")

    selected = set(select_workflow_names((os.getenv("MAIKE_LIVE_WORKFLOWS") or "all").split(",")))
    if workflow_name not in selected:
        pytest.skip(f"{workflow_name} not selected in MAIKE_LIVE_WORKFLOWS.")

    provider = os.getenv("MAIKE_LIVE_PROVIDER") or default_live_provider()
    if not provider_has_key(provider):
        pytest.skip(f"No API key detected for provider '{provider}'.")

    outcome = run_workflow_case(
        workflow_name,
        provider=provider,
        model=os.getenv("MAIKE_LIVE_MODEL") or None,
    )
    assert outcome.pipeline == WORKFLOW_CASES[workflow_name].expected_pipeline
    if os.getenv("MAIKE_KEEP_LIVE_WORKSPACES") != "1":
        shutil.rmtree(outcome.workspace, ignore_errors=True)
