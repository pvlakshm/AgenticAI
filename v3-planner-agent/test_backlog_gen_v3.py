import pytest
from unittest.mock import Mock

import backlog_gen_v3 as v3


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #

SAMPLE_REQUIREMENT = "Build a login page with email and password fields"

SAMPLE_EPIC = """Epic: User Authentication
Description: Enable users to securely log in to the application.
Acceptance Criteria:
- The system provides a login page with email and password fields
- The system validates credentials and grants access on success
- The system displays an error message on failed login"""

SAMPLE_FEATURES = """Feature: Login Form
Description: A form with email and password fields and a submit button.
Acceptance Criteria:
- The form displays an email field
- The form displays a password field
- The form displays a submit button

Feature: Credential Validation
Description: Validates user credentials against the user store.
Acceptance Criteria:
- Valid credentials grant access to the application
- Invalid credentials display a clear error message
- The system locks the account after 3 failed attempts

Feature: Session Management
Description: Manages the user session after successful login.
Acceptance Criteria:
- A session token is created on successful login
- The session expires after a configurable timeout
- The user can explicitly log out to end the session"""


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_ollama():
    """Base fixture — injects a mock ollama into the module."""
    mock = Mock()
    mock.chat.return_value = {"message": {"content": "some response"}}
    v3.ollama = mock
    yield mock


@pytest.fixture
def initial_state():
    """Returns a freshly initialized shared state dict."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "plan": [],
        "epic": None,
        "features": None,
    }


@pytest.fixture
def planned_state():
    """Returns a state dict with the plan already set."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "plan": ["epic", "features"],
        "epic": None,
        "features": None,
    }


@pytest.fixture
def full_pipeline_mock(mock_ollama):
    """Sets up sequential LLM responses for planner + epic + features."""
    mock_ollama.chat.side_effect = [
        {"message": {"content": "Plan: epic, features"}},
        {"message": {"content": SAMPLE_EPIC}},
        {"message": {"content": SAMPLE_FEATURES}},
    ]
    return mock_ollama


# --------------------------------------------------------------------------- #
# Tests for run_planner
# --------------------------------------------------------------------------- #

class TestRunPlanner:

    def test_planner_uses_planner_template_task(self, mock_ollama, initial_state):
        """run_planner should call the LLM with the planner template task."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}

        # Act
        v3.run_planner(initial_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v3.TEMPLATES["planner"]["task"] in prompt

    def test_planner_writes_plan_to_state(self, mock_ollama, initial_state):
        """run_planner should write the parsed plan to state['plan']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}

        # Act
        result_state = v3.run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_handles_epic_only_plan(self, mock_ollama, initial_state):
        """run_planner should correctly parse a plan with only epic."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic"}}

        # Act
        result_state = v3.run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic"]

    def test_planner_filters_invalid_steps(self, mock_ollama, initial_state):
        """run_planner should ignore steps that are not in TASK_MAP."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, stories, features"}}

        # Act
        result_state = v3.run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_raises_on_fully_invalid_output(self, mock_ollama, initial_state):
        """run_planner should raise ValueError when no valid steps are found."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "I cannot determine the plan"}}

        # Act / Assert
        with pytest.raises(ValueError):
            v3.run_planner(initial_state)

    def test_planner_returns_state(self, mock_ollama, initial_state):
        """run_planner should return the updated state dict."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}

        # Act
        result_state = v3.run_planner(initial_state)

        # Assert
        assert isinstance(result_state, dict)
        assert "plan" in result_state


# --------------------------------------------------------------------------- #
# Tests for TASK_MAP routing
# --------------------------------------------------------------------------- #

class TestTaskMap:

    def test_task_map_contains_epic_key(self):
        """TASK_MAP should contain an 'epic' key."""
        assert "epic" in v3.TASK_MAP

    def test_task_map_contains_features_key(self):
        """TASK_MAP should contain a 'features' key."""
        assert "features" in v3.TASK_MAP

    def test_task_map_epic_routes_to_generate_epic(self):
        """TASK_MAP['epic'] should point to the generate_epic function."""
        assert v3.TASK_MAP["epic"] == v3.generate_epic

    def test_task_map_features_routes_to_generate_features(self):
        """TASK_MAP['features'] should point to the generate_features function."""
        assert v3.TASK_MAP["features"] == v3.generate_features


# --------------------------------------------------------------------------- #
# Tests for the full pipeline with planner
# --------------------------------------------------------------------------- #

class TestPlannerPipeline:

    def test_pipeline_runs_steps_in_plan_order(self, full_pipeline_mock, initial_state):
        """Pipeline should execute steps in the order returned by the planner."""
        # Act
        state = v3.run_planner(initial_state)
        for step in state["plan"]:
            state = v3.TASK_MAP[step](state)

        # Assert
        first_prompt = full_pipeline_mock.chat.call_args_list[1].kwargs["messages"][0]["content"]
        second_prompt = full_pipeline_mock.chat.call_args_list[2].kwargs["messages"][0]["content"]
        assert v3.TEMPLATES["epic"]["task"] in first_prompt
        assert v3.TEMPLATES["features"]["task"] in second_prompt

    def test_pipeline_makes_three_llm_calls(self, full_pipeline_mock, initial_state):
        """Full pipeline (planner + epic + features) should make exactly 3 LLM calls."""
        # Act
        state = v3.run_planner(initial_state)
        for step in state["plan"]:
            state = v3.TASK_MAP[step](state)

        # Assert
        assert full_pipeline_mock.chat.call_count == 3

    def test_pipeline_populates_all_state_keys(self, full_pipeline_mock, initial_state):
        """After the full pipeline, state should have plan, epic, and features populated."""
        # Act
        state = v3.run_planner(initial_state)
        for step in state["plan"]:
            state = v3.TASK_MAP[step](state)

        # Assert
        assert state["plan"] == ["epic", "features"]
        assert state["epic"] == SAMPLE_EPIC
        assert state["features"] == SAMPLE_FEATURES
