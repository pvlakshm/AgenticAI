import pytest
from unittest.mock import Mock

import backlog_gen_v5 as v5


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
    v5.ollama = mock
    yield mock


@pytest.fixture
def initial_state():
    """Returns a freshly initialized shared state dict."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "plan": ["epic", "features"],
        "epic": None,
        "features": None,
    }


@pytest.fixture
def state_with_epic():
    """Returns a state dict with the epic already populated."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "plan": ["epic", "features"],
        "epic": SAMPLE_EPIC,
        "features": None,
    }


@pytest.fixture
def epic_agent_mock(mock_ollama):
    """Sets up LLM responses for EpicAgent: generation + approval."""
    mock_ollama.chat.side_effect = [
        {"message": {"content": SAMPLE_EPIC}},   # generation
        {"message": {"content": "APPROVED"}},     # critic
    ]
    return mock_ollama


@pytest.fixture
def features_agent_mock(mock_ollama):
    """Sets up LLM responses for FeaturesAgent: generation + approval."""
    mock_ollama.chat.side_effect = [
        {"message": {"content": SAMPLE_FEATURES}},  # generation
        {"message": {"content": "APPROVED"}},        # critic
    ]
    return mock_ollama


# --------------------------------------------------------------------------- #
# Tests for EpicAgent
# --------------------------------------------------------------------------- #

class TestEpicAgent:

    def test_epic_agent_run_returns_state(self, epic_agent_mock, initial_state):
        """EpicAgent.run should return the updated state dict."""
        # Act
        result_state = v5.EpicAgent().run(initial_state)

        # Assert
        assert isinstance(result_state, dict)

    def test_epic_agent_writes_epic_to_state(self, epic_agent_mock, initial_state):
        """EpicAgent.run should populate state['epic']."""
        # Act
        result_state = v5.EpicAgent().run(initial_state)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC

    def test_epic_agent_does_not_modify_requirement(self, epic_agent_mock, initial_state):
        """EpicAgent.run should not change state['requirement']."""
        # Act
        result_state = v5.EpicAgent().run(initial_state)

        # Assert
        assert result_state["requirement"] == SAMPLE_REQUIREMENT

    def test_epic_agent_uses_epic_template_task(self, epic_agent_mock, initial_state):
        """EpicAgent should call the LLM with the epic template task."""
        # Act
        v5.EpicAgent().run(initial_state)

        # Assert
        prompt = epic_agent_mock.chat.call_args_list[0].kwargs["messages"][0]["content"]
        assert v5.TEMPLATES["epic"]["task"] in prompt


# --------------------------------------------------------------------------- #
# Tests for FeaturesAgent
# --------------------------------------------------------------------------- #

class TestFeaturesAgent:

    def test_features_agent_run_returns_state(self, features_agent_mock, state_with_epic):
        """FeaturesAgent.run should return the updated state dict."""
        # Act
        result_state = v5.FeaturesAgent().run(state_with_epic)

        # Assert
        assert isinstance(result_state, dict)

    def test_features_agent_writes_features_to_state(self, features_agent_mock, state_with_epic):
        """FeaturesAgent.run should populate state['features']."""
        # Act
        result_state = v5.FeaturesAgent().run(state_with_epic)

        # Assert
        assert result_state["features"] == SAMPLE_FEATURES

    def test_features_agent_reads_epic_from_state(self, features_agent_mock, state_with_epic):
        """FeaturesAgent should pass state['epic'] to the LLM."""
        # Act
        v5.FeaturesAgent().run(state_with_epic)

        # Assert
        prompt = features_agent_mock.chat.call_args_list[0].kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in prompt

    def test_features_agent_does_not_modify_epic(self, features_agent_mock, state_with_epic):
        """FeaturesAgent.run should not change state['epic']."""
        # Act
        result_state = v5.FeaturesAgent().run(state_with_epic)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC


# --------------------------------------------------------------------------- #
# Tests for AGENT_MAP
# --------------------------------------------------------------------------- #

class TestAgentMap:

    def test_agent_map_contains_epic_key(self):
        """AGENT_MAP should contain an 'epic' key."""
        assert "epic" in v5.AGENT_MAP

    def test_agent_map_contains_features_key(self):
        """AGENT_MAP should contain a 'features' key."""
        assert "features" in v5.AGENT_MAP

    def test_agent_map_epic_routes_to_epic_agent(self):
        """AGENT_MAP['epic'] should point to EpicAgent."""
        assert v5.AGENT_MAP["epic"] == v5.EpicAgent

    def test_agent_map_features_routes_to_features_agent(self):
        """AGENT_MAP['features'] should point to FeaturesAgent."""
        assert v5.AGENT_MAP["features"] == v5.FeaturesAgent


# --------------------------------------------------------------------------- #
# Tests for Coordinator._run_planner
# --------------------------------------------------------------------------- #

class TestCoordinatorPlanner:

    def test_planner_writes_plan_to_state(self, mock_ollama, initial_state):
        """_run_planner should write the parsed plan to state['plan']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}
        initial_state["plan"] = []

        # Act
        result_state = v5.Coordinator()._run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_normalizes_to_lowercase(self, mock_ollama, initial_state):
        """_run_planner should normalize step names to lowercase."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: Epic, Features"}}
        initial_state["plan"] = []

        # Act
        result_state = v5.Coordinator()._run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_raises_on_invalid_output(self, mock_ollama, initial_state):
        """_run_planner should raise ValueError when no valid steps are found."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "I cannot determine the plan"}}
        initial_state["plan"] = []

        # Act / Assert
        with pytest.raises(ValueError):
            v5.Coordinator()._run_planner(initial_state)
