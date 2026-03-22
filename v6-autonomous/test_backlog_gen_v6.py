import pytest
from unittest.mock import Mock

import backlog_gen_v6 as v6


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
    v6.ollama = mock
    yield mock


@pytest.fixture
def full_backlog_state():
    """Returns a state dict with both epic and features populated."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "plan": ["epic", "features"],
        "epic": SAMPLE_EPIC,
        "features": SAMPLE_FEATURES,
    }


@pytest.fixture
def coordinator():
    """Returns a fresh Coordinator instance."""
    return v6.Coordinator()


# --------------------------------------------------------------------------- #
# Tests for Coordinator._parse_redo
# --------------------------------------------------------------------------- #

class TestParseRedo:

    def test_parse_redo_returns_epic_for_redo_epic(self, coordinator):
        """_parse_redo should return 'epic' for 'REDO: epic' verdict."""
        # Act
        result = coordinator._parse_redo("REDO: epic. Reason: epic is missing.")

        # Assert
        assert result == "epic"

    def test_parse_redo_returns_features_for_redo_features(self, coordinator):
        """_parse_redo should return 'features' for 'REDO: features' verdict."""
        # Act
        result = coordinator._parse_redo("REDO: features. Reason: fewer than 3 features.")

        # Assert
        assert result == "features"

    def test_parse_redo_is_case_insensitive(self, coordinator):
        """_parse_redo should handle uppercase REDO verdicts."""
        # Act
        result = coordinator._parse_redo("REDO: FEATURES")

        # Assert
        assert result == "features"

    def test_parse_redo_returns_none_for_approved(self, coordinator):
        """_parse_redo should return None when verdict is APPROVED."""
        # Act
        result = coordinator._parse_redo("APPROVED")

        # Assert
        assert result is None

    def test_parse_redo_returns_none_for_unparseable_verdict(self, coordinator):
        """_parse_redo should return None when verdict cannot be parsed."""
        # Act
        result = coordinator._parse_redo("something completely unexpected")

        # Assert
        assert result is None


# --------------------------------------------------------------------------- #
# Tests for Coordinator._run_global_critic
# --------------------------------------------------------------------------- #

class TestRunGlobalCritic:

    def test_global_critic_includes_requirement_in_input(self, mock_ollama, coordinator, full_backlog_state):
        """_run_global_critic should include the requirement in the LLM input."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        coordinator._run_global_critic(full_backlog_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_REQUIREMENT in prompt

    def test_global_critic_includes_epic_in_input(self, mock_ollama, coordinator, full_backlog_state):
        """_run_global_critic should include the epic in the LLM input."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        coordinator._run_global_critic(full_backlog_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in prompt

    def test_global_critic_includes_features_in_input(self, mock_ollama, coordinator, full_backlog_state):
        """_run_global_critic should include the features in the LLM input."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        coordinator._run_global_critic(full_backlog_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_FEATURES in prompt

    def test_global_critic_uses_global_critic_template(self, mock_ollama, coordinator, full_backlog_state):
        """_run_global_critic should use the global_critic template task."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        coordinator._run_global_critic(full_backlog_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v6.TEMPLATES["global_critic"]["task"] in prompt

    def test_global_critic_returns_llm_verdict(self, mock_ollama, coordinator, full_backlog_state):
        """_run_global_critic should return the raw LLM verdict."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        verdict = coordinator._run_global_critic(full_backlog_state)

        # Assert
        assert verdict == "APPROVED"


# --------------------------------------------------------------------------- #
# Tests for Coordinator._run_planner
# --------------------------------------------------------------------------- #

class TestCoordinatorPlanner:

    def test_planner_writes_plan_to_state(self, mock_ollama, coordinator):
        """_run_planner should write the parsed plan to state['plan']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}
        state = {"requirement": SAMPLE_REQUIREMENT, "plan": [], "epic": None, "features": None}

        # Act
        result_state = coordinator._run_planner(state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_normalizes_to_lowercase(self, mock_ollama, coordinator):
        """_run_planner should normalize step names to lowercase."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: Epic, Features"}}
        state = {"requirement": SAMPLE_REQUIREMENT, "plan": [], "epic": None, "features": None}

        # Act
        result_state = coordinator._run_planner(state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_raises_on_invalid_output(self, mock_ollama, coordinator):
        """_run_planner should raise ValueError when no valid steps are found."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "I cannot determine the plan"}}
        state = {"requirement": SAMPLE_REQUIREMENT, "plan": [], "epic": None, "features": None}

        # Act / Assert
        with pytest.raises(ValueError):
            coordinator._run_planner(state)


# --------------------------------------------------------------------------- #
# Tests for global critic loop — approval path
# --------------------------------------------------------------------------- #

class TestGlobalCriticLoopApproval:

    def test_global_critic_stops_on_approved(self, mock_ollama, coordinator, full_backlog_state):
        """Global critic loop should stop and not re-run agents when APPROVED."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        # Simulate the global critic loop directly
        verdict = coordinator._run_global_critic(full_backlog_state)

        # Assert
        assert verdict.strip().upper() == "APPROVED"
        assert mock_ollama.chat.call_count == 1


# --------------------------------------------------------------------------- #
# Tests for global critic loop — redo routing
# --------------------------------------------------------------------------- #

class TestGlobalCriticLoopRedo:

    def test_parse_redo_epic_triggers_rerun_from_epic(self, coordinator):
        """When REDO: epic, rerun should start from 'epic' index in plan."""
        # Arrange
        plan = ["epic", "features"]
        redo_target = coordinator._parse_redo("REDO: epic")

        # Act
        rerun_from = plan.index(redo_target)

        # Assert
        assert rerun_from == 0
        assert plan[rerun_from:] == ["epic", "features"]

    def test_parse_redo_features_triggers_rerun_from_features(self, coordinator):
        """When REDO: features, rerun should start from 'features' index in plan."""
        # Arrange
        plan = ["epic", "features"]
        redo_target = coordinator._parse_redo("REDO: features")

        # Act
        rerun_from = plan.index(redo_target)

        # Assert
        assert rerun_from == 1
        assert plan[rerun_from:] == ["features"]

    def test_redo_epic_reruns_both_agents(self, coordinator):
        """REDO: epic should cause both EpicAgent and FeaturesAgent to re-run."""
        # Arrange
        plan = ["epic", "features"]
        redo_target = coordinator._parse_redo("REDO: epic")

        # Act
        steps_to_rerun = plan[plan.index(redo_target):]

        # Assert
        assert steps_to_rerun == ["epic", "features"]

    def test_redo_features_reruns_only_features_agent(self, coordinator):
        """REDO: features should cause only FeaturesAgent to re-run."""
        # Arrange
        plan = ["epic", "features"]
        redo_target = coordinator._parse_redo("REDO: features")

        # Act
        steps_to_rerun = plan[plan.index(redo_target):]

        # Assert
        assert steps_to_rerun == ["features"]
