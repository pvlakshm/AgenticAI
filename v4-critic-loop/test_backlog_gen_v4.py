import pytest
from unittest.mock import Mock

import backlog_gen_v4 as v4


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

SAMPLE_EPIC_REVISED = """Epic: User Authentication
Description: Enable users to securely log in to the application with full security.
Acceptance Criteria:
- The system provides a login page with email and password fields
- The system validates credentials and grants access on success
- The system displays an error message on failed login
- The system supports password reset"""

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
    v4.ollama = mock
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


# --------------------------------------------------------------------------- #
# Tests for critic_loop — approval path
# --------------------------------------------------------------------------- #

class TestCriticLoopApproval:

    def test_critic_loop_returns_artifact_unchanged_on_immediate_approval(self, mock_ollama):
        """critic_loop should return the original artifact when critic approves immediately."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        result = v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        assert result == SAMPLE_EPIC

    def test_critic_loop_makes_one_llm_call_on_immediate_approval(self, mock_ollama):
        """critic_loop should make exactly 1 LLM call when critic approves immediately."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "APPROVED"}}

        # Act
        v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        assert mock_ollama.chat.call_count == 1

    def test_critic_loop_approval_is_case_insensitive(self, mock_ollama):
        """critic_loop should treat 'approved', 'Approved', 'APPROVED' all as approval."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "approved"}}

        # Act
        result = v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        assert result == SAMPLE_EPIC


# --------------------------------------------------------------------------- #
# Tests for critic_loop — revision path
# --------------------------------------------------------------------------- #

class TestCriticLoopRevision:

    def test_critic_loop_revises_on_revision_needed(self, mock_ollama):
        """critic_loop should call the revise template when critic requests revision."""
        # Arrange
        mock_ollama.chat.side_effect = [
            {"message": {"content": "REVISION NEEDED: missing security criteria"}},
            {"message": {"content": SAMPLE_EPIC_REVISED}},
            {"message": {"content": "APPROVED"}},
        ]

        # Act
        result = v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        assert result == SAMPLE_EPIC_REVISED

    def test_critic_loop_includes_feedback_in_revise_prompt(self, mock_ollama):
        """critic_loop should include critic feedback in the revise prompt."""
        # Arrange
        mock_ollama.chat.side_effect = [
            {"message": {"content": "REVISION NEEDED: missing security criteria"}},
            {"message": {"content": SAMPLE_EPIC_REVISED}},
            {"message": {"content": "APPROVED"}},
        ]

        # Act
        v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        revise_prompt = mock_ollama.chat.call_args_list[1].kwargs["messages"][0]["content"]
        assert "missing security criteria" in revise_prompt

    def test_critic_loop_returns_last_revision_when_max_reached(self, mock_ollama):
        """critic_loop should return the last revised artifact when MAX_REVISIONS is reached."""
        # Arrange
        mock_ollama.chat.side_effect = [
            {"message": {"content": "REVISION NEEDED: issue 1"}},
            {"message": {"content": SAMPLE_EPIC_REVISED}},
            {"message": {"content": "REVISION NEEDED: issue 2"}},
        ]

        # Act
        result = v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        assert result == SAMPLE_EPIC_REVISED

    def test_critic_loop_does_not_exceed_max_revisions(self, mock_ollama):
        """critic_loop should make at most MAX_REVISIONS * 2 - 1 LLM calls."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "REVISION NEEDED: always failing"}}

        # Act
        v4.critic_loop("epic", SAMPLE_EPIC, SAMPLE_REQUIREMENT)

        # Assert
        # MAX_REVISIONS critic calls + (MAX_REVISIONS - 1) revise calls
        expected_max_calls = v4.MAX_REVISIONS + (v4.MAX_REVISIONS - 1)
        assert mock_ollama.chat.call_count <= expected_max_calls


# --------------------------------------------------------------------------- #
# Tests for generate_epic with critic loop
# --------------------------------------------------------------------------- #

class TestGenerateEpicWithCriticLoop:

    def test_generate_epic_stores_approved_artifact_in_state(self, mock_ollama, initial_state):
        """generate_epic should store the approved artifact in state['epic']."""
        # Arrange
        mock_ollama.chat.side_effect = [
            {"message": {"content": SAMPLE_EPIC}},   # generation
            {"message": {"content": "APPROVED"}},     # critic
        ]

        # Act
        result_state = v4.generate_epic(initial_state)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC

    def test_generate_epic_stores_revised_artifact_in_state(self, mock_ollama, initial_state):
        """generate_epic should store the revised artifact when revision was needed."""
        # Arrange
        mock_ollama.chat.side_effect = [
            {"message": {"content": SAMPLE_EPIC}},          # generation
            {"message": {"content": "REVISION NEEDED: x"}}, # critic
            {"message": {"content": SAMPLE_EPIC_REVISED}},  # revise
            {"message": {"content": "APPROVED"}},            # critic again
        ]

        # Act
        result_state = v4.generate_epic(initial_state)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC_REVISED


# --------------------------------------------------------------------------- #
# Tests for run_planner
# --------------------------------------------------------------------------- #

class TestRunPlanner:

    def test_planner_writes_plan_to_state(self, mock_ollama, initial_state):
        """run_planner should write the parsed plan to state['plan']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "Plan: epic, features"}}
        initial_state["plan"] = []

        # Act
        result_state = v4.run_planner(initial_state)

        # Assert
        assert result_state["plan"] == ["epic", "features"]

    def test_planner_raises_on_invalid_output(self, mock_ollama, initial_state):
        """run_planner should raise ValueError when no valid steps are returned."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "I cannot determine the plan"}}
        initial_state["plan"] = []

        # Act / Assert
        with pytest.raises(ValueError):
            v4.run_planner(initial_state)
