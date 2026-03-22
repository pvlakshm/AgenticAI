import pytest
from unittest.mock import Mock

import backlog_gen_v2 as v2


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
    v2.ollama = mock
    yield mock


@pytest.fixture
def pipeline_mock(mock_ollama):
    """Sets up two sequential LLM responses."""
    mock_ollama.chat.side_effect = [
        {"message": {"content": SAMPLE_EPIC}},
        {"message": {"content": SAMPLE_FEATURES}},
    ]
    return mock_ollama


@pytest.fixture
def initial_state():
    """Returns a freshly initialized shared state dict."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "epic": None,
        "features": None,
    }


@pytest.fixture
def state_with_epic():
    """Returns a state dict with the epic already populated."""
    return {
        "requirement": SAMPLE_REQUIREMENT,
        "epic": SAMPLE_EPIC,
        "features": None,
    }


# --------------------------------------------------------------------------- #
# Tests for generate_epic
# --------------------------------------------------------------------------- #

class TestGenerateEpic:

    def test_generate_epic_reads_requirement_from_state(self, mock_ollama, initial_state):
        """generate_epic should pass state['requirement'] to the LLM."""
        # Act
        v2.generate_epic(initial_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_REQUIREMENT in prompt

    def test_generate_epic_uses_epic_template_task(self, mock_ollama, initial_state):
        """generate_epic should call the LLM with the epic template task."""
        # Act
        v2.generate_epic(initial_state)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v2.TEMPLATES["epic"]["task"] in prompt

    def test_generate_epic_writes_epic_to_state(self, mock_ollama, initial_state):
        """generate_epic should write the LLM response to state['epic']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": SAMPLE_EPIC}}

        # Act
        result_state = v2.generate_epic(initial_state)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC

    def test_generate_epic_returns_state(self, mock_ollama, initial_state):
        """generate_epic should return the updated state dict."""
        # Act
        result_state = v2.generate_epic(initial_state)

        # Assert
        assert isinstance(result_state, dict)
        assert "epic" in result_state

    def test_generate_epic_does_not_modify_requirement(self, mock_ollama, initial_state):
        """generate_epic should not change state['requirement']."""
        # Act
        result_state = v2.generate_epic(initial_state)

        # Assert
        assert result_state["requirement"] == SAMPLE_REQUIREMENT


# --------------------------------------------------------------------------- #
# Tests for generate_features
# --------------------------------------------------------------------------- #

class TestGenerateFeatures:

    def test_generate_features_reads_epic_from_state(self, mock_ollama, state_with_epic):
        """generate_features should pass state['epic'] to the LLM."""
        # Act
        v2.generate_features(state_with_epic)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in prompt

    def test_generate_features_uses_features_template_task(self, mock_ollama, state_with_epic):
        """generate_features should call the LLM with the features template task."""
        # Act
        v2.generate_features(state_with_epic)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v2.TEMPLATES["features"]["task"] in prompt

    def test_generate_features_writes_features_to_state(self, mock_ollama, state_with_epic):
        """generate_features should write the LLM response to state['features']."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": SAMPLE_FEATURES}}

        # Act
        result_state = v2.generate_features(state_with_epic)

        # Assert
        assert result_state["features"] == SAMPLE_FEATURES

    def test_generate_features_returns_state(self, mock_ollama, state_with_epic):
        """generate_features should return the updated state dict."""
        # Act
        result_state = v2.generate_features(state_with_epic)

        # Assert
        assert isinstance(result_state, dict)
        assert "features" in result_state

    def test_generate_features_does_not_modify_epic(self, mock_ollama, state_with_epic):
        """generate_features should not change state['epic']."""
        # Act
        result_state = v2.generate_features(state_with_epic)

        # Assert
        assert result_state["epic"] == SAMPLE_EPIC


# --------------------------------------------------------------------------- #
# Tests for the shared state pipeline
# --------------------------------------------------------------------------- #

class TestSharedStatePipeline:

    def test_state_flows_through_pipeline(self, pipeline_mock, initial_state):
        """State should carry epic output into the features step."""
        # Act
        state = v2.generate_epic(initial_state)
        state = v2.generate_features(state)

        # Assert
        second_call_prompt = pipeline_mock.chat.call_args_list[1].kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in second_call_prompt

    def test_pipeline_populates_both_state_keys(self, pipeline_mock, initial_state):
        """After the full pipeline, state should have both epic and features populated."""
        # Act
        state = v2.generate_epic(initial_state)
        state = v2.generate_features(state)

        # Assert
        assert state["epic"] == SAMPLE_EPIC
        assert state["features"] == SAMPLE_FEATURES

    def test_pipeline_makes_exactly_two_llm_calls(self, pipeline_mock, initial_state):
        """The full pipeline should make exactly 2 LLM calls."""
        # Act
        state = v2.generate_epic(initial_state)
        v2.generate_features(state)

        # Assert
        assert pipeline_mock.chat.call_count == 2
