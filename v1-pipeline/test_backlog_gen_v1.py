import pytest
from unittest.mock import Mock

import backlog_gen_v1 as v1


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
    v1.ollama = mock
    yield mock


@pytest.fixture
def pipeline_mock(mock_ollama):
    """Sets up two sequential LLM responses."""
    mock_ollama.chat.side_effect = [
        {"message": {"content": SAMPLE_EPIC}},
        {"message": {"content": SAMPLE_FEATURES}},
    ]
    return mock_ollama


# --------------------------------------------------------------------------- #
# Tests for ask_llm
# --------------------------------------------------------------------------- #

class TestAskLlm:

    def test_ask_llm_calls_ollama_with_correct_model(self, mock_ollama):
        """ask_llm should call ollama.chat with the configured MODEL."""
        # Act
        v1.ask_llm("epic", SAMPLE_REQUIREMENT)

        # Assert
        assert mock_ollama.chat.call_args.kwargs["model"] == v1.MODEL

    def test_ask_llm_uses_correct_template_role(self, mock_ollama):
        """ask_llm should include the template role in the prompt."""
        # Act
        v1.ask_llm("epic", SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["epic"]["role"] in prompt

    def test_ask_llm_uses_correct_template_task(self, mock_ollama):
        """ask_llm should include the template task in the prompt."""
        # Act
        v1.ask_llm("epic", SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["epic"]["task"] in prompt

    def test_ask_llm_includes_input_in_prompt(self, mock_ollama):
        """ask_llm should include the input_text in the prompt sent to the LLM."""
        # Act
        v1.ask_llm("epic", SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_REQUIREMENT in prompt

    def test_ask_llm_strips_response(self, mock_ollama):
        """ask_llm should strip whitespace from the LLM response."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": "  \n  some response  \n  "}}

        # Act
        result = v1.ask_llm("epic", SAMPLE_REQUIREMENT)

        # Assert
        assert result == "some response"

    def test_ask_llm_raises_on_invalid_template_key(self, mock_ollama):
        """ask_llm should raise a KeyError for an unknown template key."""
        # Act / Assert
        with pytest.raises(KeyError):
            v1.ask_llm("nonexistent_template", SAMPLE_REQUIREMENT)


# --------------------------------------------------------------------------- #
# Tests for generate_epic
# --------------------------------------------------------------------------- #

class TestGenerateEpic:

    def test_generate_epic_calls_ask_llm_with_epic_template(self, mock_ollama):
        """generate_epic should call ollama.chat with the epic template task."""
        # Act
        v1.generate_epic(SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["epic"]["task"] in prompt

    def test_generate_epic_uses_epic_role(self, mock_ollama):
        """generate_epic should call ollama.chat with the epic template role."""
        # Act
        v1.generate_epic(SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["epic"]["role"] in prompt

    def test_generate_epic_passes_requirement_to_llm(self, mock_ollama):
        """generate_epic should pass the requirement string to the LLM."""
        # Act
        v1.generate_epic(SAMPLE_REQUIREMENT)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_REQUIREMENT in prompt

    def test_generate_epic_returns_llm_output(self, mock_ollama):
        """generate_epic should return the LLM response."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": SAMPLE_EPIC}}

        # Act
        result = v1.generate_epic(SAMPLE_REQUIREMENT)

        # Assert
        assert result == SAMPLE_EPIC


# --------------------------------------------------------------------------- #
# Tests for generate_features
# --------------------------------------------------------------------------- #

class TestGenerateFeatures:

    def test_generate_features_calls_ask_llm_with_features_template(self, mock_ollama):
        """generate_features should call ollama.chat with the features template task."""
        # Act
        v1.generate_features(SAMPLE_EPIC)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["features"]["task"] in prompt

    def test_generate_features_uses_features_role(self, mock_ollama):
        """generate_features should call ollama.chat with the features template role."""
        # Act
        v1.generate_features(SAMPLE_EPIC)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["features"]["role"] in prompt

    def test_generate_features_passes_epic_to_llm(self, mock_ollama):
        """generate_features should pass the epic string to the LLM."""
        # Act
        v1.generate_features(SAMPLE_EPIC)

        # Assert
        prompt = mock_ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in prompt

    def test_generate_features_returns_llm_output(self, mock_ollama):
        """generate_features should return the LLM response."""
        # Arrange
        mock_ollama.chat.return_value = {"message": {"content": SAMPLE_FEATURES}}

        # Act
        result = v1.generate_features(SAMPLE_EPIC)

        # Assert
        assert result == SAMPLE_FEATURES


# --------------------------------------------------------------------------- #
# Tests for the pipeline sequence
# --------------------------------------------------------------------------- #

class TestPipeline:

    def test_pipeline_passes_epic_output_to_features(self, pipeline_mock):
        """The epic output should be passed as input to generate_features."""
        # Act
        epic = v1.generate_epic(SAMPLE_REQUIREMENT)
        v1.generate_features(epic)

        # Assert
        second_call_prompt = pipeline_mock.chat.call_args_list[1].kwargs["messages"][0]["content"]
        assert SAMPLE_EPIC in second_call_prompt

    def test_pipeline_makes_exactly_two_llm_calls(self, pipeline_mock):
        """The full pipeline (epic + features) should make exactly 2 LLM calls."""
        # Act
        epic = v1.generate_epic(SAMPLE_REQUIREMENT)
        v1.generate_features(epic)

        # Assert
        assert pipeline_mock.chat.call_count == 2

    def test_pipeline_call_order_is_epic_then_features(self, pipeline_mock):
        """The LLM should be called with epic template first, then features template."""
        # Act
        epic = v1.generate_epic(SAMPLE_REQUIREMENT)
        v1.generate_features(epic)

        # Assert
        first_call_prompt = pipeline_mock.chat.call_args_list[0].kwargs["messages"][0]["content"]
        second_call_prompt = pipeline_mock.chat.call_args_list[1].kwargs["messages"][0]["content"]
        assert v1.TEMPLATES["epic"]["task"] in first_call_prompt
        assert v1.TEMPLATES["features"]["task"] in second_call_prompt