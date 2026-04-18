# import pytest
# from unittest.mock import Mock
# import backlog_gen_v1 as v1

# # --- Fixtures ---

# @pytest.fixture
# def standard_mock_llm():
#     """Arrange: Mock for successful happy-path scenarios."""
#     mock = Mock(spec=v1.OllamaLLM)
#     mock.generate.side_effect = [
#         "Epic Title\n\nAcceptance Criteria:\n1. AC1\n2. AC2\n3. AC3",
#         "Feature 1\n\nAcceptance Criteria:\n1. AC1\n2. AC2\n3. AC3"
#     ]
#     return mock

# @pytest.fixture
# def empty_mock_llm():
#     """Arrange: Mock for boundary scenarios."""
#     mock = Mock(spec=v1.OllamaLLM)
#     mock.generate.return_value = "Default Output"
#     return mock

# @pytest.fixture
# def prompts():
#     return {"epic": "Requirement: {input}", "feature": "Epic: {input}"}

# # --- Tests ---

# # Tests using the standard_mock_llm (Happy Path & Data Flow)
# def test_pipeline_output_structure(standard_mock_llm, prompts):
#     result = v1.run_pipeline("Req", prompts, standard_mock_llm)
#     assert "epic" in result and "features" in result
#     assert "Acceptance Criteria" in result["epic"]

# def test_pipeline_data_chaining(standard_mock_llm, prompts):
#     v1.run_pipeline("Req", prompts, standard_mock_llm)
#     second_call_input = standard_mock_llm.generate.call_args_list[1][0][0]
#     assert "Epic Title" in second_call_input

# def test_acceptance_criteria_presence(standard_mock_llm, prompts):
#     result = v1.run_pipeline("Req", prompts, standard_mock_llm)
#     assert "1. AC1" in result["epic"] and "1. AC1" in result["features"]

# # Tests using the empty_mock_llm (Boundary Condition)
# def test_pipeline_handles_empty_requirement(empty_mock_llm, prompts):
#     result = v1.run_pipeline("", prompts, empty_mock_llm)
#     assert result["epic"] == "Default Output"
#     assert result["features"] == "Default Output"







import pytest
from unittest.mock import Mock, call
import backlog_gen_v1 as v1


# --- Fixtures (minimal, reusable) ---

@pytest.fixture
def mock_llm():
    return Mock(spec=v1.OllamaLLM)


@pytest.fixture
def prompts():
    return {
        "epic": "Requirement: {input}",
        "feature": "Epic: {input}"
    }


# --- Helpers (explicit behavior, reusable) ---

def setup_happy_path(mock_llm):
    mock_llm.generate.side_effect = [
        "Epic Title\n\nAcceptance Criteria:\n1. AC1\n2. AC2\n3. AC3",
        "Feature 1\n\nAcceptance Criteria:\n1. AC1\n2. AC2\n3. AC3",
    ]


def setup_default_output(mock_llm):
    mock_llm.generate.return_value = "Default Output"


# --- Tests ---

def test_pipeline_prompt_flow(prompts, mock_llm):
    setup_happy_path(mock_llm)

    v1.run_pipeline("Req", prompts, mock_llm)

    calls = mock_llm.generate.call_args_list

    assert len(calls) == 2

    first_prompt = calls[0].args[0]
    second_prompt = calls[1].args[0]

    assert first_prompt == "Requirement: Req"
    assert second_prompt.startswith("Epic: Epic Title")
    
def test_pipeline_output_structure(prompts, mock_llm):
    # Arrange
    setup_happy_path(mock_llm)

    # Act
    result = v1.run_pipeline("Req", prompts, mock_llm)

    # Assert
    assert result.keys() == {"epic", "features"}
    assert "Acceptance Criteria" in result["epic"]
    assert "Acceptance Criteria" in result["features"]

def test_pipeline_handles_empty_input(prompts, mock_llm):
    # Arrange
    setup_default_output(mock_llm)

    # Act
    result = v1.run_pipeline("", prompts, mock_llm)

    # Assert
    assert result == {
        "epic": "Default Output",
        "features": "Default Output",
    }