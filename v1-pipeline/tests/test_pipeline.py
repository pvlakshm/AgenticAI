import pytest
from pipeline import run_pipeline


# -----------------------------
# Fixtures (Arrange)
# -----------------------------
@pytest.fixture
def prompts():
    return {
        "system": (
            "You are a professional {role}.\n\n"
            "Task:\n{task}\n\n"
            "INPUT:\n{input_text}\n\n"
            "OUTPUT FORMAT:\n{format}\n\n"
            "Rules:\n"
            "- Follow the output format exactly\n"
            "- Do not add explanations\n"
            "- Do not add extra sections"
        ),
        "epic": (
            "# Role\nProduct Manager\n\n"
            "# Task\nCreate exactly ONE epic from the requirement and define business-level acceptance criteria.\n\n"
            "# Format\nEpic..."
        ),
        "feature": (
            "# Role\nProduct Manager\n\n"
            "# Task\nBreak the epic into exactly 3 features. Each feature must have acceptance criteria.\n\n"
            "# Format\nFeature..."
        )
    }


@pytest.fixture
def fake_llm():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompt):
            self.calls.append(prompt)

            if "Create exactly ONE epic" in prompt:
                return (
                    "Epic: Password Reset\n"
                    "Description: Reset passwords\n"
                    "Acceptance Criteria:\n- A\n- B\n- C"
                )

            if "Break the epic into exactly 3 features" in prompt:
                return (
                    "Feature: Email Reset\n"
                    "Description: Email flow\n"
                    "Acceptance Criteria:\n- A\n- B\n- C"
                )

            return "UNKNOWN"

    return FakeLLM()


# -----------------------------
# Tests (AAA)
# -----------------------------
def test_pipeline_runs(prompts, fake_llm):
    # Arrange
    requirement = "Users should be able to reset passwords"

    # Act
    result = run_pipeline(requirement, prompts, fake_llm)

    # Assert
    assert "Epic:" in result["epic"]
    assert "Feature:" in result["features"]


def test_llm_called_twice(prompts, fake_llm):
    # Arrange
    requirement = "Reset password"

    # Act
    run_pipeline(requirement, prompts, fake_llm)

    # Assert
    assert len(fake_llm.calls) == 2


def test_epic_output_flows_into_feature_step(prompts, fake_llm):
    # Arrange
    requirement = "Reset password"

    # Act
    run_pipeline(requirement, prompts, fake_llm)

    # Assert
    second_prompt = fake_llm.calls[1]
    assert "Epic:" in second_prompt


def test_prompt_contains_requirement(prompts, fake_llm):
    # Arrange
    requirement = "Reset password"

    # Act
    run_pipeline(requirement, prompts, fake_llm)

    # Assert
    first_prompt = fake_llm.calls[0]
    assert requirement in first_prompt