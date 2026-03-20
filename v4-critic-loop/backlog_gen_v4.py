import sys
import ollama

MODEL = "qwen3-coder:480b-cloud"

# --- Prompt Templates Configuration ---
TEMPLATES = {
    "planner": {
        "role": "Product Manager",
        "task": "Analyze the user requirement and determine which backlog artifacts need to be generated. You can choose from: epic, features.",
        "format": "Plan: <comma-separated list of steps, e.g., epic, features>"
    },
    "epic": {
        "role": "Product Manager",
        "task": "Create exactly ONE epic from the requirement and define business-level acceptance criteria.",
        "format": """
Epic: <short epic title>
Description: <1-2 sentence description>
Acceptance Criteria:
- criterion 1
- criterion 2
- criterion 3
"""
    },

    "features": {
        "role": "Product Manager",
        "task": "Break the epic into 3 features. Each feature must have acceptance criteria.",
        "format": """
Feature: <feature name>
Description: <short description>
Acceptance Criteria:
- criterion 1
- criterion 2
- criterion 3

Feature: <feature name>
Description: <short description>
Acceptance Criteria:
- criterion 1
- criterion 2
- criterion 3
"""
    },
    "critic": {
        "role": "Senior Product Manager and Quality Reviewer",
        "task": (
            "Review the generated backlog artifact against the original requirement."
            "Identify specific issues: missing coverage, or misalignment with the requirement."
            "If the artifact is acceptable, respond with exactly: APPROVED."
            "Otherwise, respond with: REVISION NEEDED: <concise bullet list of issues to fix>"
        ),
        "format": "APPROVED  OR  REVISION NEEDED: <bullet list of specific issues>"
    },
    "revise": {
        "role": "Product Manager",
        "task": "Revise the backlog artifact based on the critic's feedback. Apply every requested change while keeping the same output format.",
        "format": "<same format as the original artifact>"
    },
}

MAX_REVISIONS = 2

def ask_llm(template_key, input_text):
    config = TEMPLATES[template_key]

    prompt = f"""
You are a professional {config['role']}.

Task:
{config['task']}

INPUT:
{input_text}

OUTPUT FORMAT:
{config['format']}

Rules:
- Follow the output format exactly
- Do not add explanations
- Do not add extra sections
"""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )

    return response["message"]["content"].strip()


# --- Critic Loop Helper ---

def critic_loop(artifact_key, artifact_text, requirement):
    """
    Run up to MAX_REVISIONS critic-revise cycles for a single artifact.
    Returns the (possibly revised) artifact text.
    """
    current = artifact_text

    for attempt in range(1, MAX_REVISIONS + 1):
        print(f"  Critiquing {artifact_key} (attempt {attempt}/{MAX_REVISIONS})...")

        critic_input = (
            f"ORIGINAL REQUIREMENT:\n{requirement}\n\n"
            f"ARTIFACT TO REVIEW:\n{current}"
        )
        print(f"\n  --- Critic Input ---\n{critic_input}\n  --------------------")
        feedback = ask_llm("critic", critic_input)

        if feedback.strip().upper().startswith("APPROVED"):
            print(f"  {artifact_key.capitalize()} approved.")
            break

        # Revision needed
        print(f"  Critic feedback: {feedback}")

        if attempt < MAX_REVISIONS:
            print(f"  Revising {artifact_key}...")
            revise_input = (
                f"ORIGINAL REQUIREMENT:\n{requirement}\n\n"
                f"CURRENT ARTIFACT:\n{current}\n\n"
                f"CRITIC FEEDBACK:\n{feedback}"
            )
            current = ask_llm("revise", revise_input)
    else:
        print(f"  \nMax revisions reached. APPROVING {artifact_key} with feedback.\n")

    return current


# --- Specialized Functions ---

def generate_epic(state):
    print("Generating epic...")
    raw_epic = ask_llm("epic", state["requirement"])
    state["epic"] = critic_loop("epic", raw_epic, state["requirement"])
    return state

def generate_features(state):
    print("Generating features...")
    raw_features = ask_llm("features", state["epic"])
    state["features"] = critic_loop("features", raw_features, state["requirement"])
    return state

# Map the strings from the Planner to our functions
TASK_MAP = {
    "epic": generate_epic,
    "features": generate_features
}

# --- The Planner ---

def run_planner(state):
    print("Planning workflow...")
    plan_raw = ask_llm("planner", state["requirement"])

    valid_steps = list(TASK_MAP.keys())
    steps = [s.strip() for s in plan_raw.replace("Plan:", "").split(",")]
    steps = [s for s in steps if s in valid_steps]

    if not steps:
        raise ValueError(
            f"Planner returned invalid output: '{plan_raw}'\n"
            f"Expected a plan containing one or more of: {valid_steps}"
        )

    state["plan"] = steps
    return state


# --- Execution ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python backlog_gen_v4.py 'requirement'")
        return

    requirement = sys.argv[1]

    # Initialize the Shared State
    state = {
        "requirement": requirement,
        "plan": [],
        "epic": None,
        "features": None,
    }

    print(f"\nProcessing Requirement: {state['requirement']}")

    # Let the Planner decide the sequence
    state = run_planner(state)
    print(f"Confirmed Plan: {state['plan']}")

    # Run the sequence
    for step in state["plan"]:
        if step in TASK_MAP:
            state = TASK_MAP[step](state)

    # Final Output
    for step in state["plan"]:
        print(f"\n[{step.upper()}]")
        print(state.get(step, "Not generated"))

if __name__ == "__main__":
    main()