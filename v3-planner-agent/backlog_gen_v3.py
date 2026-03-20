import sys
import ollama

MODEL = "gemma3:1b"

# --- Prompt Templates Configuration ---
TEMPLATES = {
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
    "planner": {
        "role": "Product Manager",
        "task": "Analyze the user requirement and determine which backlog artifacts need to be generated. You can choose from: epic, features.",
        "format": "Plan: <comma-separated list of steps, e.g., epic, features>"
    },
}

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

# --- Specialized Functions ---

def generate_epic(state):
    print("Generating epic...")
    state["epic"] = ask_llm("epic", state["requirement"])
    return state

def generate_features(state):
    print("Generating features...")
    state["features"] = ask_llm("features", state["epic"])
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
        print("Usage: python backlog_gen_v3.py 'requirement'")
        return

    requirement = sys.argv[1]

    # Initialize the Shared State
    state = {
        "requirement": requirement,
        "plan": [],
        "epic": None,
        "features": None
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
    print("\n" + "=" * 60)
    for step in state["plan"]:
        print(f"\n[{step.upper()}]")
        print(state.get(step, "Not generated"))

if __name__ == "__main__":
    main()