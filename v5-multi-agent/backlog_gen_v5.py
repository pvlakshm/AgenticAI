import sys
import ollama

MODEL = "qwen3-coder:480b-cloud"

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


# --- Agents ---

class EpicAgent:
    """Responsible for generating and refining a single Epic."""

    def run(self, state):
        print("\n[EpicAgent] Starting...")
        state = self.generate_epic(state)
        print("[EpicAgent] Done.")
        return state

    def generate_epic(self, state):
        print("Generating epic...")
        raw = ask_llm("epic", state["requirement"])
        state["epic"] = critic_loop("epic", raw, state["requirement"])
        return state


class FeaturesAgent:
    """Responsible for generating and refining Features from an Epic."""

    def run(self, state):
        print("\n[FeaturesAgent] Starting...")
        state = self.generate_features(state)
        print("[FeaturesAgent] Done.")
        return state

    def generate_features(self, state):
        print("Generating features...")
        raw_features = ask_llm("features", state["epic"])
        state["features"] = critic_loop("features", raw_features, state["requirement"])
        return state


# Map the strings from the Planner to our agent classes
AGENT_MAP = {
    "epic": EpicAgent,
    "features": FeaturesAgent,
}


# --- Coordinator ---

class Coordinator:
    """Uses the Planner to decide which agents to run, then orchestrates them."""

    def run(self, requirement):
        # Initialize the Shared State
        state = {
            "requirement": requirement,
            "plan": [],
            "epic": None,
            "features": None,
        }

        print(f"\nProcessing Requirement: {state['requirement']}")

        # Let the Planner decide the sequence
        state = self._run_planner(state)
        print(f"Confirmed Plan: {state['plan']}")

        # Run agents in planned sequence
        for step in state["plan"]:
            agent = AGENT_MAP[step]()
            state = agent.run(state)

        # Final Output
        print("\n" + "=" * 60)
        for step in state["plan"]:
            print(f"\n[{step.upper()}]")
            print(state.get(step, "Not generated"))

    def _run_planner(self, state):
        print("Planning workflow...")
        plan_raw = ask_llm("planner", state["requirement"])

        valid_steps = list(AGENT_MAP.keys())
        steps = [s.strip().lower() for s in plan_raw.replace("Plan:", "").split(",")]
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
        print("Usage: python backlog_gen_v5.py 'requirement'")
        return

    requirement = sys.argv[1]

    coordinator = Coordinator()
    coordinator.run(requirement)

if __name__ == "__main__":
    main()