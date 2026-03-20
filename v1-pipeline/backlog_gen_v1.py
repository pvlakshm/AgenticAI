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

def generate_epic(requirement):
    print("Generating epic...")
    return ask_llm("epic", requirement)

def generate_features(epic):
    print("Generating features...")
    return ask_llm("features", epic)

# --- Execution ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python backlog_gen_v1.py 'requirement'")
        return

    requirement = sys.argv[1]

    print(f"\nProcessing Requirement: {requirement}")

     # Run the sequence
    epic = generate_epic(requirement)
    features = generate_features(epic)

    # Final Output
    print("\n" + "=" * 60)
    print(f"\n[EPIC]\n{epic}")
    print(f"\n[FEATURES]\n{features}")

if __name__ == "__main__":
    main()