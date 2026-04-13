import os
import sys
import ollama

# -----------------------------
# Prompt Loading
# -----------------------------
def load_prompts(prompt_dir):
    prompts = {}
    for filename in os.listdir(prompt_dir):
        if filename.endswith(".md"):
            key = filename.replace(".prompt.md", "")
            with open(os.path.join(prompt_dir, filename), "r", encoding="utf-8") as f:
                prompts[key] = f.read()
    return prompts

# -----------------------------
# LLM Abstraction
# -----------------------------
class OllamaLLM:
    def __init__(self, model=None, temperature=0.2):
        self.model = model or os.environ.get("MODEL", "gemma3:1b")
        self.temperature = temperature

    def generate(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature}
        )
        return response["message"]["content"].strip()

# -----------------------------
# Simple Pipeline
# -----------------------------
def run_pipeline(requirement, prompts, llm_client):
    # Step 1: Epic (using the epic prompt file directly)
    epic_input = f"Requirement: {requirement}"
    epic_prompt = prompts["epic"].format(input=epic_input)
    epic_output = llm_client.generate(epic_prompt)

    # Step 2: Features (using the feature prompt file directly)
    feature_input = f"Epic: {epic_output}"
    feature_prompt = prompts["feature"].format(input=feature_input)
    feature_output = llm_client.generate(feature_prompt)

    return {"epic": epic_output, "features": feature_output}

# -----------------------------
# Execution
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"<requirement>\"")
        sys.exit(1)

    requirement = sys.argv[1]
    prompts = load_prompts("prompts")
    llm = OllamaLLM()

    result = run_pipeline(requirement, prompts, llm)

    print("=== EPIC ===\n", result["epic"])
    print("\n=== FEATURES ===\n", result["features"])

if __name__ == "__main__":
    main()