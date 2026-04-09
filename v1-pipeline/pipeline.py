import os
import sys
import ollama


# -----------------------------
# Prompt Loading
# -----------------------------
def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompts(prompt_dir):
    prompts = {}
    for filename in os.listdir(prompt_dir):
        if filename.endswith(".md"):
            key = filename.replace(".prompt.md", "")
            prompts[key] = load_prompt(os.path.join(prompt_dir, filename))
    return prompts


# -----------------------------
# Prompt Rendering
# -----------------------------
def parse_sections(markdown_text):
    sections = {}
    current_key = None
    buffer = []

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = line.replace("# ", "").strip().lower()
            buffer = []
        else:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def render_prompt(system_prompt, task_prompt, input_text):
    sections = parse_sections(task_prompt)

    return system_prompt.format(
        role=sections.get("role", ""),
        task=sections.get("task", ""),
        input_text=input_text,
        format=sections.get("format", "")
    )


# -----------------------------
# LLM Abstraction (Ollama)
# -----------------------------
class OllamaLLM:
    def __init__(self, model=None, temperature=0.2):
        self.model = model or os.environ.get("MODEL", "gemma3:1b")
        self.temperature = temperature

    def generate(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": self.temperature
            }
        )

        return response["message"]["content"].strip()


def ask_llm(prompt, llm_client):
    return llm_client.generate(prompt)


# -----------------------------
# Pipeline (Sequential)
# -----------------------------
def run_pipeline(requirement, prompts, llm_client):
    system_prompt = prompts["system"]

    # Step 1: Epic
    epic_prompt = render_prompt(
        system_prompt,
        prompts["epic"],
        requirement
    )
    epic_output = ask_llm(epic_prompt, llm_client)

    # Step 2: Features
    feature_prompt = render_prompt(
        system_prompt,
        prompts["feature"],
        epic_output
    )
    feature_output = ask_llm(feature_prompt, llm_client)

    return {
        "epic": epic_output,
        "features": feature_output
    }


# -----------------------------
# CLI Entry Point (STABLE)
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"<requirement>\"")
        sys.exit(1)

    requirement = sys.argv[1]

    prompts = load_prompts("prompts")

    llm = OllamaLLM()

    result = run_pipeline(requirement, prompts, llm)

    print("=== EPIC ===")
    print(result["epic"])

    print("\n=== FEATURES ===")
    print(result["features"])