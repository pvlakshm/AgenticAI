import sys
import ollama

MODEL = "gemma3:1b"


def ask_llm(role, task, input_text, output_format):

    prompt = f"""
You are a professional {role}.

Your task:
{task}

INPUT:
{input_text}

OUTPUT FORMAT:
{output_format}

Rules:
- Follow the output format exactly.
- Do not add explanations.
- Do not add extra sections.
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


def generate_epic(requirement):

    return ask_llm(
        role="Product Manager",
        task="Create exactly ONE product epic from the requirement.",
        input_text=requirement,
        output_format="""
Epic:
<short epic title>

Description:
<1-2 sentence description>
""",
    )


def generate_features(epic):

    return ask_llm(
        role="Software Architect",
        task="Break the epic into 3 to 5 features.",
        input_text=epic,
        output_format="""
Features:
- feature 1
- feature 2
- feature 3
""",
    )


def generate_stories(features):

    return ask_llm(
        role="Scrum Master",
        task="Write user stories for each feature.",
        input_text=features,
        output_format="""
User Stories:
- As a <user>, I want <goal>, so that <benefit>.
- As a <user>, I want <goal>, so that <benefit>.
""",
    )


def generate_tests(stories):

    return ask_llm(
        role="QA Engineer",
        task="Generate one test case per user story.",
        input_text=stories,
        output_format="""
Test Cases:
- Test: <short name>
  Steps: <steps>
  Expected: <expected result>
""",
    )


def main():

    if len(sys.argv) < 2:
        print("Usage: python backlog_gen.py 'requirement'")
        return

    requirement = sys.argv[1]

    print("\nRequirement\n-----------")
    print(requirement)

    epic = generate_epic(requirement)
    print("\nEpic\n----")
    print(epic)

    features = generate_features(epic)
    print("\nFeatures\n--------")
    print(features)

    stories = generate_stories(features)
    print("\nUser Stories\n------------")
    print(stories)

    tests = generate_tests(stories)
    print("\nTest Cases\n----------")
    print(tests)


if __name__ == "__main__":
    main()