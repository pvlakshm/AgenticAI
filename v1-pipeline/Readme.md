# v1 - Scripted Pipeline
This version implements a simple backlog generator using role-based LLM prompts.

## 1. Fixed CLI interface
python backlog_gen.py "requirement"

## 2. Python as orchestrator
Pipeline:
Requirement
Product Manager --> Epic
Product Manager --> Features
Product Owner --> User Stories
QA Engineer --> Test Cases

### Controlled here:
epic = generate_epic(requirement)
features = generate_features(epic)
stories = generate_stories(features)
tests = generate_tests(stories)

## 3. Role-based pseudo agents
Roles are explicit in prompts:
role="Product Manager"
role="Product Owner"
role="QA Engineer"

## 4. Acceptance criteria defined at every level
Prompts require acceptance criteria. For tests it is implicit via expected result

## 5. Deterministic prompts (important for small models)
Prompt structure:
Task
INPUT
OUTPUT FORMAT
Rules

Works well with Gemma 3 running via Ollama.

## 6. Keeps results somewhat consistent
options={"temperature": 0.2}
