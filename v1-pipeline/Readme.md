# v1 - Scripted Pipeline
This version implements a simple backlog generator using role-based LLM prompts.

## Architecture:
Requirement
→ Product Manager creates Epic
→ Product Manager creates Features
→ Product Owner creates User Stories
→ QA Engineer creates Test Cases

Python orchestrates the pipeline while the LLM acts as role-based pseudo-agents.

LLM Runtime: Ollama
Model: gemma3:1b

## Run:
python backlog_gen.py "your requirement"