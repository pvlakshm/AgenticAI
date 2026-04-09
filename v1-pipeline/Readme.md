# v1 - Sequential Pipeline (Single File)

## Overview
This is the simplest form of an agentic system:
a sequential LLM pipeline.

Requirement -> Epic -> Features

## Key Concepts
- Prompt templates stored as `.md`
- Prompt composition (system + task)
- Sequential LLM calls
- No shared state
- No dynamic control flow

## Usage

```bash
python pipeline.py "Users should be able to reset passwords"
```

### Run all tests from the root folder
```bash
python -m pytest -v
```