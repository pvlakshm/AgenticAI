# v1-pipeline

## What this version does
The simplest possible backlog generator. Takes a requirement string, generates an Epic, then passes the Epic to generate 3 Features. Each step is a plain function call.

## What's new
- Everything. This is the baseline.
- `ask_llm()` - the core LLM call, driven by prompt templates
- `generate_epic()` and `generate_features()` - two specialized functions
- A hardcoded sequential pipeline: epic -> features

## Key concepts
- Prompt templates (`TEMPLATES` dict)
- Sequential pipeline
- LLM chaining (output of one step feeds the next)

## Limitations
- No shared state - outputs are passed as raw strings between functions
- No dynamic planning - the sequence is always epic → features
- No quality control - whatever the LLM generates is accepted as-is

## How to run
```bash
python backlog_gen_v1.py "Grandma has a car and wants to know when she should refill fuel"
```