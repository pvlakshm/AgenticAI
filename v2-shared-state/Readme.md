# v2-shared-state

## What this version does
Same pipeline as v1, but introduces a shared `state` dictionary that flows through every function. Each function reads from and writes to this shared context.

## What's new
- **Shared state** - a single `dict` initialized in `main()` and passed through every function
- `generate_epic(state)` and `generate_features(state)` now accept and return `state` instead of raw strings

## Key concepts
- Shared state as the single source of truth
- Functions as state transformers (read state -> do work -> write state -> return state)

## Why this matters
As pipelines grow, passing individual strings between functions becomes unmanageable. Shared state gives every function access to the full context without complex argument lists. This pattern is the foundation for everything that follows.

## What's unchanged from v1
- Prompt templates
- Hardcoded sequential pipeline
- No dynamic planning
- No quality control

## How to run
```bash
python backlog_gen_v2.py "Grandma has a car and wants to know when she should refill fuel"
```