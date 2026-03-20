# v4-critic-loop

## What this version does
Adds a critic-revise loop around each generated artifact. After generation, a critic LLM reviews the artifact and either approves it or requests specific revisions. The generator revises based on the feedback, up to `MAX_REVISIONS` times.

## What's new
- **`critic` template** - reviews an artifact against the original requirement, returns `APPROVED` or `REVISION NEEDED: <issues>`
- **`revise` template** - rewrites an artifact based on critic feedback
- **`critic_loop()`** - runs up to `MAX_REVISIONS` critique-revise cycles for a single artifact
- **`MAX_REVISIONS`** constant - controls the maximum number of revision attempts
- Each generator function now wraps its output in `critic_loop()` before storing in state

## Key concepts
- Self-correction loop
- Separation of generation and evaluation
- Fail loudly - if the critic cannot approve after `MAX_REVISIONS`, the pipeline halts with an error rather than passing a substandard artifact downstream

## Why this matters
A bad epic will produce bad features. Quality gating at each step prevents errors from propagating. This pattern - generate, critique, revise - is one of the most common patterns in production agentic systems.

## What's unchanged from v3
- Planner and dynamic execution
- Shared state pattern
- Prompt templates for epic and features

## How to run
```bash
python backlog_gen_v4.py "Grandma has a car and wants to know when she should refill fuel"
```