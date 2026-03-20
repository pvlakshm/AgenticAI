# v6-autonomous

## What this version does
Adds a global critic that reviews the full backlog (epic + features together) after all agents complete. If the full backlog fails review, the global critic decides which agent to re-run - and the Coordinator selectively re-executes from that point forward, without discarding work that was already approved.

## What's new
- **`global_critic` template** - reviews the full backlog holistically using objective, countable rules. Returns one of: `APPROVED`, `REDO: epic`, or `REDO: features` with a reason
- **`MAX_GLOBAL_RETRIES`** constant - controls the maximum number of global retry cycles, independent of `MAX_REVISIONS`
- **Global critic loop** in `Coordinator.run()` - runs after all agents complete, routes re-execution based on the verdict
- **`_run_global_critic()`** - assembles the full backlog into a single input and calls the global critic
- **`_parse_redo()`** - extracts the agent key from a `REDO` verdict, keeping routing logic in the LLM
- **Selective re-execution** - if `REDO: features`, only FeaturesAgent re-runs. If `REDO: epic`, both EpicAgent and FeaturesAgent re-run (since features depend on the epic)

## Key concepts
- Holistic quality review across multiple artifacts
- LLM-driven routing - the model decides where to re-enter the pipeline
- Selective re-execution - don't discard good work, only redo what failed
- Two-level quality control: per-artifact critic loop (v4) + global backlog critic (v6)
- Autonomous termination - the system decides when the backlog is done

## Why this matters
In real agentic systems, individual artifact quality is not enough - the outputs must also be coherent with each other. A global critic that can route re-execution to the right agent, without restarting the entire pipeline, is a key pattern in production systems like LangGraph.

## What's unchanged from v5
- All agent classes and their internal critic loops
- Coordinator structure and Planner logic
- All prompt templates except the addition of `global_critic`
- Shared state pattern

## How to run
```bash
python backlog_gen_v6.py "Grandma has a car and wants to know when she should refill fuel"
```