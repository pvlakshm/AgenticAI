# v5-multi-agent

## What this version does
Refactors the generator functions into self-contained agent classes. Each agent owns its own generation logic and critic loop. A Coordinator orchestrates the agents using the Planner's output.

## What's new
- **`EpicAgent`** class - owns `generate_epic()` and its critic loop
- **`FeaturesAgent`** class - owns `generate_features()` and its critic loop
- **`Coordinator`** class - initializes shared state, runs the Planner, and dispatches to agents
- **`AGENT_MAP`** - maps planner output strings to agent classes (replaces `TASK_MAP`)
- Each agent's `run()` method provides a uniform interface; the named `generate_*()` methods contain the actual logic

## Key concepts
- Agent as a self-contained unit of responsibility
- Uniform agent interface (`run(state) -> state`)
- Coordinator as the orchestration layer
- Planner retained - the LLM still decides which agents to run

## Why this matters
As systems grow, functions become hard to manage. Encapsulating generation + quality control into an agent class makes each unit independently testable, replaceable, and extensible. This is how real multi-agent frameworks like CrewAI and AutoGen are structured.

## What's unchanged from v4
- Critic loop logic
- All prompt templates
- Shared state pattern
- Planner logic (now inside `Coordinator._run_planner()`)

## How to run
```bash
python backlog_gen_v5.py "Grandma has a car and wants to know when she should refill fuel"
```