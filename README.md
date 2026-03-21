# Agentic Backlog Generator - Stepwise Refinement

An agentic AI product backlog generator in six progressive versions. Each version introduces one new pattern, building on the previous, until the system is fully autonomous.

---

## $0. No API Key. No Credit Card. No Surprises.

This project is intentionally $0. No OpenAI API key, no cloud credits, no free tiers that quietly start charging when you cross a token limit. If the fear of cloud costs has been stopping you from learning agentic AI - it doesn't have to.

---

## Approach

This project is loosely inspired by Niklaus Wirth's classic "Program Development by Stepwise Refinement" - the idea that you build understanding by starting simple and adding one concept at a time. It works just as well for agentic AI as it did for structured programming. In this case, each version introduces exactly one new agentic pattern, building on the previous, until the system is fully autonomous.

---

## What This Project Does

Given a plain-English product requirement, the system generates:
- **1 Epic** with business-level acceptance criteria
- **3 Features** derived from the Epic, each with acceptance criteria

The requirement is passed via the command line:

```bash
python backlog_gen_v6.py "Grandma has a car and wants to know when she should refill fuel"
```

---

## Prerequisites

- Python 3.10+
- [ollama](https://ollama.com) installed and running locally

---

## Installation

```bash
pip install ollama
ollama pull gemma3:1b
ollama pull qwen3-coder:480b-cloud
```

> **Note on models:** v1, v2, and v3 use `gemma3:1b` for simplicity. v4 onwards use `qwen3-coder:480b-cloud` for better instruction-following. If you want to use a different model, change the `MODEL` constant at the top of each file. `mistral` and `llama3.2:3b` are good alternatives.

---

## The Six Versions

| Version | Pattern | What's Introduced |
|---------|---------|-------------------|
| [v1-pipeline](v1-pipeline/Readme.md) | Sequential Pipeline | `ask_llm()`, prompt templates, LLM chaining |
| [v2-shared-state](v2-shared-state/Readme.md) | Shared State | Single `state` dict passed through all functions |
| [v3-planner-agent](v3-planner-agent/Readme.md) | Planner Agent | LLM decides which steps to run at runtime |
| [v4-critic-loop](v4-critic-loop/Readme.md) | Critic Loop | Per-artifact critique and revision cycle |
| [v5-multi-agent](v5-multi-agent/Readme.md) | Multi-Agent | Self-contained agent classes, Coordinator |
| [v6-autonomous](v6-autonomous/Readme.md) | Autonomous | Global critic, LLM-driven routing, selective re-execution |

---

## Suggested Learning Path

Run each version with the **same requirement** so you can compare how the output and the console behavior evolve:

```bash
python backlog_gen_v1.py "Grandma has a car and wants to know when she should refill fuel"
python backlog_gen_v2.py "Grandma has a car and wants to know when she should refill fuel"
python backlog_gen_v3.py "Grandma has a car and wants to know when she should refill fuel"
python backlog_gen_v4.py "Grandma has a car and wants to know when she should refill fuel"
python backlog_gen_v5.py "Grandma has a car and wants to know when she should refill fuel"
python backlog_gen_v6.py "Grandma has a car and wants to know when she should refill fuel"
```

---

## What to Observe at Each Version

**v1-pipeline**
The output appears immediately with no intermediate steps visible. Notice how bare it is - no planning, no checking, no structure.

**v2-shared-state**
The output looks the same as v1, but the code is fundamentally different. Study how `state` flows through the functions. This is the scaffolding everything else is built on.

**v3-planner-agent**
Watch for the `Planning workflow...` and `Confirmed Plan:` lines. The LLM is now deciding what to do before doing it.

**v4-critic-loop**
This is where the console gets interesting. Watch the `--- Critic Input ---` block to see exactly what the critic receives. Notice whether it approves immediately or requests revisions. Try a vague requirement to force revisions.

**v5-multi-agent**
Watch for the `[EpicAgent] Starting...` and `[FeaturesAgent] Starting...` labels. The agents are now self-contained units. Notice that the Planner is still driving the sequence - the Coordinator doesn't hardcode what to run.

**v6-autonomous**
Watch for the `[Global Critic] Verdict:` line after both agents complete. This is the system evaluating its own output holistically. If it returns `REDO: features`, watch the FeaturesAgent re-run while the Epic is preserved. This is selective re-execution in action.

---

## Key Agentic Patterns Covered

| Pattern | First appears in |
|---------|-----------------|
| LLM chaining | v1 |
| Shared state | v2 |
| LLM-driven control flow | v3 |
| Self-correction (critic loop) | v4 |
| Multi-agent encapsulation | v5 |
| Holistic quality review | v6 |
| LLM-driven routing | v6 |
| Selective re-execution | v6 |

---

## Real-World Parallels

| This project | Real-world equivalent |
|-------------|----------------------|
| Planner | Router / Supervisor in LangGraph |
| Shared state | AgentState in LangGraph |
| Critic loop | Reflection pattern in LangChain |
| Coordinator | Orchestrator in AutoGen / CrewAI |
| Global critic + routing | Conditional edges in LangGraph |

---

## Project Structure

```
README.md
v1-pipeline/
    backlog_gen_v1.py
    Readme.md
v2-shared-state/
    backlog_gen_v2.py
    Readme.md
v3=planner-agent/
    backlog_gen_v3.py
    Readme.md
v4-critic-loop/
    backlog_gen_v3.py
    Readme.md
v5-multi-agent
    backlog_gen_v3.py
    Readme.md
v6-autonomous
    backlog_gen_v3.py
    Readme.md
```