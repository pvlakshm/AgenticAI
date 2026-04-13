# Backlog Generator v1-Pipeline

This is the first iteration of the Backlog Generator. It demonstrates a basic **Linear Inference Chain** where a high-level requirement is transformed into an Epic, and that Epic is subsequently broken down to Features.

## Project Structure

* `backlog_gen_v1.py`: The core application logic, containing the LLM abstraction and the pipeline orchestration.
* `test_backlog_gen_v1.py`: The unit test suite using `pytest` and the **AAA (Arrange, Act, Assert)** pattern.
* `prompts/`: A directory containing `.prompt.md` files used to instruct the LLM.

## How it Works

The v1 pipeline operates as a "stateless" chain. Each step only knows about the output of the step immediately preceding it:

1.  **Requirement** -> `epic.prompt.md` -> **Epic Output**
2.  **Epic Output** -> `feature.prompt.md` -> **Feature Output**

> [!IMPORTANT]
> Because this version lacks a shared state, any global constraints provided in the initial requirement (e.g., "Generate exactly 3 features") are often lost by the time the pipeline reaches Step 2.

## Testing with Rigor

The test suite leverages `unittest.mock.Mock` to simulate LLM responses, ensuring tests are deterministic and do not require an active Ollama server.

### Key Testing Concepts Demonstrated:
* **AAA Pattern:** Every test is clearly divided into **Arrange** (setting up mocks/fixtures), **Act** (running the pipeline), and **Assert** (verifying the outcome).
* **Fixture Isolation:** We use two distinct fixtures (`standard_mock_llm` and `empty_mock_llm`) to separate "Happy Path" logic from boundary condition testing.
* **Orchestration Verification:** We don't just check the final text; we verify that the data correctly "chains" from the first LLM call to the second.

### Running the Tests
Ensure you have `pytest` installed, then run:
```bash
pytest test_backlog_gen_v1.py
```

## 📋 Learning Objectives for Students
1.  Understand how to abstract an LLM call using a class-based approach.
2.  Observe the "Context Loss" phenomenon in sequential pipelines.
3.  Learn to mock external dependencies (like an AI model) to create reliable unit tests.
4.  Identify why a shared `state` dictionary (to be introduced in v2) is necessary for maintaining project-wide constraints.

---

### Suggested Exercise
Run the pipeline with a global constraint:
`python backlog_gen_v1.py "Build a fuel tracker. Constraint: Generate exactly 3 features."`

**Observation:** Notice that the final Features output will likely ignore it.