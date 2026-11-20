# ai-lab-code

Runnable code for the experiments in the **[AI Lab](https://iambishal.com/ai-lab)** on iambishal.com.

The AI Lab is where I test ideas about agents, MCP, RAG, evaluation and AI system architecture by measuring them, not just demoing them. Every experiment has one folder here, named after its lab page slug. Each folder is self-contained: its own `pyproject.toml`, its own tests, a README that explains how to reproduce it, and (once it has run) the raw results behind every number on the page.

## Experiments

| Experiment | Folder | Categories |
|---|---|---|
| [Tool Shape Benchmark: Endpoint Wrappers vs Task-Shaped MCP Tools](https://iambishal.com/ai-lab/projects/mcp-tool-shape-benchmark) | [`mcp-tool-shape-benchmark`](mcp-tool-shape-benchmark) | MCP · Evals · Agents |
| [Poisoned Ticket: Which Prompt-Injection Defences Actually Hold](https://iambishal.com/ai-lab/projects/prompt-injection-defence-benchmark) | [`prompt-injection-defence-benchmark`](prompt-injection-defence-benchmark) | AI Security · Agents · Evals |
| [Four Shapes, One Ticket: Agent Architecture Patterns, Measured](https://iambishal.com/ai-lab/projects/agent-architecture-patterns-benchmark) | [`agent-architecture-patterns-benchmark`](agent-architecture-patterns-benchmark) | AI Architecture · Agents · Evals |

## Use one folder

```bash
# Browse it on GitHub from the link on the lab page, or clone just that folder:
git clone --filter=blob:none --sparse https://github.com/bishal-saha/ai-lab-code
cd ai-lab-code
git sparse-checkout set <experiment-slug>
cd <experiment-slug>
uv sync && uv run pytest
```

No git? Every folder can be downloaded as a zip from the [Releases](https://github.com/bishal-saha/ai-lab-code/releases) page once it's tagged.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Tests never need a key: they run against offline fakes. API keys are needed only to re-run an experiment against a real model, where a folder's `.env.example` asks for them. Copy it to `.env`, which is never committed.

## Reproducing a result

Each experiment commits its raw results (`results/`) and the script that turns them into the charts and tables on its page, so every published number can be traced to the episodes behind it. Re-running against a live model costs money and gives slightly different numbers each time; the folder's README says what a full run costs and how to cap it.

## Problems

Open an issue and say which experiment it's about. Questions about the experiment itself: [iambishal.com/contact](https://iambishal.com/contact).

## Licence

Code and results: [MIT](LICENSE). The write-ups stay on iambishal.com.
