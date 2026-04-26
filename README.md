# ProjectDev

Natural language assistant for controlling Houdini via safe structured actions.

ProjectDev turns natural language requests (e.g. *"create a sphere in Houdini"*)
into structured, validated actions that drive SideFX Houdini through its
Python API (`hou` / `hython`). The goal is to give artists and TDs a
conversational interface to scene authoring, while keeping every operation
auditable and reversible.

## Status

Early skeleton. The CLI currently only echoes the received command and loads
configuration. Houdini integration, LLM-backed planning, skills, and memory
are stubbed out as empty modules and will be implemented in later steps.

## Requirements

- Python 3.10+
- (Later) A local or remote LLM endpoint (LM Studio, OpenAI, or Anthropic)
- (Later) A Houdini installation with `hython` available

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill in the values you need:

```bash
cp .env.example .env
```

Available variables:

| Variable              | Purpose                                          |
| --------------------- | ------------------------------------------------ |
| `LMSTUDIO_BASE_URL`   | Base URL for a local LM Studio OpenAI-compatible server |
| `LMSTUDIO_MODEL`      | Model identifier served by LM Studio             |
| `OPENAI_API_KEY`      | API key for OpenAI-hosted models                 |
| `ANTHROPIC_API_KEY`   | API key for Anthropic-hosted models              |
| `HOUDINI_HYTHON_PATH` | Path to the `hython` executable                  |

## Usage

```bash
python -m app.main "create a sphere in Houdini"
```

For now this only prints the received command and the loaded configuration.

## Project layout

```
app/
  main.py        # CLI entry point
  config.py      # Environment-driven configuration
  llm/           # LLM clients and prompt orchestration (stub)
  houdini/       # Houdini bridge and action runners (stub)
  skills/        # High-level skills built on top of actions (stub)
  memory/        # Conversation and project memory (stub)
  ui/            # Future UI surfaces (stub)
tests/           # Unit tests
prompts/         # System / task prompt templates
examples/        # Example commands and scenes
```

## Development

Run the tests:

```bash
pytest
```
