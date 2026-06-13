# basic_usage

Minimal ways to run conversation-analyser.

## Install

```bash
pip install conversation-analyser
```

## CLI

```bash
conversation-analyser chat.json --json
```

Accepts `.json`, `.txt`, `.md`, `.pdf`. Without `--json` it prints a human-readable summary. Add `--llm` for the critical-thinking taxonomy tier (needs the `[llm]` extra and `ANTHROPIC_API_KEY`).

## Python

```python
from conversation_analyser import ConversationAnalyser

result = ConversationAnalyser().analyse("chat.json")
print(result.model_dump_json(indent=2))
```

## HTTP

```bash
conversation-analyser serve            # http://localhost:8009
curl -F file=@chat.json http://localhost:8009/analyse
```
