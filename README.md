# conversation-analyser

Critical-thinking and analytics for human–AI conversations — a member of the
[analyser family](https://github.com/michael-borck/lens-analysers).

It scores a single conversation on two tiers:

1. **Analytics** (always on, offline): turn/word counts, prompt/response lengths,
   question ratio, pushback hits, readability, sentiment trajectory, prompt
   self-similarity, and temporal metrics when timestamps are present.
2. **Critical thinking** (opt-in, needs an LLM): classifies every human turn under
   a 7-label prompt taxonomy, derives engagement ratios, an engagement **band**,
   and a composite **0–100 critical-thinking score** with a component breakdown.

The taxonomy reuses the validated `NQ/FU/CH/EX/DG/AC/MT` scheme from the ISYS6020
marking pipeline (copied and forked). Design: `docs/superpowers/specs/2026-05-23-conversation-analyser-design.md`.

## Install

```bash
pip install -e .                       # core: analytics + CLI + HTTP API
pip install -e '.[embeddings]'         # + prompt self-similarity (sentence-transformers)
pip install -e '.[llm]'                # + taxonomy/CT tier (anthropic)
pip install -e '.[embeddings,llm,dev]' # everything
export ANTHROPIC_API_KEY=...           # required for the critical-thinking tier
```

## CLI

Bare positional path to analyse (human summary by default, `--json` for machines);
`serve` subcommand for the HTTP API — same grammar as the rest of the family.

```bash
conversation-analyser transcript.txt              # human summary, analytics only
conversation-analyser chat.json --json            # full JSON to stdout
conversation-analyser chat.json --llm             # add the critical-thinking tier
conversation-analyser log.json --idle-gap 45      # split sub-sessions on 45-min gaps
conversation-analyser raw.txt --parse-mode llm-segment --llm
conversation-analyser serve --port 8009           # run the HTTP API
```

The critical-thinking tier is **opt-in** (`--llm`) to avoid surprise API costs;
without it you get the analytics tier only.

## HTTP API

```bash
conversation-analyser serve --port 8009
curl -F file=@chat.json 'http://127.0.0.1:8009/analyse'        # analytics only
curl -F file=@chat.json -F llm=true 'http://127.0.0.1:8009/analyse'
curl http://127.0.0.1:8009/health
```

`GET /health` and `POST /analyse` (multipart file upload, optional `llm` form
field) — the same `/analyse` contract auto-analyser routes to.

## Python API

```python
from conversation_analyser import ConversationAnalyser

result = ConversationAnalyser().analyse("transcript.txt", llm=True)
print(result.model_dump_json(indent=2))
```

## Input formats

A pluggable adapter registry tries, in order: structured adapters → heuristic
speaker markers → optional LLM segmentation → unsegmented fallback.

- **role/content** message list (OpenAI/Anthropic): `[{"role": "user", "content": "..."}, ...]`
- **AnythingLLM** rows: `[{"prompt": "...", "response": "...", "createdAt": ...}, ...]`
- **flat text** with speaker markers: `User:` / `Assistant:` / `Me:` / `ChatGPT:` /
  `You said:` / `ChatGPT said:` / `Prompt:` / `Response:`
- anything else → LLM-segment (needs `[llm]`), else a single-blob fallback

`.pdf`/`.docx` inputs are text-extracted first (needs `pdfplumber`/`markitdown`,
or pre-extract with `document-analyser`).

**Long unstructured transcripts** taking the LLM-segment path are split on
paragraph boundaries into chunks (`SEGMENT_CHUNK_CHARS`, default 36 000) and
classified chunk-by-chunk, so the whole transcript is labelled — not just its
opening — and the band/ratios/score reflect all of it. The number of chunks
(= LLM calls) is guarded by `SEGMENT_MAX_CHUNKS` (default 12); raise or lift it
per run, e.g. `CONVERSATION_ANALYSER_SEGMENT_MAX_CHUNKS=0` for unlimited. A
capped run says so in `notes` rather than silently dropping the tail.
Cleanly-labelled transcripts take the heuristic path and are never chunked.

## The taxonomy

| Code | Meaning |
|---|---|
| `NQ` | New Query — opens a new topic |
| `FU` | Follow-up — clarification/elaboration |
| `CH` | Challenge — pushes back, tests, asks why |
| `EX` | Extension — applies/compares/synthesises in a new direction |
| `DG` | Delegation — task hand-off, no engagement |
| `AC` | Acknowledgement — thanks/confirmation |
| `MT` | Meta — about the conversation itself |

`critical_thinking = (CH+EX)/turns`, `delegation = DG/turns`, `filler = (AC+MT)/turns`.
Bands: One-Shot · Delegator · Directed · Iterative · Critical.

## Graceful degradation

| Missing | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `[llm]` | `taxonomy`/`critical_thinking` null; analytics still produced; note `llm_unavailable` |
| `[embeddings]` | `prompt_self_similarity` null; note `embeddings_unavailable` |
| timestamps | temporal metrics omitted; no sub-session split; note `no timestamps` |

## Output

`ConversationAnalysis` → an `aggregate` (rolled up over all human turns, the
headline) plus one `SessionAnalysis` per idle-gap sub-session, each with
`analytics`, `taxonomy`, `critical_thinking`, and per-turn `turns` (label +
rationale + preview). See the design spec §8 for the full schema.

## Testing

```bash
pytest                    # fast, deterministic (LLM mocked, no network)
pytest -m slow            # includes sentence-transformers model download
pytest -m integration     # includes live LLM calls
```
