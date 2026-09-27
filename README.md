## DOC-QA-RAG
This RAG ingests the insurance policy Q&A PDFs and filters irrelevant questions by thresholds followed by a second defense layer using prompts against hallucination.

When the user asks a question, the service draws on PDFs that were ingested at startup: section headers are marked, split on those headers with MarkdownNodeParser so each policy section becomes its own chunk, then embedded into ChromaDB. Then it uses similarity search to retrieve the top-K chunks. If the top chunk scores below 0.45 it rejects before the LLM. Claude then answers using only the retrieved chunks via a strict prompt, and declines if they do not contain the answer.

## Stack Highlights 
LlamaIndex, ChromaDB, local embeddings (BAAI/bge-small-en-v1.5, run on-device for zero embedding cost and document privacy), claude-sonnet-4-5-20250929, FastAPI, Docker.

Note: Anthropic deprecated temperature for newer models, and llama-index-llms-anthropic==0.11.3 sends it by default, causing /ask to return 500. Will bump once a patched release ships.

## Chunking Strategy
Initial chunking was size-based, which fused unrelated sections together. The renters
policy's "Deductibles" definition ended up merged into the Declarations boilerplate, so
its embedding was dominated by company name and policy number text. Queries like "what
does deductible mean?" could never retrieve it, even though the definition was in the corpus.

Semantic chunking (SemanticSplitterNodeParser) was tried next and did not help: at
breakpoint thresholds of both 95 and 85 the chunk count and boundaries were identical,
because to a small embedding model, adjacent policy clauses have no sharp meaning-jump
to split on.

The fix was to exploit the documents' own structure. A preprocessing step marks known
section headers ("Deductibles", "Premium Payment", etc.) with markdown syntax, then
MarkdownNodeParser splits on them. Chunk count went from 8 to 21, and each policy section
now stands as its own chunk. This is deliberately fitted to this corpus: the header list
is hand-maintained and would need a header-detection step to generalize to arbitrary policies.

## Architecture
```
   src/
   ├── config.py          # Threshold, top-k, model names, paths
   ├── prompts.py         # Strict system prompt with <sources> tags
   ├── logging_config.py  # JSONL query logger
   ├── rag.py             # Pipeline setup + ask() function
   └── main.py            # CLI entry point
```
## Two-Layer Defense
The first layer is the similarity threshold. It sets a threshold where the retrieval only returns the top chunk, if it scores higher than 0.45. Initially it was 0.50 as an intuition to get the RAG working. After running sweep threshold testings, I lowered it to 0.45 to allow borderline questions through. Blocking a question that the LLM could have answered is worse than letting a borderline question through. 

The second layer is prompt. It's a strict system prompt that forces Claude to answer only from the provided `<sources>` not from its own training data. If the sources do not contain the answer, it will say so instead of guessing and making up answers.

Example: Case #11 — "what is my life insurance policy number", 0.604 score
Layer 1 let the chunk pass because it scores above 0.45. The words "policy number"
and "insurance" appear throughout the auto and renters docs, so shared vocabulary
inflates the similarity score.
Layer 2 (the strict prompt) then declines in the answer text: the `<sources>` cover
auto and renters insurance only, never life insurance, so Claude states it has no
such information instead of inventing a policy number. Note this is a prompt-driven
decline surfaced in the response, not a hard threshold block.

## Setup
1. Clone: `git clone https://github.com/jhungai/doc-qa-rag` then `cd doc-qa-rag`
2. Venv: `python -m venv venv`
3. Activate: `venv\Scripts\Activate.ps1`
4. Install: `pip install -r requirements.txt`
5. Secret: put your API key in `.env` as `ANTHROPIC_API_KEY=your-key-here`. `.env` is listed in `.gitignore` so it is never committed.
6. Run: `python -m src.main`

- When running the repo for the first time, expect a small delay because the repo builds the index and downloads the embedding model.
- Go to (console.anthropic.com) to get the API key
- Activation is OS-specific. This repo is for Windows PowerShell (`venv\Scripts\Activate.ps1`). A Mac/Linux user needs `source venv/bin/activate`.

## Run as HTTP service

In addition to the CLI (`python -m src.main`), the RAG pipeline is exposed as a FastAPI service:

```bash
fastapi dev src/api.py
```

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

### Endpoints

| Method | Path     | Description                                  |
|--------|----------|----------------------------------------------|
| GET    | /health  | Health check, returns `{"status": "ok"}`     |
| POST   | /ask     | Submit a question, returns structured answer |

### Example request

```json
POST /ask
{"question": "what is my deductible"}
```

### Example response

```json
{
  "question": "what is my deductible",
  "answer": "Your deductible is...",
  "blocked": false,
  "top_score": 0.62,
  "sources": [...],
  "refusal_reason": null
}
```

When a query is blocked, `blocked: true` and `refusal_reason` indicates which defense layer triggered (`"threshold"` for similarity gate, `"prompt"` for Claude-driven refusal).

## Docker

1. Build the image: `docker build -t doc-qa-rag .`
   Reads the Dockerfile and produces a named image.

2. Run a container: `docker run -p 8000:8000 --env-file .env -v ${PWD}/chroma_db:/app/chroma_db doc-qa-rag`
   - `-p 8000:8000` exposes the API so you can reach it from a browser
   - `--env-file .env` loads the environment variables from the host's `.env` at runtime
   - `-v ${PWD}/chroma_db:/app/chroma_db` mounts a host folder so the index persists after the container stops

3. Secrets in `.env` are injected at runtime, never baked into the image, because image layers are inspectable and anyone who pulls the image could read a baked-in key.

## Evaluation
There are four categories that the repo tests for. The first one is `on-topic` which tests for questions that have a direct answer from the `<sources>`. Second one is `off-topic` which tests for questions that have nothing to do with the provided `<sources>`. Third is `adversarial` tests for questions that are similar in topic but ultimately have nothing to do with it. An example of this is case #3: "policy number for life insurance". The corpus covers auto and renters insurance, not life insurance. Last but not least, `corpus_gap` where it tags questions that are genuinely not covered in `<sources>`.

Current result: 10-11/11 passing, depending on the run. Retrieval is deterministic
(identical scores every run), but the `blocked` and `refusal_reason` flags are not.
The cause is `_detect_refusal`, which flags a refusal by substring-matching Claude's
answer against a fixed phrase list. Claude paraphrases between runs ("I don't have
information" vs "I don't have any information"), so the same refusal is sometimes
detected and sometimes missed. This is a known limitation of keyword-based refusal
detection against a generative model.

Cases 7 and 11 use an `"any"` sentinel on those two fields, because their answers are
genuinely absent from the corpus and any refusal is correct behavior regardless of which
layer catches it. Their assertions rest on the answer text instead.

Case 10 ("find whether i have permissive use in my auto policy") deliberately does NOT use
the sentinel and flakes red on some runs. Its answer IS in the auto policy, so a decline is
the pipeline being wrong, not an alternate valid path. The red is kept as an honest signal that refusal detection sometimes misfires on an answered query, flagging a real answer as a refusal.