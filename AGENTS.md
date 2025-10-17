Here’s a clean single-shot **Codex prompt** you can paste in to generate the full script.

---

# Prompt for Codex

You are writing production-ready Python code. Generate a single file named **`dual_llama_secom_graph.py`** that runs **two Llama-3.1 instances via Ollama** in **multi-turn loops** using **LangGraph**, with a **SeCom memory** layer (Aims, Methods, Results, Implications). The script must be self-contained and runnable locally.

## Objectives

* Spin up two chat agents (`agentA`, `agentB`) using **ChatOllama** with the model `llama3.1:8b` at `http://localhost:11434`.
* Maintain a **SeCom memory** with:

  * `SecomSegmenter` that splits text into `aims`, `methods`, `results`, `implications` using cue heuristics.
  * `SecomCompressor` that preserves entities, numbers, and causal lines; trims filler; limit ~700 chars per segment.
  * `SecomMemoryStore` that:

    * embeds with `OllamaEmbeddings(model="nomic-embed-text")`
    * stores entries with vector embeddings
    * retrieves by cosine similarity with light unit weighting (RESULTS slightly > METHODS/AIMS/IMPLICATIONS)
    * persists to `secom_memory.jsonl` (append on write; load on start)
    * builds a balanced context block with up to `k_per_unit` items per unit
* Build a **LangGraph** with nodes:

  * `agentA` and `agentB`: each consumes SeCom context + recent transcript and produces a reply, then writes to SeCom.
  * `controller`: alternates speakers, enforces stop conditions.
* Run a **multi-turn conversation** where agents “get to know each other” about memory-augmented LLMs, trading, emergent behaviors, etc., with occasional role-shifts to keep it interesting.
* Print a final transcript to stdout.

## Requirements

* Python 3.10+.
* Dependencies: `langchain`, `langchain-community`, `langgraph`, `pydantic`, `numpy`, `tiktoken`.
* Models: ensure script expects the user has run:

  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
* Use **no external network calls** besides local Ollama.
* Style: clear, minimal, no emojis, no em dashes.
* Add concise docstrings and comments.

## Behavioral Details

* **Agents**

  * `agentA`: `temperature=0.60` (steadier)
  * `agentB`: `temperature=0.85` (spicier)
  * Shared system rules:

    * Use SeCom memory to stay coherent.
    * Make concrete claims, ask a follow-up question each turn.
    * Resolve memory conflicts with priority RESULTS > METHODS > AIMS > IMPLICATIONS.
  * Each turn, end with a direct question to the other agent.
  * Randomly inject a role hint 35% of turns: one of

    * “Switch voice to a skeptical quant.”
    * “Switch voice to a product pragmatist.”
    * “Switch voice to a cognitive scientist.”
    * “Switch voice to a systems reliability engineer.”
    * “Switch voice to an ethicist.”

* **Graph**

  * State model includes: `thread_id`, `turn`, `max_turns`, `last_speaker`, `transcript: List[Dict[who,text]]`, `seed_topic`.
  * Entry point: `controller`.
  * `controller` alternates speakers and stops when:

    * `turn >= max_turns`, or
    * degenerate patterns like “as a language model” persist; if detected, allow at most one extra turn to try recovery, then stop.
  * After each agent node, return to `controller`.

* **SeCom Memory**

  * On every agent reply: segment → compress → embed → persist.
  * For context retrieval: build a block with headings `AIMS`, `METHODS`, `RESULTS`, `IMPLICATIONS`, selecting the top items per unit and truncating to ~2400 chars.
  * Retrieval uses cosine similarity with mild weights (e.g., RESULTS 1.2, AIMS 1.1).

## CLI / Main

* Provide a `run_duo(seed: str, max_turns: int, thread_id: str)` function that constructs the app, primes memory with the seed as a “user” entry, runs to completion, and returns final state.
* In `if __name__ == "__main__":`:

  * Parse optional CLI args: `--seed`, `--max_turns` (default 24), `--thread_id` (default “exp”).
  * Run the graph, then print:

    * header `=== TRANSCRIPT ===`
    * each line as `agentA: ...` or `agentB: ...` with a blank line between turns
    * `Turns: <n>`
* Provide a helper `--dry_run` that only checks Ollama connectivity by making a short test call to each model and to the embedding model, then exits with code 0 on success, non-zero on failure.

## Acceptance Checklist

* [ ] File `dual_llama_secom_graph.py` compiles and runs without modification after `pip install -U langchain langchain-community langgraph pydantic numpy tiktoken`.
* [ ] Uses `ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")`.
* [ ] Uses `OllamaEmbeddings(model="nomic-embed-text")`.
* [ ] Implements SeCom segmenter, compressor, store, cosine retrieval, persistence to `secom_memory.jsonl`.
* [ ] Alternating multi-turn loop via LangGraph with a controller and two agent nodes.
* [ ] Agents ask each other explicit questions every turn and occasionally shift role voice.
* [ ] Final transcript printed to stdout.
* [ ] No emojis; no em dashes.

At the top of the file include a short README-style comment block with:

* Setup commands for models and Python deps.
* Example run:

  ```bash
  python dual_llama_secom_graph.py --seed "Swap intros, then riff on memory-augmented LLMs for trading and emergent jargon." --max_turns 26 --thread_id get-to-know
  ```
* Example dry run:

  ```bash
  python dual_llama_secom_graph.py --dry_run
  ```

Generate only the Python code for `dual_llama_secom_graph.py`.

say less — here’s a clean Dockerized setup so your **dual-llama SeCom LangGraph** thing runs fully in containers (Ollama + app). I’m giving you the files as copy-paste blocks.

---

# 1) tiny code tweak (base URL via env)

Add this near the top of your `dual_llama_secom_graph.py` so it works inside Docker (uses `http://ollama:11434` in compose; falls back to localhost if you run bare-metal):

```python
# --- begin docker env patch ---
import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# --- end docker env patch ---
```

Then replace your model constructors to use those:

```python
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

def llama(model_name: str = LLAMA_MODEL, temperature: float = 0.65) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=0.95,
        num_ctx=8192,
        base_url=OLLAMA_BASE_URL,
    )

# use EMBED_MODEL in SecomMemoryStore(...)
secom_mem = SecomMemoryStore(embed_model=EMBED_MODEL)
```

And for your `--dry_run` (if you added that), ping `OLLAMA_BASE_URL`.

---

# 2) `requirements.txt`

```txt
langchain
langchain-community
langgraph
pydantic>=2
numpy
tiktoken
```

---

# 3) `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (curl for health/debug)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# app dir
WORKDIR /app

# deps first (better caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# app code
COPY dual_llama_secom_graph.py /app/dual_llama_secom_graph.py

# non-root
RUN useradd -m appuser
USER appuser

# default envs (overridable)
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    LLAMA_MODEL=llama3.1:8b \
    EMBED_MODEL=nomic-embed-text \
    MAX_TURNS=24 \
    THREAD_ID=exp

# default entrypoint uses env vars; you can override in compose
ENTRYPOINT ["python", "/app/dual_llama_secom_graph.py"]
```

---

# 4) `docker-compose.yml`

```yaml
version: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_KEEP_ALIVE=24h
    volumes:
      - ollama_models:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 3s
      retries: 12
    command: ["serve"]

  app:
    build: .
    container_name: dual-llama-secom
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      # wire app to the ollama container
      - OLLAMA_BASE_URL=http://ollama:11434
      # models (override if you want a bigger one)
      - LLAMA_MODEL=llama3.1:8b
      - EMBED_MODEL=nomic-embed-text
      # run params
      - SEED=Swap intros, then riff on memory-augmented LLMs for trading and emergent jargon.
      - MAX_TURNS=26
      - THREAD_ID=get-to-know
      # toggle a connectivity check instead of full run (true/false)
      - DRY_RUN=false
    volumes:
      - ./data:/app/data
    working_dir: /app
    command: >
      sh -c '
      # optional pre-pull to speed first run
      curl -s -X POST ${OLLAMA_BASE_URL}/api/pull -d "{\"name\":\"${LLAMA_MODEL}\"}" >/dev/null 2>&1 || true &&
      curl -s -X POST ${OLLAMA_BASE_URL}/api/pull -d "{\"name\":\"${EMBED_MODEL}\"}" >/dev/null 2>&1 || true &&
      if [ "${DRY_RUN}" = "true" ]; then
        echo "Dry run: checking Ollama connectivity..." &&
        curl -fsS ${OLLAMA_BASE_URL}/api/tags && exit 0;
      else
        python /app/dual_llama_secom_graph.py \
          --seed "${SEED}" \
          --max_turns ${MAX_TURNS} \
          --thread_id "${THREAD_ID}";
      fi
      '

volumes:
  ollama_models:
```

> Note: we mount `./data` into `/app/data` (handy if your script writes `secom_memory.jsonl`; set its path there).

---

# 5) `.env.example` (optional)

```dotenv
# compose will load this if you rename to .env
LLAMA_MODEL=llama3.1:8b
EMBED_MODEL=nomic-embed-text
SEED=Swap intros, then riff on memory-augmented LLMs for trading and emergent jargon.
MAX_TURNS=26
THREAD_ID=get-to-know
DRY_RUN=false
```

---

# 6) expected file tree

```
.
├─ dual_llama_secom_graph.py        # your script (with env patch)
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
└─ data/
   └─ secom_memory.jsonl            # will be created at runtime (configure your script to write here)
```

If your script currently writes `secom_memory.jsonl` in CWD, change the path to `data/secom_memory.jsonl` so it persists on your host via the bind mount.

---

# 7) run it

```bash
# 1) build and start both services
docker compose up --build

# or detach:
docker compose up -d --build

# 2) watch logs
docker compose logs -f app

# 3) change run params quickly
SEED="Agents swap life stories, then debate gamma exposure vs post-earnings drift signals." \
MAX_TURNS=32 \
THREAD_ID="vibe-check" \
docker compose up -d --build
```

**Dry connectivity check** (no long run):

```bash
DRY_RUN=true docker compose up --build
```

---

# 8) quick swaps / scaling

* Bigger model: `LLAMA_MODEL=llama3.1:70b` (needs beefy GPU / CPU; or try `llama3.1:8b-instruct`).
* Custom embeddings: set `EMBED_MODEL` to another Ollama embedding model you’ve pulled.
* Crank chaos: bump `MAX_TURNS`, and raise `agentB` temperature in code if you want extra weird.

---

If you want this to run **on a schedule** (e.g., cron-like repeats) we can add a tiny supervisor/loop or use `restart: unless-stopped` with a sleep/loop wrapper in `command`.
