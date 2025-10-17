"""
Dual Llama SeCom Graph Runner
==============================

Model setup:
    ollama pull ${LLAMA_MODEL:-llama3.1:8b}
    ollama pull ${EMBED_MODEL:-nomic-embed-text}

Python dependencies:
    pip install -U langchain langchain-community langgraph pydantic numpy tiktoken

Example run:
    python dual_llama_secom_graph.py --seed "Swap intros, then riff on memory-augmented LLMs for trading and emergent jargon." --max_turns 26 --thread_id get-to-know

Dry run connectivity check:
    python dual_llama_secom_graph.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import END, StateGraph


AIMS = "AIMS"
METHODS = "METHODS"
RESULTS = "RESULTS"
IMPLICATIONS = "IMPLICATIONS"
SECOM_UNITS = [AIMS, METHODS, RESULTS, IMPLICATIONS]
ROLE_HINTS = [
    "Switch voice to a skeptical quant.",
    "Switch voice to a product pragmatist.",
    "Switch voice to a cognitive scientist.",
    "Switch voice to a systems reliability engineer.",
    "Switch voice to an ethicist.",
]
DEFAULT_LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.1:8b")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DATA_PATH = Path("data")
MEMORY_PATH = DATA_PATH / "secom_memory.jsonl"
ROLE_HINT_RATE = 0.35


def ensure_data_path() -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.touch()


@dataclass
class MemoryEntry:
    unit: str
    text: str
    embedding: List[float]
    speaker: str
    turn: int
    thread_id: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "text": self.text,
            "embedding": self.embedding,
            "speaker": self.speaker,
            "turn": self.turn,
            "thread_id": self.thread_id,
        }


class SecomSegmenter:
    """Heuristic segmenter that maps text to SeCom units."""

    def __init__(self) -> None:
        self.cues = {
            AIMS: ["goal", "aim", "intent", "purpose", "objective", "seeking"],
            METHODS: [
                "method",
                "process",
                "approach",
                "technique",
                "we do",
                "we use",
                "workflow",
            ],
            RESULTS: ["result", "found", "learned", "observed", "saw", "outcome"],
            IMPLICATIONS: ["impact", "mean", "implies", "therefore", "future", "next"],
        }

    def _score_sentence(self, sentence: str) -> str:
        lowered = sentence.lower()
        for unit, cues in self.cues.items():
            if any(token in lowered for token in cues):
                return unit
        if any(char.isdigit() for char in sentence):
            return RESULTS
        return RESULTS if "because" in lowered else AIMS

    def segment(self, text: str) -> Dict[str, List[str]]:
        segments: Dict[str, List[str]] = {unit: [] for unit in SECOM_UNITS}
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        if not sentences:
            return segments
        for sentence in sentences:
            unit = self._score_sentence(sentence)
            segments[unit].append(sentence)
        return segments


class SecomCompressor:
    """Compresses SeCom segments while preserving salient details."""

    def __init__(self, max_chars: int = 700) -> None:
        self.max_chars = max_chars

    def _prioritize(self, sentences: List[str]) -> List[str]:
        scored = []
        for sentence in sentences:
            weight = 1.0
            if any(char.isdigit() for char in sentence):
                weight += 0.4
            if "because" in sentence.lower() or "so that" in sentence.lower():
                weight += 0.3
            if any(token in sentence.lower() for token in ["increase", "decrease", "trade", "signal", "memory"]):
                weight += 0.2
            scored.append((weight, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    def compress(self, sentences: Iterable[str]) -> str:
        ordered = self._prioritize(list(sentences))
        excerpt: List[str] = []
        total = 0
        for sentence in ordered:
            if total + len(sentence) + 1 > self.max_chars:
                continue
            excerpt.append(sentence)
            total += len(sentence) + 1
            if total >= self.max_chars:
                break
        return ". ".join(excerpt)


class SecomMemoryStore:
    """Persists and retrieves SeCom memory slices with embeddings."""

    def __init__(self, path: Path, embed_model: str, weights: Optional[Dict[str, float]] = None) -> None:
        ensure_data_path()
        self.path = path
        self.embedder = OllamaEmbeddings(model=embed_model, base_url=DEFAULT_BASE_URL)
        self.weights = weights or {AIMS: 1.05, METHODS: 1.0, RESULTS: 1.2, IMPLICATIONS: 0.95}
        self.entries: List[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    entry = MemoryEntry(
                        unit=payload["unit"],
                        text=payload["text"],
                        embedding=payload["embedding"],
                        speaker=payload.get("speaker", "unknown"),
                        turn=payload.get("turn", -1),
                        thread_id=payload.get("thread_id", ""),
                    )
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    continue

    def add_entry(self, unit: str, text: str, speaker: str, turn: int, thread_id: str) -> None:
        if not text:
            return
        vector = self.embedder.embed_query(text)
        entry = MemoryEntry(unit=unit, text=text, embedding=vector, speaker=speaker, turn=turn, thread_id=thread_id)
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        a_vec = np.array(a)
        b_vec = np.array(b)
        denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
        if denom == 0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)

    def search(self, query: str, k_per_unit: int = 2) -> Dict[str, List[Tuple[MemoryEntry, float]]]:
        if not self.entries:
            return {unit: [] for unit in SECOM_UNITS}
        query_vec = self.embedder.embed_query(query or "context")
        scored: Dict[str, List[Tuple[MemoryEntry, float]]] = {unit: [] for unit in SECOM_UNITS}
        for entry in self.entries:
            score = self._cosine(query_vec, entry.embedding) * self.weights.get(entry.unit, 1.0)
            scored[entry.unit].append((entry, score))
        for unit, items in scored.items():
            items.sort(key=lambda pair: pair[1], reverse=True)
            scored[unit] = items[:k_per_unit]
        return scored

    def build_context(self, query: str, k_per_unit: int = 2, max_chars: int = 2400) -> str:
        scored = self.search(query, k_per_unit=k_per_unit)
        blocks: List[str] = []
        total_chars = 0
        for unit in SECOM_UNITS:
            items = scored.get(unit, [])
            if not items:
                continue
            unit_lines = [f"# {unit}"]
            for entry, score in items:
                unit_lines.append(f"- ({score:.3f}) {entry.text}")
            block = "\n".join(unit_lines)
            if total_chars + len(block) > max_chars:
                break
            blocks.append(block)
            total_chars += len(block)
        return "\n\n".join(blocks)


def make_system_prompt(agent_name: str) -> str:
    rules = [
        "Use SeCom memory to stay coherent.",
        "Make concrete claims and always end with a direct question to the other agent.",
        "Resolve memory conflicts with priority RESULTS > METHODS > AIMS > IMPLICATIONS.",
    ]
    return (
        f"You are {agent_name}. "
        + " ".join(rules)
        + " Keep the conversation about memory-augmented LLMs, trading signals, and emergent behavior."
    )


def format_transcript(transcript: List[Dict[str, str]]) -> str:
    lines: List[str] = ["=== TRANSCRIPT ==="]
    for entry in transcript:
        if entry["speaker"] not in {"agentA", "agentB"}:
            continue
        lines.append(f"{entry['speaker']}: {entry['text']}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def build_agent_node(
    agent_name: str,
    llm: ChatOllama,
    memory_store: SecomMemoryStore,
    segmenter: SecomSegmenter,
    compressor: SecomCompressor,
) -> Any:
    system_prompt = make_system_prompt(agent_name)

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        transcript: List[Dict[str, str]] = state["transcript"]
        last_message = transcript[-1]["text"] if transcript else state["seed_topic"]
        context_block = memory_store.build_context(last_message, k_per_unit=2)
        recent = transcript[-4:]
        dialogue_lines = [f"{item['speaker']}: {item['text']}" for item in recent if item["speaker"] in {"agentA", "agentB"}]
        convo_history = "\n".join(dialogue_lines)
        role_hint = None
        if state.get("role_hint_target") == agent_name:
            role_hint = state.get("role_hint_text")
        prompt_parts = []
        if context_block:
            prompt_parts.append("SeCom context:\n" + context_block)
        if convo_history:
            prompt_parts.append("Recent transcript:\n" + convo_history)
        prompt_parts.append(f"Latest message you are responding to:\n{last_message}")
        if role_hint:
            prompt_parts.append(f"Role hint: {role_hint}")
        prompt_parts.append("Respond succinctly (<= 180 words) and end with a direct question.")
        content = "\n\n".join(prompt_parts)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        response = llm.invoke(messages)
        reply_text = response.content.strip()
        if not reply_text.endswith("?"):
            reply_text = reply_text.rstrip(".") + "?"
        state["transcript"].append({"speaker": agent_name, "text": reply_text})
        segments = segmenter.segment(reply_text)
        for unit, sentences in segments.items():
            compressed = compressor.compress(sentences)
            memory_store.add_entry(unit, compressed, agent_name, state["turn"] + 1, state["thread_id"])
        lowered = reply_text.lower()
        if "as a language model" in lowered:
            state["degeneracy_streak"] = state.get("degeneracy_streak", 0) + 1
        else:
            state["degeneracy_streak"] = 0
        state["turn"] += 1
        state["last_speaker"] = agent_name
        state["role_hint_target"] = None
        state["role_hint_text"] = None
        return state

    return node


def build_controller_node(max_turns: int) -> Tuple[Any, Any]:
    def controller(state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("degeneracy_streak", 0)
        state.setdefault("last_speaker", None)
        state.setdefault("turn", 0)
        state.setdefault("role_hint_target", None)
        state.setdefault("role_hint_text", None)
        if random.random() < ROLE_HINT_RATE:
            next_hint = random.choice(ROLE_HINTS)
            next_speaker = "agentA" if state.get("last_speaker") == "agentB" else "agentB"
            state["role_hint_target"] = next_speaker
            state["role_hint_text"] = next_hint
        return state

    def router(state: Dict[str, Any]) -> str:
        if state.get("turn", 0) >= state.get("max_turns", max_turns):
            return "end"
        if state.get("degeneracy_streak", 0) >= 2:
            return "end"
        last_speaker = state.get("last_speaker")
        if last_speaker == "agentA":
            return "agentB"
        return "agentA"

    return controller, router


def build_graph(
    memory_store: SecomMemoryStore,
    segmenter: SecomSegmenter,
    compressor: SecomCompressor,
    agentA_llm: ChatOllama,
    agentB_llm: ChatOllama,
    max_turns: int,
) -> Any:
    graph = StateGraph(dict)
    agentA_node = build_agent_node("agentA", agentA_llm, memory_store, segmenter, compressor)
    agentB_node = build_agent_node("agentB", agentB_llm, memory_store, segmenter, compressor)
    controller_node, router = build_controller_node(max_turns)
    graph.add_node("controller", controller_node)
    graph.add_node("agentA", agentA_node)
    graph.add_node("agentB", agentB_node)
    graph.set_entry_point("controller")
    graph.add_conditional_edges("controller", router, {"agentA": "agentA", "agentB": "agentB", "end": END})
    graph.add_edge("agentA", "controller")
    graph.add_edge("agentB", "controller")
    return graph.compile()


def prime_memory(
    seed: str,
    memory_store: SecomMemoryStore,
    segmenter: SecomSegmenter,
    compressor: SecomCompressor,
    thread_id: str,
) -> None:
    segments = segmenter.segment(seed)
    for unit, sentences in segments.items():
        compressed = compressor.compress(sentences)
        memory_store.add_entry(unit, compressed, "seed", 0, thread_id)


def run_duo(seed: str, max_turns: int, thread_id: str) -> Dict[str, Any]:
    ensure_data_path()
    segmenter = SecomSegmenter()
    compressor = SecomCompressor()
    memory_store = SecomMemoryStore(MEMORY_PATH, DEFAULT_EMBED_MODEL)
    prime_memory(seed, memory_store, segmenter, compressor, thread_id)
    agentA_llm = ChatOllama(model=DEFAULT_LLAMA_MODEL, base_url=DEFAULT_BASE_URL, temperature=0.60)
    agentB_llm = ChatOllama(model=DEFAULT_LLAMA_MODEL, base_url=DEFAULT_BASE_URL, temperature=0.85)
    graph = build_graph(memory_store, segmenter, compressor, agentA_llm, agentB_llm, max_turns)
    initial_state: Dict[str, Any] = {
        "thread_id": thread_id,
        "turn": 0,
        "max_turns": max_turns,
        "last_speaker": "agentB",
        "transcript": [{"speaker": "seed", "text": seed}],
        "seed_topic": seed,
        "degeneracy_streak": 0,
        "role_hint_target": None,
        "role_hint_text": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state


def check_connectivity() -> bool:
    try:
        test_llm = ChatOllama(model=DEFAULT_LLAMA_MODEL, base_url=DEFAULT_BASE_URL, temperature=0.1)
        _ = test_llm.invoke([HumanMessage(content="Say ready in one word.")])
        embedder = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_BASE_URL)
        _ = embedder.embed_query("ping")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Connectivity check failed: {exc}", file=sys.stderr)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual Llama agents with SeCom memory via LangGraph.")
    parser.add_argument("--seed", type=str, default="Swap intros, then riff on memory-augmented LLMs.", help="Seed topic for the conversation.")
    parser.add_argument("--max_turns", type=int, default=24, help="Maximum number of agent turns.")
    parser.add_argument("--thread_id", type=str, default="exp", help="Thread identifier for memory records.")
    parser.add_argument("--dry_run", action="store_true", help="Only verify connectivity to Ollama models and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        success = check_connectivity()
        sys.exit(0 if success else 1)
    final_state = run_duo(args.seed, args.max_turns, args.thread_id)
    transcript = final_state.get("transcript", [])
    print(format_transcript(transcript))
    print(f"Turns: {final_state.get('turn', 0)}")


if __name__ == "__main__":
    main()
