import json
import os
import pickle
import csv
import re
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import altair as alt
import faiss
import google.generativeai as genai
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "rag_index"
TESTS_PATH = BASE_DIR / "A1_Hamad _testcases.json"
RATINGS_OUTPUT_PATH = BASE_DIR / "evaluation_ratings.csv"
PROMPT_ONLY_SCORES_PATH = BASE_DIR / "evaluation_outputs_to_score.csv"
FIGURES_DIR = BASE_DIR / "evaluation_figures"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "models/gemini-flash-lite-latest"
DEFAULT_TOP_K = 3


# The system prompt was designed to ensure that the model generates
# safe, structured, and evidence-grounded responses using only the
# retrieved drug documents. It also instructs the model to avoid
# unsupported claims, prescribing behavior, or dosage recommendations,
# and to clearly express uncertainty when the retrieved context is insufficient.

SYSTEM_PROMPT = """
You are a medication interaction assistant.

Use only the retrieved context to answer.
If the retrieved context is insufficient, say:
"I do not have enough information in the retrieved documents to answer confidently."

Rules:
- Do not prescribe medicines.
- Do not give dosage advice.
- Be cautious and safe.
- Answer in this format:

1. Retrieved Medicines / Ingredients
2. Interaction Decision (Safe / Caution / Uncertain / Not Safe)
3. Explanation
4. Safety Advice
5. Sources

For sources, cite them like [1], [2].
"""

# These examples are shown to the user as default examples 
EXAMPLE_QUERIES = [
    "Can I take Panadol with warfarin?",
    "Is ibuprofen safe with aspirin?",
    "Can metformin interact with alcohol?",
]

# The rating dimensions include the metrics used in Assignment 1
# (accuracy, clarity, and safety) in addition to three RAG-specific
# evaluation metrics: groundedness, citation accuracy, and retrieval
# relevance. These metrics help assess both the quality of the model's
# responses and how well the answers are supported by the retrieved
# documents.
RATING_DIMENSIONS = [
    "accuracy",
    "clarity",
    "safety",
    "groundedness",
    "citation_accuracy",
    "retrieval_relevance",
]


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load the sentence-transformer once and reuse it across reruns."""
    return SentenceTransformer(EMBED_MODEL)


def load_docs(path: Path) -> List[Dict]:
    """Load the raw medication documents from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def simple_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split long source text into overlapping chunks for retrieval."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    """Normalize source documents into chunk records used by the app."""
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(simple_chunk(doc["document"])):
            chunks.append(
                {
                    "chunk_id": f'{doc["id"]}_chunk{i}',
                    "doc_id": doc["id"],
                    "title": doc["medicine_name"],
                    "source": doc["source_website"],
                    "url": doc.get("source_website", ""),
                    "text": chunk,
                }
            )
    return chunks



def build_bm25_index(chunks: List[Dict]) -> Dict:
    """Precompute BM25 statistics so lexical retrieval is fast at runtime."""
    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    doc_freqs = []
    doc_lengths = []
    document_frequency = Counter()

    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        doc_freqs.append(dict(frequencies))
        doc_lengths.append(len(tokens))
        for token in frequencies:
            document_frequency[token] += 1

    total_docs = len(tokenized_docs)
    avgdl = sum(doc_lengths) / total_docs if total_docs else 0.0
    idf = {
        token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in document_frequency.items()
    }

    return {
        "tokenized_docs": tokenized_docs,
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "k1": 1.5,
        "b": 0.75,
    }


def build_and_save_index(index_path: Path, chunks_path: Path, bm25_path: Path):
    """Rebuild the full hybrid index when deployed artifacts are missing."""
    docs_path = BASE_DIR / "drug_docs.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            "Missing both the RAG index files and the source document file "
            "'drug_docs.json', so the index cannot be rebuilt."
        )

    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = load_docs(docs_path)
    chunks = build_chunks(docs)

    embedder = load_embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dense_index = faiss.IndexFlatIP(embeddings.shape[1])
    dense_index.add(embeddings.astype(np.float32))
    bm25_index = build_bm25_index(chunks)

    faiss.write_index(dense_index, str(index_path))
    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)
    with open(bm25_path, "wb") as file:
        pickle.dump(bm25_index, file)

    return dense_index, chunks, bm25_index


@st.cache_resource
def load_index():
    """Load the dense index, chunk metadata, and BM25 statistics."""
    index_path = INDEX_DIR / "docs.index"
    chunks_path = INDEX_DIR / "chunks.pkl"
    bm25_path = INDEX_DIR / "bm25.pkl"

    if not index_path.exists() or not chunks_path.exists() or not bm25_path.exists():
        return build_and_save_index(index_path, chunks_path, bm25_path)

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)
    with open(bm25_path, "rb") as file:
        bm25_index = pickle.load(file)
    return index, chunks, bm25_index


@st.cache_data
def load_test_cases() -> List[Dict]:
    """Read evaluation prompts from the configured test-case file."""
    if not TESTS_PATH.exists():
        return []

    with open(TESTS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("test_cases", [])


def tokenize(text: str) -> List[str]:
    """Tokenize text for lightweight lexical matching."""
    return re.findall(r"\b\w+\b", text.lower())


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Scale scores into a comparable 0-1 range before fusion."""
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-9:
        return np.ones_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def bm25_scores(query: str, bm25_index: Dict) -> np.ndarray:
    """Score every chunk lexically using the precomputed BM25 statistics."""
    tokens = tokenize(query)
    if not tokens:
        return np.zeros(len(bm25_index["doc_freqs"]), dtype=np.float32)

    scores = np.zeros(len(bm25_index["doc_freqs"]), dtype=np.float32)
    avgdl = max(float(bm25_index["avgdl"]), 1e-9)
    k1 = float(bm25_index["k1"])
    b = float(bm25_index["b"])
    idf = bm25_index["idf"]

    for idx, frequencies in enumerate(bm25_index["doc_freqs"]):
        doc_length = bm25_index["doc_lengths"][idx]
        score = 0.0
        for token in tokens:
            term_frequency = frequencies.get(token, 0)
            if term_frequency == 0:
                continue
            token_idf = idf.get(token, 0.0)
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (
                1 - b + b * (doc_length / avgdl)
            )
            score += token_idf * (numerator / denominator)
        scores[idx] = score

    return scores


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
    """Rank chunks with hybrid retrieval that blends dense and BM25 signals."""
    embedder = load_embedder()
    dense_index, chunks, bm25_index = load_index()

    query_vector = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    dense_scores, _ = dense_index.search(
        query_vector.astype(np.float32), len(chunks)
    )
    dense_scores = dense_scores[0]
    lexical_scores = bm25_scores(query, bm25_index)

    # Dense similarity captures semantic matches; BM25 helps with exact terms.
    normalized_dense = min_max_normalize(dense_scores)
    normalized_lexical = min_max_normalize(lexical_scores)
    hybrid_scores = 0.65 * normalized_dense + 0.35 * normalized_lexical
    ranked_ids = np.argsort(hybrid_scores)[::-1][:k]

    results = []
    seen = set()
    for idx in ranked_ids:
        item = chunks[idx]
        key = item["chunk_id"]
        if key in seen:
            continue

        seen.add(key)
        item = dict(item)
        item["score"] = float(hybrid_scores[idx])
        item["dense_score"] = float(dense_scores[idx])
        item["bm25_score"] = float(lexical_scores[idx])
        results.append(item)

    return results


def build_context(results: List[Dict]) -> str:
    """Format retrieved chunks into the grounded context sent to Gemini."""
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[{i}] Title: {result['title']}\n"
            f"Source: {result['source']}\n"
            f"Text: {result['text']}"
        )
    return "\n\n".join(blocks)


def get_google_api_key() -> str:
    """Prefer Streamlit secrets, with env vars as a local fallback."""
    return st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))


def ask_gemini(system_prompt: str, user_query: str, context: str) -> str:
    """Generate a grounded answer using only the retrieved evidence."""
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. Add it to Streamlit secrets before "
            "publishing the app."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEN_MODEL)

    full_prompt = f"""
{system_prompt}

User Query:
{user_query}

Retrieved Context:
{context}
"""
    response = model.generate_content(
        full_prompt,
        generation_config={"temperature": 0.2},
    )
    return response.text.strip()


def run_rag_pipeline(query: str) -> Dict:
    """Run retrieval first, then pass the grounded context to the generator."""
    results = retrieve(query, k=DEFAULT_TOP_K)
    if not results:
        raise ValueError(
            "No supporting documents were retrieved for this question. "
            "Please try a different medication query."
        )

    context = build_context(results)
    answer = ask_gemini(SYSTEM_PROMPT, query, context)
    return {"query": query, "answer": answer, "results": results}


def score_label(score: float) -> str:
    """Convert a numeric retrieval score into a friendlier UI label."""
    if score >= 0.8:
        return "Strong match"
    if score >= 0.6:
        return "Relevant"
    return "Possible match"


def score_percent(score: float) -> int:
    """Convert a score into a bounded progress-bar percentage."""
    return max(0, min(100, int(score * 100)))


def distinct_cited_sources(answer: str, results: List[Dict]) -> List[Dict]:
    """Keep only distinct retrieved sources that are cited in the answer."""
    cited_indices = []
    for match in re.findall(r"\[(\d+)\]", answer):
        idx = int(match) - 1
        if 0 <= idx < len(results) and idx not in cited_indices:
            cited_indices.append(idx)

    filtered_results = [results[idx] for idx in cited_indices] if cited_indices else results

    distinct_results = []     
    seen_sources = set()
    for result in filtered_results:
        source_key = (
            result.get("title", "").strip().lower(),
            result.get("source", "").strip().lower(),
            result.get("url", "").strip().lower(),
        )
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        distinct_results.append(result)

    return distinct_results


def render_main_source(result: Dict) -> None:
    """Show the primary cited source."""
    st.subheader("Main Source")
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-title">{result['title']}</div>
            <div class="source-meta">
                {score_label(result['score'])} | score {result['score']:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score_percent(result["score"]))
    st.caption(f"Source: {result['source']}")


def render_other_sources(results: List[Dict]) -> None:
    """Show the remaining distinct cited sources underneath the main source."""
    if len(results) <= 1:
        return

    st.subheader("Other Sources")
    for result in results[1:]:
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{result['title']}</div>
                <div class="source-meta">
                    {score_label(result['score'])} | score {result['score']:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(score_percent(result["score"]))
        st.caption(f"Source: {result['source']}")


def is_case_fully_rated(case_id: str, ratings: Dict) -> bool:
    """Return True only when every evaluation dimension has a score."""
    case_ratings = ratings.get(case_id, {})
    return all(dimension in case_ratings for dimension in RATING_DIMENSIONS)


def case_label(case: Dict, ratings: Dict) -> str:
    """Display a case label with a completion marker in the selector."""
    rated_suffix = " [Done]" if is_case_fully_rated(case["id"], ratings) else ""
    return f"{case['id']} | {case['category']} | {case['input']}{rated_suffix}"


def save_ratings_to_csv(test_cases: List[Dict], ratings: Dict) -> Path:
    """Persist the current evaluation state so progress is not lost."""
    fieldnames = ["id", "category", "query", "notes"] + RATING_DIMENSIONS

    with open(RATINGS_OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for case in test_cases:
            case_ratings = ratings.get(case["id"], {})
            if not case_ratings:
                continue

            row = {
                "id": case["id"],
                "category": case.get("category", ""),
                "query": case.get("input", ""),
                "notes": case.get("notes", ""),
            }
            for dimension in RATING_DIMENSIONS:
                row[dimension] = case_ratings.get(dimension, "")
            writer.writerow(row)

    return RATINGS_OUTPUT_PATH


# Comparing the RAG system with the baseline using the stronger model evaluated in assignment 1 - model B
def load_prompt_only_model_b_scores() -> List[Dict]:
    """Load prompt-only baseline scores, keeping only Model B columns."""
    if not PROMPT_ONLY_SCORES_PATH.exists():
        return []

    rows = []
    with open(PROMPT_ONLY_SCORES_PATH, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed_row = {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
            }
            for dimension in RATING_DIMENSIONS:
                csv_key = f"{dimension}_B"
                if csv_key in row and row[csv_key] not in ("", None):
                    parsed_row[dimension] = float(row[csv_key])
            rows.append(parsed_row)
    return rows


def average_metric_scores(rows: List[Dict], metrics: List[str]) -> Dict[str, float]:
    """Compute simple mean scores for a chosen list of metrics."""
    if not rows:
        return {}

    return {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in metrics
    }


def shared_prompt_metrics(prompt_rows: List[Dict]) -> List[str]:
    """Find which rated metrics exist in the external prompt-only CSV."""
    if not prompt_rows:
        return []

    return [
        dimension
        for dimension in RATING_DIMENSIONS
        if all(dimension in row for row in prompt_rows)
    ]


def rag_specific_metrics() -> List[str]:
    """Return metrics that are specific to evaluating the RAG pipeline."""
    return ["groundedness", "citation_accuracy", "retrieval_relevance"]


def save_chart_files(chart_spec: Dict, stem: str) -> Dict[str, Path]:
    """Persist a chart as both Vega-Lite JSON and HTML."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    json_path = FIGURES_DIR / f"{stem}.json"
    html_path = FIGURES_DIR / f"{stem}.html"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(chart_spec, file, ensure_ascii=True, indent=2)

    alt.Chart.from_dict(chart_spec).save(str(html_path))
    return {"json": json_path, "html": html_path}


st.set_page_config(
    page_title="Medication Interaction RAG Assistant",
    page_icon="M",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(16, 185, 129, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 24%),
            linear-gradient(180deg, #020617 0%, #0f172a 55%, #111827 100%);
        color: #e5eefb;
    }
    .hero-card, .info-card, .answer-card {
        border-radius: 20px;
        padding: 1.2rem 1.3rem;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 16px 40px rgba(2, 6, 23, 0.45);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.5;
    }
    .step-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        background: rgba(14, 165, 233, 0.16);
        color: #bae6fd;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .tiny-label {
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .source-card {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 5px solid #38bdf8;
        margin-bottom: 0.85rem;
    }
    .source-title {
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .source-meta {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .eval-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 1rem;
    }
    .eval-id {
        color: #7dd3fc;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .stMarkdown, .stText, p, label, .stCaption {
        color: #e5eefb !important;
    }
    .stTextInput > div > div > input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.9);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.24);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "eval_outputs" not in st.session_state:
    st.session_state.eval_outputs = {}
if "eval_ratings" not in st.session_state:
    st.session_state.eval_ratings = {}
if "selected_eval_case_id" not in st.session_state:
    st.session_state.selected_eval_case_id = ""
if "show_eval_results" not in st.session_state:
    st.session_state.show_eval_results = False

assistant_tab, evaluation_tab = st.tabs(["Assistant", "Evaluation"])

with assistant_tab:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">I am your medication interaction assistant.</div>
            <div class="hero-subtitle">
                Ask about possible medicine interactions, ingredient overlap, and safety concerns.
                I will search the reference documents, identify relevant evidence, and build a grounded answer from those sources.
            </div>
            <div style="margin-top:0.75rem;">
                <span class="step-chip">1. Search the question</span>
                <span class="step-chip">2. Retrieve matching evidence</span>
                <span class="step-chip">3. Generate a grounded answer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Try one of these example questions or type your own below.")
    example_columns = st.columns(len(EXAMPLE_QUERIES))
    for column, example in zip(example_columns, EXAMPLE_QUERIES):
        if column.button(example, width="stretch"):
            st.session_state.query_input = example

    query = st.text_input(
        "Your medication question",
        key="query_input",
        placeholder="Example: Can I take Panadol with warfarin?",
    )
    run_rag = st.button("Analyze Interaction", type="primary", width="stretch")

    if run_rag:
        if not query.strip():
            st.warning("Please enter a medication interaction question.")
        else:
            try:
                status = st.status("Starting RAG workflow...", expanded=True)
                status.write("Understanding your question.")
                with st.spinner("Searching the medication knowledge base..."):
                    output = run_rag_pipeline(query)
                st.session_state.last_answer = output["answer"]
                st.session_state.last_query = output["query"]
                st.session_state.last_results = output["results"]
                status.write(
                    f"Retrieved {len(st.session_state.last_results)} supporting document chunks."
                )
                status.update(
                    label="RAG workflow complete", state="complete", expanded=False
                )
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.last_answer:
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.subheader("Answer")
        st.markdown(st.session_state.last_answer)
        st.markdown("</div>", unsafe_allow_html=True)

        cited_sources = distinct_cited_sources(
            st.session_state.last_answer, st.session_state.last_results
        )
        render_main_source(cited_sources[0])
        render_other_sources(cited_sources)

with evaluation_tab:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("Evaluation Workspace")
    st.write(
        "Load test queries from the test-case file, generate an answer with the main source, then rate accuracy, clarity, safety, groundedness, citation accuracy, and retrieval relevance from 1 to 5."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    test_cases = load_test_cases()
    if not test_cases:
        st.warning("No test cases were found in the configured test-case file.")
    else:
        case_lookup = {case["id"]: case for case in test_cases}
        case_ids = [case["id"] for case in test_cases]
        if (
            not st.session_state.selected_eval_case_id
            or st.session_state.selected_eval_case_id not in case_lookup
        ):
            st.session_state.selected_eval_case_id = case_ids[0]

        selected_case_id = st.selectbox(
            "Choose a test case",
            case_ids,
            index=case_ids.index(st.session_state.selected_eval_case_id),
            format_func=lambda case_id: case_label(
                case_lookup[case_id], st.session_state.eval_ratings
            ),
            key="selected_eval_case_id",
        )
        selected_case = case_lookup[selected_case_id]

        st.markdown(
            f"""
            <div class="eval-card">
                <div class="eval-id">{selected_case['id']}</div>
                <div><strong>Query:</strong> {selected_case['input']}</div>
                <div style="margin-top:0.55rem;"><strong>Evaluation notes:</strong> {selected_case['notes']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Run Evaluation Case", type="primary", width="stretch"):
            try:
                with st.spinner("Running the selected test case..."):
                    st.session_state.eval_outputs[selected_case["id"]] = run_rag_pipeline(
                        selected_case["input"]
                    )
            except Exception as exc:
                st.error(str(exc))

        eval_output = st.session_state.eval_outputs.get(selected_case["id"])
        if eval_output:
            result_col, review_col = st.columns([1.3, 1], gap="large")

            with result_col:
                st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                st.subheader("Generated Answer")
                st.markdown(eval_output["answer"])
                st.markdown("</div>", unsafe_allow_html=True)

                cited_sources = distinct_cited_sources(
                    eval_output["answer"], eval_output["results"]
                )
                render_main_source(cited_sources[0])
                render_other_sources(cited_sources)

            with review_col:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.subheader("Rate The Answer")

                case_ratings = st.session_state.eval_ratings.get(selected_case["id"], {})
                with st.form(key=f"ratings_form_{selected_case['id']}"):
                    updated_ratings = dict(case_ratings)
                    for dimension in RATING_DIMENSIONS:
                        rating_key = f"{dimension}_{selected_case['id']}"
                        default_rating = case_ratings.get(dimension)
                        rating = st.radio(
                            dimension.replace("_", " ").title(),
                            options=[1, 2, 3, 4, 5],
                            horizontal=True,
                            index=[1, 2, 3, 4, 5].index(default_rating)
                            if default_rating in [1, 2, 3, 4, 5]
                            else None,
                            key=rating_key,
                        )
                        if rating is not None:
                            updated_ratings[dimension] = rating
                        elif dimension in updated_ratings:
                            del updated_ratings[dimension]

                    save_ratings = st.form_submit_button(
                        "Save Ratings", type="primary", width="stretch"
                    )

                if save_ratings:
                    missing_dimensions = [
                        dimension
                        for dimension in RATING_DIMENSIONS
                        if dimension not in updated_ratings
                    ]
                    if missing_dimensions:
                        st.warning(
                            "Please rate all metrics before saving: "
                            + ", ".join(
                                dimension.replace("_", " ").title()
                                for dimension in missing_dimensions
                            )
                        )
                    else:
                        st.session_state.eval_ratings[selected_case["id"]] = updated_ratings
                        output_path = save_ratings_to_csv(test_cases, st.session_state.eval_ratings)
                        st.write(
                            "Saved ratings: "
                            f"accuracy {updated_ratings['accuracy']}/5, "
                            f"clarity {updated_ratings['clarity']}/5, "
                            f"safety {updated_ratings['safety']}/5, "
                            f"groundedness {updated_ratings['groundedness']}/5, "
                            f"citation accuracy {updated_ratings['citation_accuracy']}/5, "
                            f"retrieval relevance {updated_ratings['retrieval_relevance']}/5"
                        )
                        st.caption(f"Ratings file updated: {output_path.name}")
                st.caption("1 = poor, 3 = acceptable, 5 = excellent")
                st.markdown("</div>", unsafe_allow_html=True)

        completed_count = sum(
            1 for case in test_cases if is_case_fully_rated(case["id"], st.session_state.eval_ratings)
        )
        total_count = len(test_cases)
        st.caption(f"Fully rated queries: {completed_count}/{total_count}")

        if st.button("Show Results", width="stretch"):
            st.session_state.show_eval_results = True

        rated_cases = [
            case for case in test_cases if is_case_fully_rated(case["id"], st.session_state.eval_ratings)
        ]

        if st.session_state.show_eval_results:
            saved_figure_paths = []
            if not rated_cases:
                st.info("Rate at least one full query to see the evaluation plots.")
            else:
                st.success(
                    f"Showing results for {len(rated_cases)} rated quer"
                    f"{'y' if len(rated_cases) == 1 else 'ies'}."
                )

            chart_values = []
            for case in rated_cases:
                case_ratings = st.session_state.eval_ratings.get(case["id"], {})
                for dimension in RATING_DIMENSIONS:
                    chart_values.append(
                        {
                            "query": case["id"],
                            "metric": dimension.replace("_", " ").title(),
                            "score": case_ratings[dimension],
                        }
                    )

            st.subheader("Ratings Per Query")
            ratings_per_query_chart = {
                "data": {"values": chart_values},
                "mark": {"type": "bar", "cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5},
                "encoding": {
                    "x": {
                        "field": "query",
                        "type": "nominal",
                        "sort": [case["id"] for case in test_cases],
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "xOffset": {"field": "metric"},
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 5]},
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "scale": {
                            "domain": [
                                "Accuracy",
                                "Clarity",
                                "Safety",
                                "Groundedness",
                                "Citation Accuracy",
                                "Retrieval Relevance",
                            ],
                            "range": [
                                "#38bdf8",
                                "#34d399",
                                "#f59e0b",
                                "#a78bfa",
                                "#f472b6",
                                "#f87171",
                            ],
                        },
                        "legend": {
                            "labelColor": "#e5eefb",
                            "titleColor": "#e5eefb",
                        },
                    },
                    "tooltip": [
                        {"field": "query", "type": "nominal"},
                        {"field": "metric", "type": "nominal"},
                        {"field": "score", "type": "quantitative"},
                    ],
                },
                "width": "container",
                "height": 360,
                "config": {
                    "background": "#0f172a",
                    "view": {"stroke": "transparent"},
                    "axis": {"gridColor": "rgba(148, 163, 184, 0.2)"},
                },
            }
            st.vega_lite_chart(ratings_per_query_chart, width="stretch")
            saved_figure_paths.append(save_chart_files(ratings_per_query_chart, "ratings_per_query"))

            prompt_only_rows = load_prompt_only_model_b_scores()
            rag_rows = []
            for case in rated_cases:
                case_ratings = st.session_state.eval_ratings.get(case["id"], {})
                rag_row = {
                    "id": case["id"],
                    "category": case.get("category", ""),
                }
                for dimension in RATING_DIMENSIONS:
                    rag_row[dimension] = case_ratings[dimension]
                rag_rows.append(rag_row)

            rag_metrics = rag_specific_metrics()
            rag_metric_summary_values = [
                {
                    "metric": metric.replace("_", " ").title(),
                    "score": average_metric_scores(rag_rows, rag_metrics)[metric],
                }
                for metric in rag_metrics
            ]

            st.subheader("RAG-Specific Metric Means")
            rag_metric_means_chart = {
                "data": {"values": rag_metric_summary_values},
                "mark": {"type": "bar", "cornerRadiusTopLeft": 6, "cornerRadiusTopRight": 6},
                "encoding": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 5]},
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "scale": {
                            "domain": [
                                "Groundedness",
                                "Citation Accuracy",
                                "Retrieval Relevance",
                            ],
                            "range": ["#a78bfa", "#f472b6", "#38bdf8"],
                        },
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "metric", "type": "nominal"},
                        {"field": "score", "type": "quantitative", "format": ".2f"},
                    ],
                },
                "width": "container",
                "height": 320,
                "config": {
                    "background": "#0f172a",
                    "view": {"stroke": "transparent"},
                    "axis": {"gridColor": "rgba(148, 163, 184, 0.2)"},
                },
            }
            st.vega_lite_chart(rag_metric_means_chart, width="stretch")
            saved_figure_paths.append(
                save_chart_files(rag_metric_means_chart, "rag_specific_metric_means")
            )

            rag_distribution_values = [
                {
                    "query": row["id"],
                    "metric": metric.replace("_", " ").title(),
                    "score": row[metric],
                }
                for row in rag_rows
                for metric in rag_metrics
            ]

            st.subheader("RAG-Specific Score Distribution Across Test Cases")
            rag_distribution_chart = {
                "data": {"values": rag_distribution_values},
                "mark": {"type": "boxplot", "extent": "min-max"},
                "encoding": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 5]},
                        "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "scale": {
                            "domain": [
                                "Groundedness",
                                "Citation Accuracy",
                                "Retrieval Relevance",
                            ],
                            "range": ["#a78bfa", "#f472b6", "#38bdf8"],
                        },
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "query", "type": "nominal"},
                        {"field": "metric", "type": "nominal"},
                        {"field": "score", "type": "quantitative", "format": ".2f"},
                    ],
                },
                "width": "container",
                "height": 320,
                "config": {
                    "background": "#0f172a",
                    "view": {"stroke": "transparent"},
                    "axis": {"gridColor": "rgba(148, 163, 184, 0.2)"},
                },
            }
            st.vega_lite_chart(rag_distribution_chart, width="stretch")
            saved_figure_paths.append(
                save_chart_files(
                    rag_distribution_chart, "rag_specific_score_distribution_across_test_cases"
                )
            )

            rag_heatmap_values = []
            categories = ["typical", "varied", "edge_case", "rag_needed"]
            for category in categories:
                category_rows = [row for row in rag_rows if row.get("category") == category]
                if not category_rows:
                    continue
                category_means = average_metric_scores(category_rows, rag_metrics)
                for metric in rag_metrics:
                    rag_heatmap_values.append(
                        {
                            "category": category,
                            "metric": metric.replace("_", " ").title(),
                            "score": category_means[metric],
                        }
                    )

            if rag_heatmap_values:
                st.subheader("RAG-Specific Mean Score By Test Case Category")
                rag_heatmap_chart = {
                    "data": {"values": rag_heatmap_values},
                    "mark": "rect",
                    "encoding": {
                        "x": {
                            "field": "metric",
                            "type": "nominal",
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "y": {
                            "field": "category",
                            "type": "nominal",
                            "sort": categories,
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "color": {
                            "field": "score",
                            "type": "quantitative",
                            "scale": {"domain": [1, 5], "scheme": "blues"},
                            "legend": {
                                "labelColor": "#e5eefb",
                                "titleColor": "#e5eefb",
                            },
                        },
                        "tooltip": [
                            {"field": "category", "type": "nominal"},
                            {"field": "metric", "type": "nominal"},
                            {"field": "score", "type": "quantitative", "format": ".2f"},
                        ],
                    },
                    "width": "container",
                    "height": 220,
                    "config": {
                        "background": "#0f172a",
                        "view": {"stroke": "transparent"},
                    },
                }
                st.vega_lite_chart(rag_heatmap_chart, width="stretch")
                saved_figure_paths.append(
                    save_chart_files(rag_heatmap_chart, "rag_specific_mean_score_by_category")
                )

            prompt_only_by_id = {row["id"]: row for row in prompt_only_rows if row.get("id")}
            rag_by_id = {row["id"]: row for row in rag_rows if row.get("id")}
            shared_case_ids = [case_id for case_id in prompt_only_by_id if case_id in rag_by_id]

            aligned_prompt_only_rows = [prompt_only_by_id[case_id] for case_id in shared_case_ids]
            aligned_rag_rows = [rag_by_id[case_id] for case_id in shared_case_ids]

            comparable_metrics = shared_prompt_metrics(aligned_prompt_only_rows)

            if shared_case_ids and comparable_metrics:
                prompt_only_avg = average_metric_scores(
                    aligned_prompt_only_rows, comparable_metrics
                )
                rag_avg = average_metric_scores(aligned_rag_rows, comparable_metrics)
                comparison_values = [
                    {
                        "approach": approach,
                        "metric": metric.replace("_", " ").title(),
                        "score": score,
                    }
                    for approach, scores in [
                        ("Prompting Only", prompt_only_avg),
                        ("RAG Enhanced", rag_avg),
                    ]
                    for metric, score in scores.items()
                ]

                st.subheader("Prompting Only vs RAG Enhanced")
                comparison_chart = {
                    "data": {"values": comparison_values},
                    "mark": {"type": "bar", "cornerRadiusTopLeft": 6, "cornerRadiusTopRight": 6},
                    "encoding": {
                        "x": {
                            "field": "metric",
                            "type": "nominal",
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "xOffset": {"field": "approach"},
                        "y": {
                            "field": "score",
                            "type": "quantitative",
                            "scale": {"domain": [0, 5]},
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "color": {
                            "field": "approach",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Prompting Only", "RAG Enhanced"],
                                "range": ["#f97316", "#38bdf8"],
                            },
                            "legend": {
                                "labelColor": "#e5eefb",
                                "titleColor": "#e5eefb",
                            },
                        },
                        "tooltip": [
                            {"field": "approach", "type": "nominal"},
                            {"field": "metric", "type": "nominal"},
                            {"field": "score", "type": "quantitative", "format": ".2f"},
                        ],
                    },
                    "width": "container",
                    "height": 320,
                    "config": {
                        "background": "#0f172a",
                        "view": {"stroke": "transparent"},
                        "axis": {"gridColor": "rgba(148, 163, 184, 0.2)"},
                    },
                }
                st.vega_lite_chart(comparison_chart, width="stretch")
                saved_figure_paths.append(
                    save_chart_files(comparison_chart, "prompting_only_vs_rag_enhanced")
                )

                prompt_overall = [
                    {
                        "approach": "Prompting Only",
                        "id": row["id"],
                        "overall": sum(row[metric] for metric in comparable_metrics)
                        / len(comparable_metrics),
                    }
                    for row in aligned_prompt_only_rows
                ]
                rag_overall = [
                    {
                        "approach": "RAG Enhanced",
                        "id": row["id"],
                        "overall": sum(row[metric] for metric in comparable_metrics)
                        / len(comparable_metrics),
                    }
                    for row in aligned_rag_rows
                ]
                overall_values = prompt_overall + rag_overall

                st.subheader("Overall Score Distribution Across Test Cases")
                overall_distribution_chart = {
                    "data": {"values": overall_values},
                    "mark": {"type": "boxplot", "extent": "min-max"},
                    "encoding": {
                        "x": {
                            "field": "approach",
                            "type": "nominal",
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "y": {
                            "field": "overall",
                            "type": "quantitative",
                            "scale": {"domain": [0, 5]},
                            "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                        },
                        "color": {
                            "field": "approach",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Prompting Only", "RAG Enhanced"],
                                "range": ["#f97316", "#38bdf8"],
                            },
                            "legend": None,
                        },
                        "tooltip": [
                            {"field": "approach", "type": "nominal"},
                            {
                                "field": "overall",
                                "type": "quantitative",
                                "format": ".2f",
                            },
                        ],
                    },
                    "width": "container",
                    "height": 320,
                    "config": {
                        "background": "#0f172a",
                        "view": {"stroke": "transparent"},
                        "axis": {"gridColor": "rgba(148, 163, 184, 0.2)"},
                    },
                }
                st.vega_lite_chart(overall_distribution_chart, width="stretch")
                saved_figure_paths.append(
                    save_chart_files(
                        overall_distribution_chart,
                        "overall_score_distribution_across_test_cases",
                    )
                )

                categories = ["typical", "varied", "edge_case"]
                heatmap_values = []
                for category in categories:
                    prompt_category_rows = [
                        row
                        for row in aligned_prompt_only_rows
                        if row.get("category") == category
                    ]
                    rag_category_rows = [
                        row for row in aligned_rag_rows if row.get("category") == category
                    ]

                    if prompt_category_rows:
                        prompt_mean = sum(
                            sum(row[metric] for metric in comparable_metrics)
                            / len(comparable_metrics)
                            for row in prompt_category_rows
                        ) / len(prompt_category_rows)
                        heatmap_values.append(
                            {
                                "category": category,
                                "approach": "Prompting Only",
                                "score": prompt_mean,
                            }
                        )

                    if rag_category_rows:
                        rag_mean = sum(
                            sum(row[metric] for metric in comparable_metrics)
                            / len(comparable_metrics)
                            for row in rag_category_rows
                        ) / len(rag_category_rows)
                        heatmap_values.append(
                            {
                                "category": category,
                                "approach": "RAG Enhanced",
                                "score": rag_mean,
                            }
                        )

                if heatmap_values:
                    st.subheader("Mean Overall Score By Test Case Category")
                    heatmap_chart = {
                        "data": {"values": heatmap_values},
                        "mark": "rect",
                        "encoding": {
                            "x": {
                                "field": "approach",
                                "type": "nominal",
                                "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                            },
                            "y": {
                                "field": "category",
                                "type": "nominal",
                                "sort": categories,
                                "axis": {"labelColor": "#e5eefb", "titleColor": "#e5eefb"},
                            },
                            "color": {
                                "field": "score",
                                "type": "quantitative",
                                "scale": {"domain": [1, 5], "scheme": "blues"},
                                "legend": {
                                    "labelColor": "#e5eefb",
                                    "titleColor": "#e5eefb",
                                },
                            },
                            "tooltip": [
                                {"field": "category", "type": "nominal"},
                                {"field": "approach", "type": "nominal"},
                                {"field": "score", "type": "quantitative", "format": ".2f"},
                            ],
                        },
                        "width": "container",
                        "height": 180,
                        "config": {
                            "background": "#0f172a",
                            "view": {"stroke": "transparent"},
                        },
                    }
                    st.vega_lite_chart(heatmap_chart, width="stretch")
                    saved_figure_paths.append(
                        save_chart_files(heatmap_chart, "mean_overall_score_by_category")
                    )
            else:
                st.info(
                    "The RAG vs Prompting Only comparison needs shared rated case IDs and "
                    "matching prompt-only metrics. Rate the same cases present in the "
                    "prompt-only CSV to populate these plots."
                )

            if rated_cases:
                saved_names = ", ".join(paths["html"].name for paths in saved_figure_paths)
                st.caption(f"Saved figures in {FIGURES_DIR.name}: {saved_names}")
