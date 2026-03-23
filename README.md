# Medication Interaction RAG Assistant

This project is a Streamlit-based medication interaction assistant built with retrieval-augmented generation (RAG). It retrieves supporting medication documents from a local FAISS + BM25 index, sends the grounded context to Gemini, and returns a structured answer with cited sources.

The project also includes an evaluation workspace for rating generated answers and scripts for building the RAG index and generating saved evaluation figures for different retrieval settings.

## Implemented Features

- Streamlit app with two tabs:
  - `Assistant` for normal question answering
  - `Evaluation` for running test cases and rating outputs
- Hybrid retrieval:
  - dense retrieval with `sentence-transformers` + FAISS
  - lexical retrieval with BM25-style scoring
  - fused ranking between dense and lexical scores
- Gemini-based grounded answer generation
- Main Source and Other Sources display based on the citations actually used in the answer
- Evaluation workflow with manual rating for:
  - Accuracy
  - Clarity
  - Safety
  - Groundedness
  - Citation Accuracy
  - Retrieval Relevance
- Saved evaluation plots inside the app
- Standalone `plot_figures.py` script to generate Matplotlib figures for:
  - `K=3`
  - `K=5`
  - `K=7`

## Project Structure

```text
Assignment2/
├── .streamlit/
│   └── secrets.toml                # local only, not for GitHub
├── figures/                        # saved Matplotlib figures from plot_figures.py
│   ├── k_3/
│   ├── k_5/
│   └── k_7/
├── rag_index/
│   ├── docs.index
│   ├── chunks.pkl
│   └── bm25.pkl
├── A1_Hamad _testcases.json        # evaluation test cases
├── app.py                          # Streamlit app
├── build_index.py                  # builds FAISS + BM25 index
├── drug_docs.json                  # document collection used for retrieval
├── evaluation_outputs_to_score.csv # prompt-only baseline scores (Model B used)
├── evaluation_ratings_K_3.csv      # RAG ratings for K=3
├── evaluation_ratings_K_5.csv      # RAG ratings for K=5
├── evaluation_ratings_K_7.csv      # RAG ratings for K=7
├── plot_figures.py                 # saves paper-style Matplotlib figures
├── requirements.txt
└── README.md
```

## Retrieval Pipeline

The retrieval pipeline works as follows:

1. Documents from `drug_docs.json` are split into chunks.
2. Each chunk is embedded using `all-MiniLM-L6-v2`.
3. Dense vectors are stored in a FAISS index.
4. BM25-style lexical statistics are also precomputed.
5. At query time:
   - dense similarity scores are computed
   - BM25 lexical scores are computed
   - both are normalized and fused
6. The top-ranked chunks are passed to Gemini as grounded context.

## Evaluation Workflow

The `Evaluation` tab lets you:

1. choose a test case from `A1_Hamad_testcases.json`
2. run the query through the RAG pipeline
3. inspect the generated answer
4. rate six evaluation dimensions
5. display evaluation plots at any time

The app also compares RAG against prompt-only baseline scores from `evaluation_outputs_to_score.csv`, using shared test-case IDs only.

## Saved Figures

`plot_figures.py` reads:

- `evaluation_ratings_K_3.csv`
- `evaluation_ratings_K_5.csv`
- `evaluation_ratings_K_7.csv`
- `evaluation_outputs_to_score.csv` using only Model B columns

and saves PNG figures to:

```text
figures/
├── k_3/
├── k_5/
└── k_7/
```

Generated figures include:

- `ratings_per_query.png`
- `prompting_only_vs_rag_enhanced.png`
- `overall_score_distribution_across_test_cases.png`
- `mean_overall_score_by_category.png`
- `rag_specific_metric_means.png`
- `rag_specific_score_distribution_across_test_cases.png`
- `rag_specific_mean_score_by_category.png`
- `retrieval_quality_vs_answer_accuracy.png`

## Setup

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv llm_layth_A2
llm_layth_A2\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

Do not commit this file to GitHub.

### 4. Build the retrieval index

```bash
python build_index.py
```

This creates:

- `rag_index/docs.index`
- `rag_index/chunks.pkl`
- `rag_index/bm25.pkl`

### 5. Run the app

```bash
streamlit run app.py
```

## Generate Figures

To generate all saved evaluation figures:

```bash
python plot_figures.py
```

## Main Dependencies

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Generative AI
- NumPy
- Matplotlib

## Notes

- `secrets.toml` should stay local and be ignored by Git.
- The prompt-only comparison uses Model B values only.
- The RAG vs prompt-only comparison is aligned by shared test-case IDs.
- The plotting script uses Matplotlib and saves publication-style PNG files.
