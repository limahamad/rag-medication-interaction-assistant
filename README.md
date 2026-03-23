# Drug Interaction Assistant

## Project Description
The Drug Interaction Assistant is an AI-powered system designed to answer medication-related questions, with a focus on **drug–drug interaction detection**. The system analyzes user queries, identifies the mentioned drug names and then check interactions at the **ingredient level**, which improves accuracy because many brand names may contain the same active compounds.

The system also uses a **Retrieval-Augmented Generation (RAG)** architecture to retrieve relevant drug information from a knowledge base before generating responses with a large language model. By combining ingredient-level interaction detection with contextual retrieval, the assistant provides more reliable and medically grounded responses.

The workflow ensures that even if users provide brand names, misspellings, or different drug formats, the system can normalize them and evaluate interactions based on the underlying active ingredients.

---

# System Architecture

The system consists of several modules that process a user query and determine possible drug interactions.

```
User Query
     |
     v
Drug Name Extraction
     |
     v
Drug Name Normalization
(brand → standardized drug name)
     |
     v
Ingredient Mapping
(drug → active ingredients)
     |
     v
Ingredient-Level Interaction Check
     |
     v
Retrieval System (RAG)
     |
Retrieve Relevant Drug Information
     |
     v
LLM Response Generation
     |
     v
Final Answer
```

---

# Architecture Components

## User Interface
- Built using **Streamlit**
- Allows users to ask natural language questions about medications and drug interactions

## Drug Name Normalization
- Converts user-provided drug names to standardized names
- Handles brand names, spelling variations, and synonyms

Example:

```
Advil → Ibuprofen
Tylenol → Acetaminophen
```

## Ingredient Mapping
Each normalized drug is mapped to its **active ingredients**.

Example:

```
Advil → Ibuprofen
Augmentin → Amoxicillin + Clavulanic Acid
```

## Ingredient-Level Interaction Detection
Drug interactions are determined by checking **interactions between active ingredients**, not just drug names.

Example:

```
Drug A: Augmentin → Amoxicillin + Clavulanic Acid
Drug B: Warfarin → Warfarin

Check interactions:
Amoxicillin ↔ Warfarin
Clavulanic Acid ↔ Warfarin
```

This approach ensures more accurate detection of interactions.

## Retrieval-Augmented Generation (RAG)

The system retrieves relevant drug information before generating responses.

Steps:
1. Drug documents are split into text chunks
2. Each chunk is converted into vector embeddings
3. Embeddings are stored in a **vector database**
4. When a query is received, similar documents are retrieved
5. Retrieved context is passed to the **language model** to generate a grounded answer

---

# Project Structure

```
drug-interaction-assistant/
│
├── app.py
├── interaction_engine.py
├── normalization.py
├── rag_pipeline.py
│
├── data/
│   ├── drug_database.json
│   └── interaction_data.json
│
├── vector_db/
│
├── requirements.txt
└── README.md
```

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/drug-interaction-assistant.git
cd drug-interaction-assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

### Mac / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure API Key

Create a `.env` file in the project directory.

```
OPENAI_API_KEY=your_api_key_here
```

---

## 5. Run the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Example Queries

Users can ask questions such as:

```
Can I take Advil with warfarin?
Does ibuprofen interact with aspirin?
Is it safe to take Tylenol with amoxicillin?
What drugs interact with metformin?
```

The system will:
1. Extract drug names
2. Normalize them
3. Map them to ingredients
4. Check ingredient-level interactions
5. Generate an explanation of the interaction

---

# Technologies Used

- Python
- Streamlit
- OpenAI API
- Vector Database (Chroma / FAISS)
- Retrieval-Augmented Generation (RAG)
- Drug normalization and ingredient mapping

---

# Future Improvements

Potential improvements include:

- Expanding the drug database with larger pharmaceutical datasets
- Adding interaction severity classification
- Providing citations from medical sources
- Supporting multi-drug interaction analysis
- Integrating official medical knowledge bases such as DrugBank or RxNorm