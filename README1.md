  ## System Architecture Flow

```text
[ TN Government Portal ] 
        │
        ▼ (Playwright Scraper)
  [ Raw JSON Data ] (data/tn_agriculture_schemes.json)
        │
        ▼ (LangChain + OpenAI Embeddings: text-embedding-3-small)
  [ Neo4j Aura Database ] (Nodes: AgriculturalScheme, Index: tn_scheme_vector_index)
        │
        ▲ (Hybrid/Vector Similarity Retrieval via langchain-neo4j)
        │
  [ Streamlit UI (app.py) ] ◄── User Chat Input
        │
        ▼ (LangChain LCEL & ChatOpenAI: gpt-4o-mini)
  [ Grounded RAG Response & Source Metadata ] ──► Displayed to User

```

---

### Component Breakdown

1. **Data Extraction Layer (`scraper.py`)**
* Uses **Playwright with Chromium** to automate navigation through the Tamil Nadu Agriculture and Farmers Welfare Department portal.
* Extracts detailed scheme fields (title, department, benefits, eligibility, how-to-apply) and serializes them into local JSON storage.


2. **Data Ingestion Layer (`ingest_neo4j.py`)**
* Parses JSON scheme records into LangChain `Document` objects.
* Utilizes OpenAI's **`text-embedding-3-small`** model to generate vector representations.
* Connects securely to **Neo4j Aura** using `neo4j+ssc://` (self-signed certificate configuration) and populates the vector store with the `AgriculturalScheme` node label and `tn_scheme_vector_index`.


3. **Application & RAG Layer (`app.py`)**
* **Streamlit UI**: Provides an interactive, stateful web chat interface (`localhost:8501`).
* **Retrieval**: Queries the Neo4j Aura vector index (`Neo4jVector.from_existing_index`) to fetch the top-k most relevant agricultural schemes.
* **Generation**: Uses LangChain Expression Language (LCEL), custom prompt templates, and OpenAI's **`gpt-4o-mini`** LLM to synthesize accurate, grounded answers accompanied by expandable source links.