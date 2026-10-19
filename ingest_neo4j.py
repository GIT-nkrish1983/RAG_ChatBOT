import json
import os
from pathlib import Path
#from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()

# Configure your environment variables
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DATA_PATH = Path("data/tn_agriculture_schemes.json")


def ingest_schemes_to_neo4j():
  if not DATA_PATH.exists():
    print(f"Data file not found at {DATA_PATH}. Run the Playwright scraper first.")
    return

  with open(DATA_PATH, "r", encoding="utf-8") as f:
    schemes = json.load(f)

  documents = []
  for scheme in schemes:
    content_lines = [
        f"Scheme Title: {scheme.get('title')}",
        f"Department: {scheme.get('department')}",
        f"Organisation: {scheme.get('organisation')}",
        f"Sponsored By: {scheme.get('sponsored_by')}",
        f"Funding Pattern: {scheme.get('funding_pattern')}",
        f"Beneficiary: {scheme.get('beneficiary')}",
        f"Scheme Type: {scheme.get('scheme_type')}",
        f"Benefits: {scheme.get('benefits')}",
        f"Description: {scheme.get('description')}",
        f"How to Apply: {scheme.get('how_to_apply')}",
    ]
    page_content = "\n".join(content_lines)

    metadata = {
        "scheme_id": scheme.get("scheme_id", ""),
        "title": scheme.get("title", ""),
        "url": scheme.get("url", ""),
        "department": scheme.get("department", ""),
        "beneficiary": scheme.get("beneficiary", ""),
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

  print(f"Loaded {len(documents)} schemes. Ingesting into Neo4j Aura...")

  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  Neo4jVector.from_documents(
      documents,
      embeddings,
      url=os.environ["NEO4J_URI"],
      username=os.environ["NEO4J_USERNAME"],
      password=os.environ["NEO4J_PASSWORD"],
      index_name="tn_scheme_vector_index",
      node_label="AgriculturalScheme",
      text_node_property="text",
      embedding_node_property="embedding",
      search_type="hybrid",
  )

  print("Ingestion Complete: Schemes stored successfully in Neo4j Aura!")


if __name__ == "__main__":
  ingest_schemes_to_neo4j()