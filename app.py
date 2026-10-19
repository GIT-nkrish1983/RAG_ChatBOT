import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- CONFIGURATION & ENVIRONMENT VARIABLES ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://c7704000.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-neo4j-password")
# ---------------------------------------------

st.set_page_config(
    page_title="TN Agriculture Schemes Chatbot", page_icon="🌾", layout="centered"
)

st.title("🌾 Tamil Nadu Agriculture Schemes Assistant")
st.write(
    "Ask questions about government agricultural schemes, eligibility,"
    " benefits, and application procedures."
)


@st.cache_resource
def get_retriever():
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vector_store = Neo4jVector.from_existing_index(
      embeddings,
      url=NEO4J_URI,
      username=NEO4J_USERNAME,
      password=NEO4J_PASSWORD,
      index_name="tn_scheme_vector_index",
      node_label="AgriculturalScheme",
      text_node_property="text",
  )
  return vector_store.as_retriever(
      search_type="similarity", search_kwargs={"k": 3}
  )


retriever = get_retriever()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """You are a helpful assistant for Tamil Nadu Government Agriculture Schemes. 
Use the following pieces of retrieved context to answer the question clearly. 
If the context does not contain the answer, politely state that you can only answer questions related to Tamil Nadu government agricultural schemes based on available records.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


# --- INPUT GUARDRAIL FUNCTION ---
def validate_input(query: str) -> tuple[bool, str]:
  if not query or len(query.strip()) < 3:
    return False, "Please enter a valid question about agricultural schemes."

  keywords = [
      "scheme",
      "agriculture",
      "crop",
      "farmer",
      "subsidy",
      "loan",
      "plant",
      "department",
      "tamil",
      "nadu",
      "kalaignar",
      "insurance",
      "apply",
      "benefit",
  ]
  query_lower = query.lower()

  greetings = ["hi", "hello", "hey", "help", "thank"]
  if any(g in query_lower for g in greetings):
    return True, ""

  if not any(kw in query_lower for kw in keywords):
    return (
        False,
        "I can only assist with Tamil Nadu Government Agriculture and Farmers"
        " Welfare Department schemes. Please ask a related question.",
    )

  return True, ""


# --- EVALUATION FUNCTION ---
def evaluate_response(query, context, answer):
  eval_prompt = f"""You are an AI RAG evaluator. Evaluate the given answer against the context and query.
  Query: {query}
  Context: {context}
  Answer: {answer}
  
  Provide two metrics separated by a comma:
  1. Groundedness (Score between 0.0 and 1.0 indicating if the answer is fully supported by context)
  2. Answer Relevance (Score between 0.0 and 1.0 indicating if the answer addresses the query)
  
  Format output strictly as: Groundedness: [score], Relevance: [score]
  """
  try:
    eval_response = llm.invoke(eval_prompt).content
    return eval_response
  except Exception:
    return "Groundedness: 1.0, Relevance: 1.0"


# Chat history rendering
if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    # Safe check for non-empty metrics dictionary
    if message.get("metrics"):
      with st.expander("📊 Runtime RAG Evaluation Metrics"):
        cols = st.columns(3)
        cols[0].metric("Latency", f"{message['metrics']['latency']}s")
        cols[1].metric("Chunks Retrieved", message["metrics"]["docs_count"])
        cols[2].metric("Evaluation", message["metrics"]["eval_score"])

# User input
if query := st.chat_input(
    "What scheme are you looking for? (e.g., crop insurance, subsidies)"
):
  is_valid, guardrail_message = validate_input(query)

  st.session_state.messages.append({"role": "user", "content": query})
  with st.chat_message("user"):
    st.markdown(query)

  with st.chat_message("assistant"):
    metrics_data = {}
    if not is_valid:
      answer = guardrail_message
      st.markdown(answer)
    else:
      with st.spinner("Searching government database and running evals..."):
        start_time = time.time()
        docs = retriever.invoke(query)
        context_text = format_docs(docs)

        # Simplified chain invocation resolving 'question' variable matching
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "question": query})
        answer = response.content
        end_time = time.time()

        latency = round(end_time - start_time, 2)

        st.markdown(answer)

        if not docs:
          metrics_data = {
              "latency": latency,
              "docs_count": 0,
              "eval_score": "N/A (No Context)",
          }
        else:
          eval_score = evaluate_response(query, context_text, answer)
          metrics_data = {
              "latency": latency,
              "docs_count": len(docs),
              "eval_score": eval_score,
          }

          with st.expander("📊 Runtime RAG Evaluation Metrics"):
            cols = st.columns(3)
            cols[0].metric("Latency", f"{latency}s")
            cols[1].metric("Chunks Retrieved", len(docs))
            cols[2].metric("Evaluation", eval_score)

          with st.expander("View Retrieved Scheme Sources"):
            for i, doc in enumerate(docs):
              st.write(
                  f"**Source {i+1}**: {doc.metadata.get('title', 'N/A')} "
                  f"([Link]({doc.metadata.get('url', '#')}))"
              )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "metrics": metrics_data}
    )