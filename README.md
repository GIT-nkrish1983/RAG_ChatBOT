# RAG_ChatBOT
Welcome to your RAG chatbot project! This is a comprehensive, production-grade architecture that combines Retrieval-Augmented Generation (RAG) with a Knowledge Graph (Neo4j Aura), LLM guardrails, and evaluation metrics.


# Technology Stack

Component	Technology	Purpose
Language	Python	Entire application
Scraping	Requests + BeautifulSoup initially	Download and parse accessible HTML
Document loading	LangChain Document objects	Standardise source content
Text splitting	RecursiveCharacterTextSplitter	Divide content into chunks
Entity extraction	LangChain LLMGraphTransformer	Extract entities and relationships using an LLM
Graph storage	Neo4j	Store entities, relationships and source chunks
Vector search	Neo4j vector index	Retrieve semantically relevant chunks
Chat model	OpenAI through LangChain	Generate evidence-based answers
UI	Streamlit	Interactive chatbot window
Persistence	Neo4j database on disk or hosted service	Retain data across sessions and restarts



Step 1 (Part 1): Web Scraping TN Government Schemes via Playwright  ( Scrape.py)
This documentation covers Step 1 (Part 1) of your RAG chatbot project, which focuses on automated web scraping of the Tamil Nadu government agricultural schemes portal using Playwright and BeautifulSoup, extracting structured tabular key-value fields, and saving them into a raw JSON file.

Prerequisites
Python 3.8+ installed in your environment.

Internet connection to access the Tamil Nadu government portal.

Process steps : 
1. Install Playwright
2. Use Scraper.py 
3. Input - Provide the URL 
4. Output - JSON file will be created with web scraped details 


Step 1 (Part 2): Neo4j Aura Data Ingestion  ( ingest_neo4j.py) 
This guide covers the process of loading the scraped agricultural schemes data (tn_agriculture_schemes.json) and ingesting it into Neo4j Aura using LangChain and OpenAI Embeddings to create a hybrid vector and graph search index.

Prerequisites
An active Neo4j Aura instance (or local Neo4j database).

An OpenAI API Key for generating vector embeddings (text-embedding-3-small).

Completed execution of the Playwright scraper, which outputs data/tn_agriculture_schemes.json.

PRocess Steps:
1. Create Neo4j Aura DB instance , copy the User name, passoword , DB, URI details 
2. Apply the Neo4j details and Run the scraper.py to ingest the data from JSON file to the Vector DB
3. Verify the data in Neo4j 
-   Log in to your Neo4j Aura Console.
-   Open Neo4j Browser for your instance.
-   Run the following Cypher query to verify that the nodes and vector indices were created successfully:


Step 2: Streamlit Frontend & LangChain RAG Chatbot
This documentation covers Step 2 of your project, which involves building an interactive web user interface using Streamlit and connecting it to your hybrid vector store hosted in Neo4j Aura via LangChain and OpenAI.

Prerequisites
Completed execution of Step 1 (data scraped and ingested into Neo4j Aura under AgriculturalScheme and tn_scheme_vector_index).
An active OpenAI API Key.

Process steps
1. Use Streamlit as front end, Langchain for connecting the LLM
