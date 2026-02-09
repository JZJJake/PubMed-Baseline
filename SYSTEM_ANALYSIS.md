# PubMed Literature Assistant System Analysis

This document provides an overview of the functionality of the PubMed Literature Assistant system based on code analysis.

## System Overview

The system is a command-line interface (CLI) tool designed to assist researchers in downloading, parsing, indexing, and querying PubMed literature data. It integrates keyword search, semantic search (using vector embeddings), and AI-powered question answering.

## Key Components

### 1. Data Synchronization (`src/downloader.py`)
- **Functionality**: Automates the process of downloading baseline XML data files from the PubMed FTP server (`https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/`).
- **Features**:
  - Supports resuming interrupted downloads (Range header).
  - Implements robust retry logic with exponential backoff.
  - displays download progress bars.

### 2. Data Parsing (`src/parser.py`)
- **Functionality**: Parses the downloaded `.xml.gz` files to extract structured metadata.
- **Extracted Fields**: PMID, Title, Abstract, Authors, Journal, Year, DOI, Keywords, Mesh Terms, etc.
- **Output**: Saves the extracted data into a JSONL file (`data/metadata.jsonl`) for efficient reading and processing.

### 3. Vector Indexing (`src/vector_store.py`)
- **Functionality**: Creates a semantic search index from the parsed papers.
- **Technology**:
  - Uses **ChromaDB** as the vector database.
  - Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to generate embeddings for paper titles and abstracts.
- **Storage**: Persists the index locally in `data/chroma_db/`.

### 4. Search Engine (`src/vector_store.py` & `main.py`)
- **Keyword Search**: Performs a simple text match against the title and abstract in the JSONL file.
- **Semantic Search**:
  - Utilizes the vector index to find papers that are semantically similar to the user's query, even if they don't share exact keywords.
  - *Note*: A critical bug in the semantic search implementation was identified and fixed during this analysis.

### 5. AI Question Answering (`src/ai.py`)
- **Functionality**: Integrates with the **DeepSeek API** to answer natural language questions.
- **Process (RAG - Retrieval Augmented Generation)**:
  1.  User asks a question.
  2.  System extracts keywords from the question.
  3.  System retrieves relevant papers using semantic search (or keyword search as fallback).
  4.  System constructs a prompt containing the question and the abstracts of the retrieved papers.
  5.  AI generates an answer based on the provided context.

### 6. Interactive CLI (`main.py`)
- **Functionality**: Orchestrates all the above components through an interactive shell.
- **Commands**:
  - `sync`: Download data.
  - `parse`: Parse downloaded data.
  - `index`: Build vector index.
  - `search`: Search for papers (supports `-v` flag for semantic search).
  - `ask`: Ask a question to the AI.
  - `config`: Set configuration (e.g., API keys).
- **UI**: Uses the `rich` library for formatted output (tables, markdown, progress bars).

## Conclusion

The system provides an end-to-end workflow for local PubMed data management and intelligent retrieval, leveraging modern NLP techniques (embeddings, LLMs) to enhance literature search capabilities.
