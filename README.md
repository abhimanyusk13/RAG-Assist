# Smart Research Assistant

The **Smart Research Assistant** is a Retrieval-Augmented Generation (RAG) chatbot that answers user queries by retrieving relevant information from a curated knowledge base (e.g., Wikipedia articles, arXiv papers) and generating natural language responses. This project demonstrates a practical application of RAG, combining vector search for retrieval and a generative model for response synthesis, deployed as a web app.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Implementation Steps](#implementation-steps)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Smart Research Assistant enables users to ask questions in natural language (e.g., "What are transformers in NLP?") and receive accurate, context-aware answers based on a knowledge base. It uses a RAG pipeline:
1. **Retrieval**: Embeds user queries and retrieves relevant document chunks using FAISS and Sentence Transformers.
2. **Generation**: Combines the query and retrieved documents to generate a response using a Hugging Face model (e.g., T5) or an external API (e.g., xAI's Grok 3).
3. **Web Interface**: Provides a user-friendly interface via Streamlit to input queries and view responses with source references.

This project is ideal for showcasing NLP skills, building a portfolio piece, or exploring RAG applications in domains like research, education, or customer support.

## Features
- Query answering grounded in a customizable knowledge base.
- Web interface for interactive user experience.
- Source attribution (displays document snippets and metadata).
- Modular design for easy extension (e.g., new datasets or models).
- Support for local models or cloud APIs for generation.

## Tech Stack
- **Retriever**: Sentence Transformers (`all-MiniLM-L6-v2`), FAISS
- **Generator**: Hugging Face Transformers (T5 or BART) or xAI/OpenAI API
- **Web Framework**: Streamlit (or Flask for custom UI)
- **Data Processing**: Python, NumPy, Pandas
- **Knowledge Base**: Text files (e.g., Wikipedia, arXiv papers)
- **Dependencies**: Listed in `docs/requirements.txt`
- **Optional**: Pinecone for cloud-based vector search, Heroku for deployment

## Project Structure
```
smart_research_assistant/
├── data/
│   ├── raw/documents/ (raw documents, e.g., .txt or .pdf)
│   ├── processed/chunks/ (preprocessed text chunks)
│   ├── processed/embeddings/ (FAISS index)
│   └── metadata/ (document metadata in JSON)
├── src/
│   ├── preprocessing/ (document cleaning and embedding)
│   ├── retrieval/ (vector search logic)
│   ├── generation/ (response generation)
│   └── utils/ (logging, helpers)
├── app/
│   ├── main.py (Streamlit app)
│   ├── templates/ (HTML for Flask, optional)
│   └── static/ (CSS/JS, optional)
├── tests/ (unit tests)
├── config/ (settings and secrets)
├── scripts/ (automation scripts)
├── docs/
│   ├── README.md (this file)
│   └── requirements.txt (dependencies)
├── .gitignore
└── environment.yml (optional Conda config)
```

## Setup Instructions
Follow these steps to set up the project locally.

### Prerequisites
- **Hardware**: Laptop/desktop with 8GB+ RAM (GPU optional for faster embedding/generation).
- **Software**: Python 3.8+, pip, Git.
- **Optional**: Conda for environment management, Docker for deployment.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/smart-research-assistant.git
   cd smart-research-assistant
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r docs/requirements.txt
   ```
   Key packages: `sentence-transformers`, `transformers`, `faiss-cpu`, `streamlit`, `numpy`, `pandas`.

4. **Optional: Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate smart-research-assistant
   ```

5. **Configure Settings**:
   - Edit `config/settings.yaml` to set:
     - `knowledge_base_path`: Path to `data/processed/chunks/`.
     - `faiss_index_path`: Path to `data/processed/embeddings/faiss_index.bin`.
     - `model_name`: Hugging Face model (e.g., `t5-small`) or API endpoint.
     - `top_k`: Number of documents to retrieve (e.g., 3).
   - Add API keys to `config/secrets.yaml` (e.g., for xAI API; see https://x.ai/api).

6. **Prepare Knowledge Base**:
   - Place raw documents in `data/raw/documents/` (e.g., `.txt` files from Wikipedia or arXiv).
   - Run preprocessing (see [Implementation Steps](#implementation-steps)).

## Implementation Steps
Here’s a detailed guide on what needs to be done to build the project, broken down by component.

### 1. Prepare Knowledge Base
**Goal**: Collect and preprocess documents for retrieval.
- **Tasks**:
  - **Collect Documents**: Download a small dataset (e.g., 100-1000 documents) from sources like:
    - Wikipedia (use `wikipedia-api` or download dumps from https://dumps.wikimedia.org).
    - arXiv (use `arxiv` Python package or API).
    - Custom dataset (e.g., text files or PDFs).
    - Save in `data/raw/documents/`.
  - **Preprocess Documents**:
    - Clean text (remove HTML, special characters) using `src/preprocessing/preprocess.py`.
    - Chunk documents into 200-500 word segments for efficient retrieval.
    - Save chunks in `data/processed/chunks/`.
    - Generate metadata (e.g., document title, source) in `data/metadata/document_metadata.json`.
  - **Generate Embeddings**:
    - Use `src/preprocessing/embed.py` with Sentence Transformers (`all-MiniLM-L6-v2`).
    - Convert chunks to embeddings and build a FAISS index.
    - Save index in `data/processed/embeddings/faiss_index.bin`.
  - **Script**:
    ```bash
    bash scripts/preprocess_data.sh
    bash scripts/build_index.sh
    ```
- **Estimated Time**: 8-12 hours (2-3 hours for collection, 3-4 for preprocessing, 3-5 for embeddings).

### 2. Build Retriever
**Goal**: Implement document retrieval using vector search.
- **Tasks**:
  - Use `src/retrieval/retriever.py` to:
    - Load FAISS index and Sentence Transformer model.
    - Embed user queries and retrieve top-k document chunks via cosine similarity.
    - Return document text and metadata.
  - Configure retrieval settings in `src/retrieval/config.py` (e.g., `top_k=3`).
  - Test retrieval with sample queries to ensure relevance.
- **Example**:
  ```python
  from sentence_transformers import SentenceTransformer
  import faiss
  import numpy as np

  model = SentenceTransformer('all-MiniLM-L6-v2')
  index = faiss.read_index('data/processed/embeddings/faiss_index.bin')
  query = "What are transformers in NLP?"
  query_embedding = model.encode([query])
  distances, indices = index.search(query_embedding, k=3)
  ```
- **Estimated Time**: 5-7 hours (3-4 for coding, 2-3 for testing).

### 3. Build Generator
**Goal**: Generate responses using retrieved documents.
- **Tasks**:
  - Use `src/generation/generator.py` to:
    - Combine query and retrieved documents into a prompt (see `src/generation/prompts.py`).
    - Pass prompt to a Hugging Face model (e.g., `t5-small`) or API (e.g., xAI’s Grok 3).
    - Extract and format the generated response.
  - Example prompt: `Based on this context: {retrieved_docs}, answer: {query}`.
  - Test generation with sample queries to ensure coherence.
- **Example**:
  ```python
  from transformers import pipeline
  generator = pipeline('text2text-generation', model='t5-small')
  context = "Transformers are neural networks used in NLP."
  query = "What are transformers in NLP?"
  prompt = f"Based on this context: {context}, answer: {query}"
  response = generator(prompt, max_length=100)[0]['generated_text']
  ```
- **Estimated Time**: 5-7 hours (3-4 for coding, 2-3 for prompt tuning).

### 4. Develop Web Interface
**Goal**: Create a user-friendly interface for querying and viewing responses.
- **Tasks**:
  - Use `app/main.py` to build a Streamlit app:
    - Add text input for user queries.
    - Display generated response, retrieved document snippets, and metadata (e.g., source titles).
    - Include basic styling (e.g., via Streamlit’s layout options).
  - Test UI with various queries to ensure functionality.
- **Example**:
  ```python
  import streamlit as st
  from src.retrieval.retriever import retrieve_documents
  from src.generation.generator import generate_response

  st.title("Smart Research Assistant")
  query = st.text_input("Ask a question:")
  if query:
      docs = retrieve_documents(query)
      response = generate_response(query, docs)
      st.write("**Answer:**", response)
      st.write("**Sources:**", [doc['metadata']['title'] for doc in docs])
  ```
- **Estimated Time**: 8-12 hours (4-6 for basic UI, 2-4 for enhancements, 2-3 for testing).

### 5. Testing
**Goal**: Ensure the pipeline works reliably.
- **Tasks**:
  - Write unit tests in `tests/` for preprocessing, retrieval, and generation.
    - Example: Test `retrieve_documents()` returns relevant chunks.
  - Perform end-to-end testing with diverse queries (e.g., factual, open-ended).
  - Debug issues like irrelevant retrievals or incoherent responses.
- **Tools**: `pytest` for unit tests.
- **Estimated Time**: 6-10 hours (3-4 for testing, 2-4 for optimization).

### 6. Deployment
**Goal**: Host the app for public access.
- **Tasks**:
  - **Local Deployment**:
    ```bash
    streamlit run app/main.py
    ```
  - **Cloud Deployment**:
    - Use Streamlit Cloud (free tier):
      - Push code to GitHub.
      - Connect repository to Streamlit Cloud.
      - Configure `requirements.txt` and `settings.yaml`.
    - Alternative: Heroku or Hugging Face Spaces.
  - Verify app accessibility and performance.
- **Estimated Time**: 4-8 hours (1-2 for local, 2-4 for cloud, 1-2 for documentation).

## Usage
1. **Run Locally**:
   ```bash
   streamlit run app/main.py
   ```
   Open `http://localhost:8501` in your browser.

2. **Ask Questions**:
   - Enter a query (e.g., "What is the role of transformers in NLP?").
   - View the response and source documents.

3. **Example Output**:
   - Query: "What are transformers in NLP?"
   - Response: "Transformers are neural networks pivotal in NLP, using self-attention for tasks like translation."
   - Sources: "Wikipedia: Transformer (machine learning model)", "arXiv: BERT paper".

## Testing
- Run unit tests:
  ```bash
  pytest tests/
  ```
- Manually test with queries covering different topics and formats.
- Check logs in `src/utils/logger.py` for debugging.

## Deployment
- **Streamlit Cloud**:
  - Create an account at https://share.streamlit.io.
  - Link your GitHub repository and deploy.
- **Heroku**:
  - Install Heroku CLI, create an app, and push code:
    ```bash
    heroku create
    git đẩy heroku main
    ```
- Ensure `data/` is accessible (e.g., host embeddings locally or use Pinecone).

## Future Enhancements
- **Real-Time Data**: Integrate web scraping or X post retrieval for fresh content.
- **Multi-Modal RAG**: Support images or tables in documents using CLIP.
- **Advanced UI**: Use React for a polished frontend with query history and visualizations.
- **Voice Support**: Add speech-to-text (Whisper) and text-to-speech.
- **Scalability**: Use Pinecone for larger datasets or distributed FAISS for faster retrieval.

## Contributing
- Fork the repository and submit pull requests for bug fixes or features.
- Report issues via GitHub Issues.
- Follow code style guidelines (e.g., PEP 8).