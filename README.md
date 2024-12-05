# Medical Bot with Advanced RAG Pipeline 🏥 🤖

An intelligent medical information retrieval system that combines advanced RAG (Retrieval-Augmented Generation) pipeline with comprehensive medical document analysis capabilities.


## 🌟 Features

- **Advanced RAG Pipeline**
  - Context-aware information retrieval and chuncking
  - Hybrid Semantic search capabilities
  - Dynamic document ranking
  - Query routing

- **Medical Document Analysis**
  - Insight generation 
  - Diagnosis and recommendations
  - Metadata extraction and organization

**Retriever-Augmented Generation** (RAG) is a technique that enhances the quality and relevance of generated text by incorporating a retriever. The retriever selects the most pertinent context from external documents, which then informs the generation process. This approach is valuable for producing accurate and contextually relevant responses, as it provides the model with focused context from external sources.

---

## Building a RAG-Powered Chatbot

In this section, we demonstrate how to build a chatbot using RAG to answer questions based on a given context. We’ll use **Ollama models** and **embeddings** to create a simple Streamlit app capable of answering questions based on context ingested in a vector store database.

### Requirements 
In order to run this app we need to install first our requirements dependencies using pip, to do that make sure to use the code below:
```bash
$ pip install -r requirements.txt

```

### Steps:

1. **Ingesting Documents**: 
   - The `ingest.py` script processes documents from the `documents` folder. This script extracts text from documents, splits it into chunks, generates embeddings, and stores them in a Chroma vector database.
   
2. **Generating Answers**:
   - The `LLM.py` script retrieves answers using the ingested database. Optionally, it can answer queries based on a PDF uploaded by the user.

---



## Ollama

Ollama is a library built on top of the Hugging Face Transformers library, offering an easy way to implement Retriever-Augmented Generation (RAG) in projects. With a simple API, Ollama enables seamless integration of any retriever and generator model from the Hugging Face model hub. 

You can download Ollama from its [official website](https://ollama.com/).

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/el-Badr07/medical-bot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configurations

# Run the application
python medical_bot/main.py
```



## 📚 Documentation

Comprehensive documentation is available at [medical-bot.readthedocs.io](https://medical-bot.readthedocs.io/)




Project Link: [https://github.com/el-Badr07/medical-bot]

## 🗺️ Feature improvements

- [ ] Add support for medical imaging analysis
- [ ] Implement some agents to help the retrieval and context refinement
- [ ] Generalise the pipeline to multiple fields 
