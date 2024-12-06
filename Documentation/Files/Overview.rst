Medical Bot with Advanced RAG Pipeline 🏥 🤖
==========================================

An intelligent medical information retrieval system that combines advanced RAG (Retrieval-Augmented Generation) pipeline with comprehensive medical document analysis capabilities.

🌟 Features
-----------

- **Advanced RAG Pipeline**
  - Context-aware information retrieval and chunking
  - Hybrid Semantic search capabilities
  - Dynamic document ranking
  - Query routing

- **Medical Document Analysis**
  - Insight generation
  - Diagnosis and recommendations
  - Metadata extraction and organization

**Retriever-Augmented Generation** (RAG) is a technique that enhances the quality and relevance of generated text by incorporating a retriever. The retriever selects the most pertinent context from external documents, which then informs the generation process. This approach is valuable for producing accurate and contextually relevant responses, as it provides the model with focused context from external sources.

---

Building a RAG-Powered Chatbot
------------------------------

In this section, we demonstrate how to build a chatbot using RAG to answer questions based on a given context. We’ll use **Ollama models** and **embeddings** to create a simple Streamlit app capable of answering questions based on context ingested in a vector store database.

### Requirements

In order to run this app, we need to install the required dependencies using pip. To do that, make sure to use the code below:

```bash
$ pip install -r requirements.txt
