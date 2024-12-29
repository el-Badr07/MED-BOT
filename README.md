# Medical Bot with Advanced RAG Pipeline 🏥 🤖

An intelligent medical information retrieval system that combines advanced RAG (Retrieval-Augmented Generation) pipeline with comprehensive medical document analysis capabilities.
## Overview

The **Medical Bot** is an intelligent and powerful healthcare solution designed to assist both medical professionals and patients by providing accurate, evidence-based medical information. It leverages the cutting-edge **Retrieval-Augmented Generation (RAG)** pipeline, which enhances traditional AI models by combining the power of document retrieval and generative language models. This hybrid approach allows the bot to access and retrieve relevant medical documents from vast datasets, analyze them, and generate responses tailored to user queries.

With this system, users can interact with the bot to receive answers grounded in the latest medical research, guidelines, and patient-specific data. It processes both structured data, such as medical records and diagnosis codes, and unstructured data, such as medical research articles, making it highly versatile and capable of supporting a wide variety of medical use cases.

The bot can be integrated into telemedicine platforms, used for patient education, and serve as a decision support tool for healthcare providers. By delivering accurate, context-aware responses in real time, it improves clinical decision-making, enhances patient engagement, and supports healthcare innovation.

This solution is designed to seamlessly integrate with existing healthcare infrastructures, ensuring that medical professionals and institutions can leverage it without significant changes to their workflows.


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

# Docker Image Documentation

## Overview

This Docker image provides a containerized version of the `medbot` application, ready to run in any environment that supports Docker.

The Docker image includes everything needed to run the app, including the necessary libraries and dependencies. This image is designed to work for anyone needing to deploy the app in a consistent and isolated environment.

## Prerequisites

Before using the image, make sure you have the following:

- Docker installed on your system.
- Access to Docker Hub or the ability to pull from a private registry.

## How to Pull the Image

To download or "pull" the image from Docker Hub, use the following command:

```bash
docker pull ion780/medbot:latest
```
Once the image is pulled, you can start a container from it. Here’s how to do it:

```bash
docker run -d -p 8501:8501 --name myapp-container ion780/medbot:latest
```
## Ollama

Ollama is a library built on top of the Hugging Face Transformers library, offering an easy way to implement Retriever-Augmented Generation (RAG) in projects. With a simple API, Ollama enables seamless integration of any retriever and generator model from the Hugging Face model hub. 

You can download Ollama from its [official website](https://ollama.com/).

---

## Getting started:
In order to run this app we need to install first our requirements dependencies using pip, to do that make sure to use the code below:
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

cd MED-BOT

# Run the application
streamlit run LLM.py.py
```

## 📚 Documentation

Comprehensive documentation is available at [medical-bot.readthedocs.io](https://medical-bot.readthedocs.io/)


Project Link: [https://github.com/el-Badr07/medical-bot]

## 🗺️ Feature improvements

- [ ] Add support for medical imaging analysis with vlms
- [ ] Implement some agents to help the retrieval and context refinement
- [ ] Generalise the pipeline to multiple fields
- [ ] Add audio input and multilanguage support
