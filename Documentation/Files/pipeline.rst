Medical Bot Pipeline Architecture 
=======================

This pipeline is designed to efficiently process user queries in the context of a
**Medical Bot** powered by an advanced **Retrieval-Augmented Generation (RAG)**
pipeline. The goal of this system is to provide accurate, context-aware, and
domain-specific responses (Medical, Biology, and Psychology) by combining
state-of-the-art retrieval and language generation techniques.

Pipeline Overview
=================

1. **User Query Input**:
   The process begins when a user submits a query related to medical, biological,
   or psychological topics. This query is the starting point for the entire pipeline.

2. **Multi-Query Generation**:
   The initial user query is passed through a **Multi-Query Generation Chain**.
   This process generates several variations of the original query to capture
   different phrasings or angles of the user's intent. This increases the diversity
   of results that can be retrieved in the subsequent steps.
   
   Example:
   - "What are the symptoms of a stroke?"
   - Variations: "What are the signs of a stroke?", "How do you recognize a stroke?"

3. **Context Retrieval using Ensemble Retrievers**:
   The system uses an **Ensemble Retriever** combining **BM25** and **Chroma**
   retrieval methods to retrieve relevant context.
   
   - **BM25** (Best Matching 25): A classical text retrieval algorithm based on
     term frequency and inverse document frequency (TF-IDF).
   - **Chroma**: An embedding-based retrieval method that uses vector search to
     retrieve context by comparing semantic similarity of the query and document
     embeddings.
   
   The retrieved contexts from both BM25 and Chroma are aggregated to form a
   rich pool of relevant information.

4. **Reciprocal Ranking**:
   The retrieved contexts are passed through a **Reciprocal Ranker**. This
   componenttechnique to combine the results from multiple retrieval sources and rank them according to their relevance to the user query. This ensures that the most pertinent and valuable documents are prioritized for the subsequent response generation step.

5. **Semantic Router**:
   At this stage, the system performs **Semantic Routing** to determine which
   domain (Medical, Biology, or Psychology) the query belongs to.
   
   - Using a **semantic classifier**, the system routes the query and its context
     to the appropriate prompt template based on the field it belongs to. This
     ensures that the query is answered by leveraging domain-specific knowledge.
   
   Example: 
   - If the query is related to disease symptoms, it might be routed to the
     **Medical** field.
   - If the query is related to brain functionality, it might be routed to
     **Psychology**.

6. **LLM (Large Language Model) Response Generation**:
   Finally, the routed query and the top-ranked context are passed to a **Large
   Language Model (LLM)** to generate a coherent and accurate response.
   
   The LLM synthesizes the user query with the retrieved context, and based on
   the semantic routing, it generates a contextually appropriate answer for the
   field (Medical, Biology, or Psychology).
   
   The response is generated with a focus on relevance and accuracy, ensuring
   the output is as informative as possible.

7. **Output**:
   The generated response is returned to the user as the final output. This
   response addresses the user’s query, informed by high-quality, relevant
   context and domain-specific knowledge.

Flow Summary
============

1. **User Query** → 2. **Multi-Query Generation** → 3. **Ensemble Context Retrieval (BM25 + Chroma)**
→ 4. **Reciprocal Ranking** → 5. **Semantic Routing (Medical, Biology, Psychology)** →
6. **LLM Response Generation** → 7. **Final Answer to User**.

Key Components
==============

1. **Multi-Query Generation Chain**:
   Expands the user query to capture different phrasings, helping to gather
   diverse and relevant context.

2. **Ensemble Retriever (BM25 + Chroma)**:
   Uses both traditional keyword-based retrieval (BM25) and modern semantic
   search (Chroma) to gather context, ensuring that the system can handle both
   keyword-based and semantic queries.

3. **Reciprocal Ranking**:
   Ranks the retrieved contexts based on relevance to the user’s query, ensuring
   only the most relevant information is used in the response.

4. **Semantic Router**:
   Determines the appropriate domain (Medical, Biology, or Psychology) based
   on the user query, ensuring that the response is routed to the correct prompt.

5. **LLM Response Generation**:
   Leverages large language models to synthesize a response based on the user
   query and the relevant context.



