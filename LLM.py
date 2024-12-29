import streamlit as st
import os 
import PyPDF2
#from agent import process_query
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema import Document
from ingest import initialize_vector_store
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from typing import List
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from llmwhisper import interpret_json
import tempfile
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from agentscrewai import AgentSystem



def rag_fusion(question):
    # Step 1: Query Generation - generate multiple search queries related to the input question
    prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful medical assistant that generates multiple medical search queries based on a single input query. 
    Generate multiple search queries related to: {question} 
    Reply with the generated queries in the following format:
    [Generated Query 1]
    [Generated Query 2]
    [Generated Query 3]
    [Generated Query 4]
    Output (4 queries) no additional text""")
    
    def parse_queries_output(message):
        return message.content.split('\n')
    
    llm1 = ChatOllama(model="llama3.2")

    query_gen = prompt_rag_fusion | llm1 | parse_queries_output
    
    # Generate the multiple queries
    generated_queries = query_gen.invoke({"question": question})
    print(generated_queries)


    # Step 2: Retrieve documents for each generated query (assuming you have a retriever defined)
    # Batch retrieve documents based on the generated queries
    retriever_results = []
    db = initialize_vector_store()

    #retriever = db.similarity_search(question, k=2)
    # Replace the similarity_search_with_score with a cosine similarity search
    for query in generated_queries:
        results = db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        ).get_relevant_documents(query)
        retriever_results.append(results)
    #print(retriever_results.)
    # Step 3: Reciprocal Rank Fusion (RRF) on the retrieved results  
    def reciprocal_rank_fusion(results: list[list], k=5):
        """Reciprocal Rank Fusion on multiple lists of ranked documents."""
        fused_scores = {}
        documents = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                # Handle both Document objects and (Document, score) tuples
                if isinstance(doc, tuple):
                    doc_content = doc[0].page_content
                    actual_doc = doc[0]
                else:
                    doc_content = doc.page_content
                    actual_doc = doc
                    
                if doc_content not in fused_scores:
                    fused_scores[doc_content] = 0
                    documents[doc_content] = actual_doc
                fused_scores[doc_content] += 1 / (rank + k)
        
        reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
        return [documents[doc_str] for doc_str in reranked_doc_strs][:2]

    medical_template = ChatPromptTemplate.from_template("""You are an experienced and compassionate medical professional. You are great at answering medical questions, explaining symptoms, treatments, and diagnoses in a clear and empathetic way. You make sure to address the patient's concerns carefully. 
    Answer the following question based on:
    {context}
    Question: {question}
    """)
    biology_template = ChatPromptTemplate.from_template("""You are an expert in biology, with a deep understanding of topics ranging from molecular biology to ecology. You excel at explaining biological concepts in simple and engaging terms, ensuring clarity. When answering, you break down complex biological processes into understandable steps and use relatable examples.
    Answer the following question based on :
    {context}
    Question: {question}
    """)
    psychology_template = ChatPromptTemplate.from_template("""You are a knowledgeable psychologist, skilled in explaining mental health concepts, psychological theories, and behavioral science in a sensitive and thoughtful manner. You aim to provide clear insights into psychological conditions, therapies, and research findings. You offer advice with empathy and recognize the importance of consulting professionals when needed.
    Answer the following question based on:
    {context}
    Question: {question}
    """)

    # Step 4: Generate the final answer using the retrieved documents and original question
    embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")

   
    # Step 3: Apply Reciprocal Rank Fusion (RRF) on the merged results
    fused_docs = reciprocal_rank_fusion(retriever_results)

    # Step 4: Prepare the context and answer the question
    context = "\n".join([doc.page_content for doc in fused_docs])
    prompt_templates = [
        medical_template.invoke({"context": context, "question": question}),
        biology_template.invoke({"context": context, "question": question}),
        psychology_template.invoke({"context": context, "question": question}),
    ]

    query_router = QueryRouter(prompt_templates, embeddings)
    semantic_router = query_router.prompt_router(question)
    system = AgentSystem()
        
    # Process query with context
    result = system.process_query(question,prompt_templates[0] ,context)


    #final_answer = llm.invoke(semantic_router) 
    
    return result

def rag_fusion1(doc,question):
    # Step 1: Query Generation - generate multiple search queries related to the input question
    prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful assistant that generates multiple search queries based on a single input query. 
    Generate multiple search queries related to: {question} 
    Output (4 queries):""")
    
    def parse_queries_output(message):
        return message.content.split('\n')
    
    llm = ChatOllama(model="llama3.2")

    query_gen = prompt_rag_fusion | llm | parse_queries_output
    
    # Generate the multiple queries
    generated_queries = query_gen.invoke({"question": question})

    # Step 2: Retrieve documents for each generated query (assuming you have a retriever defined)
    # Batch retrieve documents based on the generated queries
    retriever_results = []
    db = initialize_vector_store()
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever=vectorstore.asimilarity_search_by_vector(k=4)
    #retriever = vectorstore.as_retriever(k=3)
    #retriever.get_relevant_documents()
    

    
    for query in generated_queries:
        retriever_results.append(retriever.invoke(query)) # Assuming 'retriever' is predefined
    
    # Step 3: Reciprocal Rank Fusion (RRF) on the retrieved results
    def reciprocal_rank_fusion(results: list[list], k=5):
        """Reciprocal Rank Fusion on multiple lists of ranked documents."""
        fused_scores = {}
        documents = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc.page_content
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                    documents[doc_str] = doc
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
        return [documents[doc_str] for doc_str in reranked_doc_strs][:3]
    
    # Apply RRF on the retrieved documents
    fused_docs = reciprocal_rank_fusion(retriever_results)
    #fused_docs1=merge_results(bm25_func(question),fused_docs)
    ####
    medical_template = ChatPromptTemplate.from_template("""You are an experienced and compassionate medical professional. You are great at answering medical questions, explaining symptoms, treatments, and diagnoses in a clear and empathetic way. You provide evidence-based information and make sure to address the patient's concerns carefully. If you're uncertain about a specific medical question, you acknowledge that and recommend consulting a healthcare provider. 
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    biology_template = ChatPromptTemplate.from_template("""You are an expert in biology, with a deep understanding of topics ranging from molecular biology to ecology. You excel at explaining biological concepts in simple and engaging terms, ensuring clarity. When answering, you break down complex biological processes into understandable steps and use relatable examples. If you're unsure about an answer, you clearly state that and encourage further exploration.
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    psychology_template = ChatPromptTemplate.from_template("""You are a knowledgeable psychologist, skilled in explaining mental health concepts, psychological theories, and behavioral science in a sensitive and thoughtful manner. You aim to provide clear insights into psychological conditions, therapies, and research findings. You offer advice with empathy and recognize the importance of consulting professionals when needed. If you are uncertain about a topic, you make sure to express that and suggest seeking expert guidance.
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)

    # Step 4: Generate the final answer using the retrieved documents and original question
    embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
    
    
    
    # Prepare the context (retrieved and fused documents)
    context = "\n".join([doc.page_content for doc in fused_docs])
    prompt_templates = [medical_template.invoke({"context": context, "question": question}), biology_template.invoke({"context": context, "question": question}),psychology_template.invoke({"context": context, "question": question})]

    # Instantiate the QueryRouter with the prompt templates and embeddings
    query_router = QueryRouter(prompt_templates, embeddings)

    # Create the chain that handles the full query routing and answering process
    semantic_router = query_router.prompt_router(question)

    # Example query
    
    
    # Invoke the chain with the query and get the answer
    #result = semantic_router.invoke({"query": question})
    # Format the prompt with the context and the question
    #formatted_prompt = prompt.invoke({"context": context, "question": question})
    
    # Generate the final answer using the LLM
    final_answer = llm.invoke(semantic_router)
    
    return final_answer

def rag_fusion2(doc,question):
    # Step 1: Query Generation - generate multiple search queries related to the input question
    prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful medical assistant that generates multiple medical search queries based on a single input query. 
        Generate multiple search queries related to: {question} 
        Reply with the generated queries in the following format:
        [Generated Query 1]
        [Generated Query 2]
        [Generated Query 3]
        [Generated Query 4]
        Output (4 queries) no additional text""")
    
    
    def parse_queries_output(message):
        return message.content.split('\n')
    
    llm = ChatOllama(model="llama3.2")

    query_gen = prompt_rag_fusion | llm | parse_queries_output
    
    # Generate the multiple queries
    generated_queries = query_gen.invoke({"question": question})
    print(len(generated_queries))

    # Step 2: Retrieve documents for each generated query (assuming you have a retriever defined)
    # Batch retrieve documents based on the generated queries
    retriever_results = []
    
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=900, chunk_overlap=150)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )

    retriever1 = vectorstore.as_retriever(k=4,search_type="similarity")

    bm25_retriever = BM25Retriever.from_documents(doc_splits)

    # Combine them into an EnsembleRetriever
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever1],
        retriever_weights={"BM25Retriever": 0.4, "ChromaRetriever": 0.6},
    )
    #retriever.get_relevant_documents()

    for query in generated_queries:
        retriever_results.append(retriever.get_relevant_documents(query)[:4]
)
    

    
    
    # Step 3: Reciprocal Rank Fusion (RRF) on the retrieved results
    def reciprocal_rank_fusion(results: list[list], k=6):
        """Reciprocal Rank Fusion on multiple lists of ranked documents."""
        fused_scores = {}
        documents = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc.page_content
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                    documents[doc_str] = doc
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
        return [documents[doc_str] for doc_str in reranked_doc_strs][:3]
    
    # Step 2: Retrieve documents for each generated query and merge results
    medical_template = ChatPromptTemplate.from_template("""You are an experienced and compassionate medical professional. You are great at answering medical questions, explaining symptoms, treatments, and diagnoses in a clear and empathetic way. You provide evidence-based information and make sure to address the patient's concerns carefully. If you're uncertain about a specific medical question, you acknowledge that and recommend consulting a healthcare provider. 
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    biology_template = ChatPromptTemplate.from_template("""You are an expert in biology, with a deep understanding of topics ranging from molecular biology to ecology. You excel at explaining biological concepts in simple and engaging terms, ensuring clarity. When answering, you break down complex biological processes into understandable steps and use relatable examples. If you're unsure about an answer, you clearly state that and encourage further exploration.
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    psychology_template = ChatPromptTemplate.from_template("""You are a knowledgeable psychologist, skilled in explaining mental health concepts, psychological theories, and behavioral science in a sensitive and thoughtful manner. You aim to provide clear insights into psychological conditions, therapies, and research findings. You offer advice with empathy and recognize the importance of consulting professionals when needed. If you are uncertain about a topic, you make sure to express that and suggest seeking expert guidance.
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)

    # Step 4: Generate the final answer using the retrieved documents and original question
    embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")

   
    # Step 3: Apply Reciprocal Rank Fusion (RRF) on the merged results
    fused_docs = reciprocal_rank_fusion(retriever_results)

    # Step 4: Prepare the context and answer the question
    context = "\n".join([doc.page_content for doc in fused_docs])
    prompt_templates = [
        medical_template.invoke({"context": context, "question": question}),
        biology_template.invoke({"context": context, "question": question}),
        psychology_template.invoke({"context": context, "question": question}),
    ]

    query_router = QueryRouter(prompt_templates, embeddings)
    semantic_router = query_router.prompt_router(question)
    system = AgentSystem()
        
    # Process query with context
    result = system.process_query(question,prompt_templates[0] ,context)


    return result

    
    

def retrieve_combined_results(query, retriever,bm25_weight=0.4, chroma_weight=0.6, top_k=5):
    """
    Retrieves documents for a query using BM25 and Chroma, and merges the results.
    Args:
        query (str): The input query.
        bm25_weight (float): Weight for BM25 scores.
        chroma_weight (float): Weight for Chroma scores.
        top_k (int): Number of top results to return.
    Returns:
        list: Merged results from BM25 and Chroma.
    """
    # Retrieve BM25 results
    #bm25_results = bm25_func(query)  # Returns list of (doc_id, score)
    
    # Retrieve Chroma results
    chroma_results = retriever.invoke(query)  # Assumes retriever is already defined
    
    # Merge BM25 and Chroma results
    #merged_results = merge_results(bm25_results, chroma_results, bm25_weight, chroma_weight, top_k)
    
    # Convert merged results to a format compatible with the reciprocal ranker
    #combined_results = [
        #Document(page_content=doc_id, metadata={"score": score})
        #for doc_id, score in merged_results]
    
    #return combined_results

# function to read the pdf file
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    all_page_text = ""
    for page in pdfReader.pages:
        all_page_text += page.extract_text() + "\n"
    return all_page_text

def merge_results(bm25_results, chroma_results, bm25_weight=0.4, chroma_weight=0.6, top_k=5):
    """
    Merges results from BM25 and Chroma, then selects the top-k results.
    Args:
        bm25_results (list): List of (doc_id, bm25_score) tuples from BM25.
        chroma_results (dict): Results from Chroma containing documents and distances.
        bm25_weight (float): Weight for BM25 scores.
        chroma_weight (float): Weight for Chroma scores.
        top_k (int): Number of top results to return.
    Returns:
        list: Top-k results sorted by combined score.
    """
    # Normalize BM25 scores
    bm25_max_score = max(bm25_results, key=lambda x: x[1])[1] if bm25_results else 1
    normalized_bm25 = {idx: score / bm25_max_score for idx, score in bm25_results}

    # Normalize Chroma distances (convert to similarity scores)
    chroma_docs = chroma_results["documents"]
    chroma_distances = chroma_results["distances"]
    chroma_max_distance = max(chroma_distances) if chroma_distances else 1
    normalized_chroma = {
        chroma_docs[i]: 1 - (chroma_distances[i] / chroma_max_distance)
        for i in range(len(chroma_docs))
    }

    # Combine scores
    combined_scores = {}
    for doc_id, bm25_score in normalized_bm25.items():
        combined_scores[doc_id] = bm25_weight * bm25_score

    for doc in normalized_chroma.keys():
        if doc in combined_scores:
            combined_scores[doc] += chroma_weight * normalized_chroma[doc]
        else:
            combined_scores[doc] = chroma_weight * normalized_chroma[doc]

    # Sort by score and return top-k results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def retrieve_from_db(question):
    # get the model
    model = ChatOllama(model="llama3.2")
    # initialize the vector store
    db = initialize_vector_store()

    retriever = db.similarity_search(question, k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )

    return after_rag_chain.invoke({"context": retriever, "question": question})


def retriever(doc, question):
    model_local = ChatOllama(model="llama3.2")
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever = vectorstore.as_retriever(k=3)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)

class QueryRouter1:
    def __init__(self, prompt_templates: List[str], embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")):
        self.prompt_templates = prompt_templates
        self.embeddings = embeddings
        # Embed the prompt templates at the initialization step
        self.prompt_embeddings = self.embeddings.embed_documents(self.prompt_templates)
    
    def route_question(self, query: str) -> PromptTemplate:
        """Route the question to the most relevant prompt template."""
        # Embed the query to compare with prompt templates
        query_embedding = self.embeddings.embed_query(query)
        
        # Compute cosine similarity between the query and the prompt templates
        similarity_scores = cosine_similarity([query_embedding], self.prompt_embeddings)[0]
        
        # Select the most similar prompt template based on the similarity scores
        most_similar_index = similarity_scores.argmax()
        most_similar_prompt = self.prompt_templates[most_similar_index]
        
        # Return a PromptTemplate based on the most similar prompt
        return PromptTemplate.from_template(most_similar_prompt)
    
    def create_chain(self) -> chain:
        """Create the chain that will route and process the query."""
        @chain
        def prompt_router(query: str):
            # Route the question to the most appropriate prompt template
            selected_prompt = self.route_question(query)
            return selected_prompt
        
        # Define the full pipeline with OpenAI and output parser
        semantic_router = (
            prompt_router
            
        )
        
        return semantic_router

test='''
st.title("Medical Chatbot")
st.write("This is chatbot developed to answer questions in the medical field.")
file = st.file_uploader("Upload a PDF file", type=["pdf"])
if file:
    doc = read_pdf(file)
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        answer = rag_fusion2(doc, question)
        st.write(answer)
else:
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        answer = rag_fusion(question)
        st.write(answer)

'''

class QueryRouter:
    def __init__(self, prompt_templates: List[str],embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")):
        self.prompt_templates = prompt_templates
        self.embeddings = embeddings
        # Embed the prompt templates at the initialization step
        self.prompt_embeddings = self.embeddings.embed_documents(self.prompt_templates)
    
    def route_question(self, query: str) -> PromptTemplate:
        """Route the question to the most relevant prompt template."""
        # Embed the query to compare with prompt templates
        query_embedding = self.embeddings.embed_query(query)
        
        # Compute cosine similarity between the query and the prompt templates
        similarity_scores = cosine_similarity([query_embedding], self.prompt_embeddings)[0]
        
        # Select the most similar prompt template based on the similarity scores
        most_similar_index = similarity_scores.argmax()
        most_similar_prompt = self.prompt_templates[most_similar_index]
        
        # Return a PromptTemplate based on the most similar prompt
        return most_similar_prompt
    
   
    def prompt_router(self,query: str):
            # Route the question to the most appropriate prompt template
            selected_prompt = self.route_question(query)
            return selected_prompt
#megaparse



# Initialize session state for conversation history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a page:", ["Chatbot", "Medical Report Analysis"])

# Chatbot Page
if option == "Chatbot":
    st.title("Medical Chatbot")
    st.write("This is a chatbot developed to answer questions in the medical field.")


    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file:
        doc = read_pdf(file)
        question = st.text_input("Ask a question")
        if st.button("Ask"):
            answer = rag_fusion2(doc, question)
            # Add to chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.write(answer)
    else:
        question = st.text_input("Ask a question")
        if st.button("Ask"):
            answer = rag_fusion(question)
            # Add to chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.write(answer)

# Medical Report Analysis Page
elif option == "Medical Report Analysis":
    st.title("Medical Report Analysis")
    st.write("Upload a medical report to analyze.")

    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        if st.button("Analyze"):
            answer = interpret_json(file=temp_file_path)
            st.write(answer)
    else:
        st.write("you must Upload a medical report to analyze.")

# Add a clear history button in the sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
