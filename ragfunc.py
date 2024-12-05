from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document


def retriever(doc):
    model_local = ChatOllama(model="mistral")
    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever = vectorstore.as_retriever(k=2)
    return retriever

def execute_subquery_agent(question, retriever):
    # Step 1: Decompose the main question into sub-queries
    decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. 
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. 
    Generate multiple search queries related to: {question} 
    Output (3 queries):"""
    
    # Set up the prompt template for decomposition
    decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)

    # Initialize the LLM for generating sub-queries
    llm_decompose = ChatOpenAI(temperature=0)

    # Create the chain to generate sub-queries from the main question
    generate_queries_decomposition = decomposition_prompt | llm_decompose | StrOutputParser() | (lambda x: x.split("\n"))
    
    # Generate sub-queries from the input question
    questions = generate_queries_decomposition.invoke({"question": question})
    
    # Function to format Q&A pair
    def format_qa_pair(question, answer):
        """Format Q and A pair into a standard string."""
        formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    # Create RAG chain to retrieve context and answer each sub-query
    def create_rag_chain(question, q_a_pairs, retriever):
        """Create a RAG chain to retrieve context and generate answers."""
        context_prompt = """Here is the question you need to answer:

        \n --- \n {question} \n --- \n

        Here is any available background question + answer pairs:

        \n --- \n {q_a_pairs} \n --- \n

        Here is additional context relevant to the question:

        \n --- \n {context} \n --- \n

        Use the above context and any background question + answer pairs to answer the question: \n {question}
        """

        # Set up the prompt for context + question answering
        decomposition_prompt = ChatPromptTemplate.from_template(context_prompt)
        
        # Define the LLM used for generating the answers
        llm_answer = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Define the RAG chain
        rag_chain = (
            {"context": itemgetter("question") | retriever,
             "question": itemgetter("question"),
             "q_a_pairs": itemgetter("q_a_pairs")}
            | decomposition_prompt
            | llm_answer
            | StrOutputParser()
        )
        
        return rag_chain

    # Step 2: Process each sub-query and generate answers
    q_a_pairs = ""
    for q in questions:
        # Create RAG chain for each sub-query
        rag_chain = create_rag_chain(q, q_a_pairs, retriever)

        # Get the answer for the sub-query
        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})

        # Format and store the question-answer pair
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

    # Return the combined question-answer pairs
    return q_a_pairs

# Example usage
"""""
question = "What are the main components of an LLM-powered autonomous agent system?"
retriever = retriever() # Replace with your actual retriever (e.g., vector search, document search, etc.)

# Execute the subquery agent
q_a_pairs = execute_subquery_agent(question, retriever)
print(q_a_pairs)
"""""

####
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.runnables import RunnableLambda

# Define the function to encapsulate the entire pipeline
def stepback(question):
    # Few-shot examples
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]

    # Example prompt template
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # Few-shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # Step-back question generation prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ]
    )

    # Step-back question generation chain
    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

    # Response prompt template
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""

    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    # Full pipeline chain with context retrieval and answer generation
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,  # Assumes retriever is predefined
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,  # Assumes retriever is predefined
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    # Invoke the chain with the provided question
    return chain.invoke({"question": question})

# Example usage of the function
"""""
question = "What is task decomposition for LLM agents?"
response = generate_answer(question)
print(response)
"""""

###


# Define the function to encapsulate the entire process
def hyde(question):
    # Step 1: HyDE document generation (to create a scientific paper passage)
    template_hyde = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    
    prompt_hyde = ChatPromptTemplate.from_template(template_hyde)
    
    # Chain for generating document (HyDE)
    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
    )
    
    # Generate the passage for the question
    generated_docs = generate_docs_for_retrieval.invoke({"question": question})

    # Step 2: Retrieve context (this step assumes you have a predefined retriever function)
    retrieval_chain = generate_docs_for_retrieval | retriever  # Assumes 'retriever' is defined elsewhere
    
    # Retrieve relevant documents based on the generated context (HyDE)
    retrieved_docs = retrieval_chain.invoke({"question": question})

    # Step 3: RAG (Retrieve and Generate) - use retrieved context to generate an answer
    rag_template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    
    prompt_rag = ChatPromptTemplate.from_template(rag_template)
    
    # Chain for generating the final RAG response
    final_rag_chain = (
        prompt_rag
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    
    # Final response generation based on retrieved documents
    final_answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
    
    return final_answer

# Example usage of the function
"""""
question = "What is task decomposition for LLM agents?"
response = generate_rag_answer(question)
print(response)
"""""
####
# Define the function to encapsulate the entire RAG-Fusion process
def rag_fusion(question):
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
    for query in generated_queries:
        retriever_results.append(retriever.retrieve(query))  # Assuming 'retriever' is predefined
    
    # Step 3: Reciprocal Rank Fusion (RRF) on the retrieved results
    def reciprocal_rank_fusion(results: list[list], k=10):
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
        return [documents[doc_str] for doc_str in reranked_doc_strs]
    
    # Apply RRF on the retrieved documents
    fused_docs = reciprocal_rank_fusion(retriever_results)
    
    # Step 4: Generate the final answer using the retrieved documents and original question
    prompt = ChatPromptTemplate.from_template("""Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    
    # Prepare the context (retrieved and fused documents)
    context = "\n".join([doc.page_content for doc in fused_docs])
    
    # Format the prompt with the context and the question
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    
    # Generate the final answer using the LLM
    final_answer = llm.invoke(formatted_prompt)
    
    return final_answer

# Example usage of the function
"""""
question = "What is task decomposition for LLM agents?"
response = generate_rag_fusion_answer(question)
print(response)
"""""
###

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field


# Data model for routing the queries
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question",
    )

# Define the LLM and its output structure
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# System message for routing logic
system = """You are an expert at routing a user question to the appropriate data source.
Based on what the question is referring to, route it to the relevant data source."""

# Prompt for generating routing queries
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Router function: Process the question and route it to the correct data source
class QueryRouter:
    def __init__(self, prompt, structured_llm):
        self.router = prompt | structured_llm
    
    def route_question(self, question: str) -> str:
        """Route the question based on its content."""
        # Invoke the router and get the datasource
        result = self.router.invoke({"question": question})
        return result.datasource

    def choose_route(self, result: RouteQuery) -> str:
        """Choose the correct chain based on the datasource."""
        if "python_docs" in result.datasource.lower():
            # Logic for routing to Python docs chain
            return "chain for python_docs"
        elif "js_docs" in result.datasource.lower():
            # Logic for routing to JS docs chain
            return "chain for js_docs"
        else:
            return "No matching datasource found"

    def full_routing_chain(self, question: str) -> str:
        """Complete chain for routing and selecting the relevant data source."""
        result = self.route_question(question)
        return self.choose_route(result)

# Example usage
router = QueryRouter(prompt, structured_llm)

# Example question
question = """Why doesn't the following code work:
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

# Use the full routing chain to get the selected data source chain
full_chain_result = router.full_routing_chain(question)
print(full_chain_result)

###
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List

class QueryRouter:
    def __init__(self, prompt_templates: List[str], embeddings: OpenAIEmbeddings):
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
            | ChatOpenAI()
            | StrOutputParser()
        )
        
        return semantic_router

# Example usage

if __name__ == "__main2__":
    # Define the prompt templates for physics and math
    physics_template = """You are a very smart physics professor. You are great at answering questions about physics in a concise and easy to understand manner. When you don't know the answer to a question you admit that you don't know.
    Here is a question:
    {query}"""

    math_template = """You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.
    Here is a question:
    {query}"""

    # Initialize the embeddings and templates
    embeddings = OpenAIEmbeddings()
    prompt_templates = [physics_template, math_template]

    # Instantiate the QueryRouter with the prompt templates and embeddings
    query_router = QueryRouter(prompt_templates, embeddings)

    # Create the chain that handles the full query routing and answering process
    semantic_router = query_router.create_chain()

    # Example query
    question = "How does Newton's second law explain the motion of objects?"
    
    # Invoke the chain with the query and get the answer
    result = semantic_router.invoke({"query": question})
    print(result)

    ###
