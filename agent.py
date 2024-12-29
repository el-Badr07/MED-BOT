import ssl

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context

import os
import urllib3
import requests

# Suppress only the single warning from urllib3 needed.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.verify = False

requests.sessions.Session = lambda: session
from typing import List
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from operator import itemgetter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain import hub


# ...existing code...


# ...existing code...



from langchain_nvidia_ai_endpoints import ChatNVIDIA




from tavily import TavilyClient
client = TavilyClient(api_key="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")

# Step 2. Executing a simple search query




# First, let's create our vector stores for Collection A and B
def create_vector_store(texts: List[str], collection_name: str):
    embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
    vector_store = Chroma.from_texts(
        texts,
        embeddings,
        collection_name=collection_name
    )
    return vector_store

# Create vector search tools
@tool
def search_collection_a(query: str) -> str:
    """Search in Collection A for relevant information."""
    # Initialize vector store A (you would normally do this once at startup)
    collection_a = create_vector_store(
        ["sample text for collection A"], # Replace with your actual texts
        "collection_a"
    )
    results = collection_a.similarity_search(query, k=1)
    return results[0].page_content

@tool
def search_collection_b(query: str) -> str:
    """Search in Collection B for relevant information."""
    # Initialize vector store B (you would normally do this once at startup)
    collection_b = create_vector_store(
        ["sample text for collection B"], # Replace with your actual texts
        "collection_b"
    )
    results = collection_b.similarity_search(query, k=1)
    return results[0].page_content

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Create the agent with tools

@tool
def tavily_search(query: str) -> str:
    """Use Tavily Client's search functionality."""
    context = client.get_search_context(query)
    return context  # or modify the returned data as needed

def create_rag_agent():
    # Initialize the language model
    

    # Initialize tools
    tools = [
        #search_collection_a,
        #search_collection_b,
        #calculator,
        tavily_search
    ]
    #llm1 = ChatOllama(model="llama3.2",tools=tools)
    llm = ChatNVIDIA(
        model="nvidia/nemotron-4-340b-instruct",
        api_key="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", 
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        ).bind_tools(tools=[
        #search_collection_a,
        #search_collection_b,
        calculator,
        tavily_search
    ])
    # Create the agent prompt
    

    
    template1 = '''Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat 3 times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}'''
    prompt1 = PromptTemplate.from_template(template1)

    template = """You are an intelligent agent that carefully analyzes queries and provided context before deciding on actions.

            CONTEXT EVALUATION PROCESS:
            1. First, always analyze the provided context in relation to the query note that the context will be provided beforee query
            2. Determine if the context is:
            - Complete: Contains all necessary information
            - Relevant: Directly addresses the query
            - Current: Not outdated for time-sensitive queries
            - Specific: Detailed enough to answer the query

            Available tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Context: the provided context you must analyze first
            Thought: always analyze the context first and think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat 1 time)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Remember:
            - Don't web search if context is sufficient
            - Cite whether information comes from context or web search
            - Use step-by-step reasoning for calculations

            Question: {input}
            Thought:{agent_scratchpad}"""

    prompt2 = PromptTemplate.from_template(template)
    prompt= hub.pull("hwchase17/react")


    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor

def evaluate_context_sufficiency(context: str, query: str) -> bool:
    """Evaluate if the context is sufficient to answer the query."""
    
    evaluation_template = """
    You are a context evaluation agent. Analyze if the given context is sufficient to answer the query.
    
    Context: {context}
    Query: {query}
    
    Instructions:
    1. Analyze the relevance of context to the query
    2. Check if context contains enough details
    3. Determine if web search would be neccessary to answer the query or not
    
    Respond with either 'SUFFICIENT' or 'INSUFFICIENT'
    
    Thought: Let me analyze the context and query...
    Decision:"""
    
    evaluation_prompt = PromptTemplate.from_template(evaluation_template)
    
    llm = ChatOllama(model="mistral")
    chain = evaluation_prompt | llm | StrOutputParser()
    
    result = chain.invoke({"context": context,
        "query": query
    })
    
    return "SUFFICIENT" in result.upper()


def evaluate_context(context: str, query: str) -> bool:
    eval_template = """
    Analyze if this context is sufficient to answer the query.
    Context: {context}
    Query: {query}
    Return only 'SUFFICIENT' or 'INSUFFICIENT'
    """
    
    eval_prompt = PromptTemplate.from_template(eval_template)
    eval_chain = eval_prompt | ChatOllama(model="mistral") | StrOutputParser()
    result = eval_chain.invoke({"context": context, "query": query})
    return "SUFFICIENT" in result.upper()

def enhance_context(original_context: str, query: str) -> str:
    search_tool = DuckDuckGoSearchRun()
    search_results = search_tool.run(query)
    return f"{original_context}\n\nAdditional Information:\n{search_results}"

def process_query(query: str) -> str:
    # Initialize agent
    agent = create_rag_agent()
    
    # Get initial context
    context = get_initial_context(query)  # Your existing context retrieval
    
    # Evaluate context sufficiency
    if not evaluate_context(context, query):
        context = enhance_context(context, query)
    
    # Process with enhanced context
    result = agent.invoke({
        "input": query,
        "context": context
    })
    
    return result["output"]
# Example usage function
def process_query(query: str):
    agent = create_rag_agent()
    response = agent.invoke({"input": query})
    return response

# Example error handling wrapper
def safe_query_processing(query: str):
    try:
        return process_query(query)
    except Exception as e:
        return {
            "output": f"An error occurred while processing your query: {str(e)}",
            "error": True
        }
    
def process_query2():
    llm = ChatNVIDIA(
        model="nvidia/nemotron-4-340b-instruct",
        api_key="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", 
        temperature=0.2,
        top_p=0.7,
        max_tokens=3024,
        )
    medical_template = ChatPromptTemplate.from_template("""You are an experienced and compassionate medical professional. You are great at answering medical questions, explaining symptoms, treatments, and diagnoses in a clear and empathetic way. You provide evidence-based information and make sure to address the patient's concerns carefully. If you're uncertain about a specific medical question, you acknowledge that and recommend consulting a healthcare provider. 
    Answer the following question based on this context:
    {context}
    Question: {question}
    """)
    formated=medical_template.invoke({"context": "no context", "question": "What is the treatment for diabetes?"})
    response = llm.invoke(formated)
    return response
    



#print(process_query('Real-world diagnostic, referral, and treatment patterns in early Alzheimer’s disease'))
#print(process_query2())

from agentscrewai import AgentSystem
def process_medical_query(query: str, context: str = "") -> str:
    try:
        # Initialize the agent system
        system = AgentSystem()
        
        # Process query with context
        result = system.process_query(query, context)
        
        return result
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return "Error processing your request"

def main():
    # Example usage
    query = "What are the symptoms of diabetes?"
    context = """Diabetes is a metabolic disease that causes high blood sugar. 
                 The hormone insulin moves sugar from the blood into your cells 
                 to be stored or used for energy."""
    
    result = process_medical_query(query, context)
    print(f"Response: {result}")

if __name__ == "__main__":
    main()
