import os
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
        calculator,
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
    



print(process_query('do you have knowledge in medical field and can you respond to some questions in it'))
