import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA




from tavily import TavilyClient
client = TavilyClient(api_key="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")

##############################################################################
# Existing Functions (evaluate_context, enhance_context, create_rag_agent)
##############################################################################

def evaluate_context(context: str, query: str) -> bool:
    eval_template = """
    Analyze if this context is sufficient to answer the query.
    Context: {context}
    Query: {query}
    Return only 'SUFFICIENT' or 'INSUFFICIENT'
    """
    eval_prompt = PromptTemplate.from_template(eval_template)
    eval_chain = eval_prompt | ChatOllama(model="smollm") | StrOutputParser()
    result = eval_chain.invoke({"context": context, "query": query})
    return "SUFFICIENT" in result.upper()

def enhance_context(original_context: str, query: str) -> str:
    """Use Tavily Client's search functionality."""
    context = client.get_search_context(query)  # or modify the returned data as needed
    return f"{original_context}\n\n[Additional Information]\n{context}"

def create_rag_agent() -> AgentExecutor:
    """
    Replace this stub with your actual RAG (Retrieve-and-Generate) agent setup.
    Must return an AgentExecutor that takes {"input": ..., "context": ...}.
    """
    # Example pseudo-code:
    llm = ChatOllama(model="smollm")
    # tools = []
    #prompt_template = PromptTemplate.from_template("RAG prompt")
    # agent = create_react_agent(llm, tools, prompt_template)
    # return AgentExecutor(agent=agent, tools=tools, verbose=True)
    raise NotImplementedError("Replace with your actual RAG agent code.")

##############################################################################
# Convert Each Function into a Tool
##############################################################################

def evaluate_context_tool_fn(query: str) -> str:
    # ReAct Tools must return strings; we map True -> SUFFICIENT, False -> INSUFFICIENT
    is_sufficient = evaluate_context(query)
    return "SUFFICIENT" if is_sufficient else "INSUFFICIENT"

evaluate_context_tool = Tool(
    name="EvaluateContext",
    func=evaluate_context_tool_fn,
    description="Determines if the current context is enough to answer the query."
)

def enhance_context_tool_fn(data: str) -> str:
    # Merges old context with web data
    return enhance_context(data)

enhance_context_tool = Tool(
    name="EnhanceContext",
    func=enhance_context_tool_fn,
    description="Enhances context by performing a web search."
)

def rag_agent_tool_fn(query) -> str:
    # Runs your RAG agent on the context + query
    #rag_exec = create_rag_agent()
    llm= ChatOllama(model="smollm")
    response = llm.invoke({"input": query})
    return response

rag_agent_tool = Tool(
    name="ProcessRAG",
    func=rag_agent_tool_fn,
    description="produce the final answer when context is sufficient."
)

##############################################################################
# Orchestrator (ReAct) Prompt
##############################################################################

# The Orchestrator can do the following:
# 1. Evaluate context -> SUFFICIENT / INSUFFICIENT
# 2. If SUFFICIENT, just call ProcessRAG
# 3. If INSUFFICIENT, call EnhanceContext, then ProcessRAG
# Use typical ReAct “Thought/Action/Action Input/Observation” steps.
orchestrator_prompt = """
You are the Orchestrator Agent using ReAct. You have these tools:
{tool_names}

Tools detail:
{tools}



Your job:
1. Check if the provided context is sufficient for the query using EvaluateContext[query,context]
2. If the context is 'SUFFICIENT', invoke ProcessRAG.
3. If the context is 'INSUFFICIENT', first invoke EnhanceContext, then call ProcessRAG.
Use only these tools. Return the final answer in the format:

Final Answer: <your concise answer here>

Begin.

Question: {query}
Context: {context}
Thought:
Scratchpad:
{agent_scratchpad}
""".strip()

orchestrator_prompt1 = """
Use these tools:
{tool_names}

Tools detail:
{tools}

You are the Orchestrator. You must do the following:
1. First call EvaluateContext(query) with the question and context.
2. If EvaluateContext result is 'SUFFICIENT', call ProcessRAG.
3. If EvaluateContext result is 'INSUFFICIENT', call EnhanceContext, merge results, then call ProcessRAG.
Make sure to follow these steps exactly in your ReAct chain-of-thought.
Do NOT assume tool outputs; you must actually call them.

Use the following format repeatedly:
Thought: <what you are thinking>
Action: <tool name>
Action Input: <call the tool with the correspending arguments like: EvaluateContext(query,context) >
Observation: <tool response>
... (continue if more steps needed)
Thought: <final reasoning>
Final Answer: <answer to user>

Question: {query}
Context: {context}
Scratchpad:
{agent_scratchpad}
""".strip()

##############################################################################
# Build Orchestrator Agent
##############################################################################

  # Choose your orchestrator model
from langchain.prompts import PromptTemplate
from langchain import hub
prompt3= hub.pull("hwchase17/react")


orchestrator_prompt_template = PromptTemplate.from_template(orchestrator_prompt1)
tools = [evaluate_context_tool, enhance_context_tool, rag_agent_tool]
llm = ChatNVIDIA(
        model="nvidia/nemotron-4-340b-instruct",
        api_key="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", 
        temperature=0.2,
        top_p=0.7,
        max_tokens=4024,
        ).bind_tools(tools=tools)
orchestrator_agent = create_react_agent(llm, tools, orchestrator_prompt_template)
orchestrator_executor = AgentExecutor(
    agent=orchestrator_agent,
    tools=tools,
    verbose=True
)

##############################################################################
# Usage Example
##############################################################################



def process_query(query: str) -> str:
    context = 'to trat diabetes follow a healthy diet plan, engage in regular physical activity, maintain a healthy weight, and work closely with a healthcare provider to develop a personalized treatment plan, which may include medications such as insulin or metformin to control blood sugar levels.'
    response = orchestrator_executor.invoke({"query": query, "context": context})
    return response

if __name__ == "__main__":
    q = "How to treat diabetes?"
    final_answer = process_query(q)
    print("\nORCHESTRATED ANSWER:\n", final_answer)