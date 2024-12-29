from crewai import Agent, Task, Crew
from typing import Dict, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from crewai import LLM
from tavily import TavilyClient
client = TavilyClient(api_key="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")


class AgentSystem:
    def __init__(self):
        self.llm1 =LLM(model="ollama/smollm:latest",base_url="http://localhost:11434",)
        self.llm=LLM(model="nvidia_nim/meta/llama-3.1-70b-instruct",temperature=0.2, max_tokens=50000,api_key="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")

        self.search_tool = self.tavily_search
    @tool
    def tavily_search(query: str) -> str:
        """Use Tavily Client's search functionality."""
        context = client.get_search_context(query)
        return context  # or modify the returned data as needed

    def create_orchestrator_agent(self) -> Agent:
        return Agent(
            role='Orchestrator',
            goal='Coordinate query processing and delegate tasks',
            backstory='''Experienced coordinator that manages information flow and decides 
                        when additional context is needed
                        For the search tool use it only if the response from context evaluator agent is 'INSUFFICIENT' ''',
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

    def create_context_evaluator(self) -> Agent:
        return Agent(
            role='Context Evaluator',
            goal='Evaluate if provided context is sufficient for answering query',
            backstory='''Expert at analyzing context completeness and relevance to queries''',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_response_agent(self) -> Agent:
        return Agent(
            role='Response Generator',
            goal='Generate comprehensive and accurate responses',
            backstory='''Specialized in creating clear, well-structured answers 
                        based on provided context''',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def process_query(self, query: str, prompt: str, initial_context: str = "") -> str:
        # Create agents
        orchestrator = self.create_orchestrator_agent()
        evaluator = self.create_context_evaluator()
        responder = self.create_response_agent()

        # Define tasks
        evaluation_task = Task(
            description=f"""Evaluate if this context is sufficient to answer the query or if a websearch is needed.
                          Query: {query}
                          Context: {initial_context}
                          """,
            expected_output="Return only: 'SUFFICIENT' or 'INSUFFICIENT' without further explenation",  # Add this line

            agent=evaluator
        )

        search_task = Task(
            description=f"""Only perform this task if the context is insufficient.
            Perform web search to gather additional context for:
                          Query: {query}""",
            expected_output="Return: additional context gathered from web search: {web_search_results}",
            agent=orchestrator
        )

        response_task = Task(
            description=f"""prompt:{prompt}
                          web_search_results: {{web_search_results}}
                          """,
            expected_output="Return: comprehensive response to the query and mention if:{web_search_results} content were used",
            agent=responder
        )

        # Create crew with workflow
        crew = Crew(
            agents=[orchestrator, evaluator, responder],
            tasks=[evaluation_task, search_task, response_task],
            verbose=True
        )

        result = crew.kickoff()
        return result

""" def main():
    system = AgentSystem()
    query = "What are the key principles of quantum computing?"
    context = "The key principles of quantum computing are rooted in the mathematical framework of Hilbert spaces, enabling the description of qubits, quantum gates, entanglement, and quantum measurement. The first principle of quantum computing is the concept of superposition, where a qubit can exist in multiple states simultaneously, represented by a complex vector in a high-dimensional space. Quantum computing relies on the principles of superposition and entanglement to perform operations on multiple states simultaneously. The principles of quantum computing also rely on the concept of entanglement, which allows for the connection of qubits in such a way that the state of one qubit is instantaneously affected by the state of the other. Additionally, qubits can represent 0, 1, or a state that is partly 0 and partly 1 simultaneously, enabling quantum computations. These principles enable quantum computers to explore many possibilities at once and solve complex problems faster than classical computers. Quantum computing makes use of quantum phenomena, such as quantum bits, superposition, and entanglement to perform data operations, and has the potential to revolutionize industries and solve problems beyond the reach of classical systems."
    result = system.process_query(query, context)
    print(f"Final Response: {result}")

if __name__ == "__main__":
    main() """