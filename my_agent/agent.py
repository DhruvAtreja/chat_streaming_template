from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, tool_node, should_continue, PreprocessNode
from my_agent.utils.state import AgentState
from my_agent.utils.memory_manager import MemoryManager
from dotenv import load_dotenv
import os

load_dotenv()

class GraphConfig(TypedDict):
    model_name: Literal["gpt-4o", "haiku", "gpt-4o-mini", "sonnet-3.5"]
    system_instructions: str

class AgentWorkflow:
    def __init__(self):
        self.workflow = StateGraph(AgentState, config_schema=GraphConfig)
        self.memory_manager = MemoryManager(api_key=os.getenv("MEM0_API_KEY"))
        self.setup_nodes()
        self.setup_edges()

    def setup_nodes(self):
        # self.workflow.add_node("preprocess", self.preprocess_message)
        self.workflow.add_node("agent", call_model)
        self.workflow.add_node("action", tool_node)
        self.workflow.set_entry_point("agent")

    def setup_edges(self):
        # Uncomment this and set the entry point to "preprocess" to add memory to the agent
        # self.workflow.add_edge("preprocess", "agent")
        self.workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        self.workflow.add_edge("action", "agent")

    def preprocess_message(self, state):
        return PreprocessNode.preprocess_message(state, self.memory_manager)

    def compile(self):
        return self.workflow.compile()

# Usage
agent_workflow = AgentWorkflow()
graph = agent_workflow.compile()

# for local testing
# if __name__ == "__main__":
#     graph.invoke({
#         "messages": [{"role": "user", "content": "What's langchain?"}],
#         "userId": "dhruv",
#     })
