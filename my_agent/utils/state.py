from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, Optional

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    userId: str
    system_instructions: Optional[str]
