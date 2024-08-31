from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from my_agent.utils.memory_manager import MemoryManager

@lru_cache(maxsize=4)
def _get_model(model_name: str):

    if model_name == "gpt-4o":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "haiku":
        model = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
    elif model_name == "gpt-4o-mini":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "sonnet-3.5":
        model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant. Ask follow up questions if needed before providing an answer. Answer in markdown format when relevant."""

# Define the function to execute tools
tool_node = ToolNode(tools)

class PreprocessNode:
    @staticmethod
    def preprocess_message(state, memory_manager):
        messages = state["messages"]
        user_id = state["userId"]        
        # Add the new message to user memory
        # Convert messages to a JSON-serializable format
        print(messages)
        serializable_messages = [
            {
                "role": msg.type,
                "content": msg.content
            } for msg in messages
        ]
        memory_manager.add_to_memory(serializable_messages, user_id)
        
        # Get relevant context from user memory
        last_message = serializable_messages[-1]["content"]
        relevant_context = memory_manager.get_relevant_context(last_message, user_id)
        
        # Prepare the messages list
        processed_messages = []        
        # Add relevant context to the messages
        if relevant_context:
            context_message = {"role": "system", "content": f"Relevant context: {relevant_context}"}
            processed_messages.append(context_message)
                
        # Add the user messages
        processed_messages.extend(messages)
        
        return {"messages": processed_messages}

# Update the call_model function to use the updated messages
def call_model(state, config):
    messages = state["messages"]
    system_instructions = config.get('configurable', {}).get("system_instructions", None)

    system_content = system_prompt
    if system_instructions is not None:
        system_content += system_instructions
    messages = [{"role": "system", "content": system_content}] + messages
    model_name = config.get('configurable', {}).get("model_name", "gpt-4o-mini")
    model = _get_model(model_name)
    response = model.invoke(messages)
    print(response)
    return {"messages": [response]}