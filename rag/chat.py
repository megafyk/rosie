from typing import Annotated

from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing_extensions import TypedDict

from config import LLM_MODEL_NAME


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def chatbot(state: State, llm: HuggingFacePipeline) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


def stream_graph_updates(graph, user_input: str, config):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
        for value in event.values():
            print(value)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


model_id = LLM_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Extract generation kwargs from pipeline creation kwargs
generation_kwargs = {
    "device": "cpu",
    "temperature": 0.2,
    "return_full_text": False,
    "max_new_tokens": 1024,
}

pipeline_instance = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    **generation_kwargs
)

llm: HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline_instance)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", lambda state: chatbot(state, llm))
memory = MemorySaver()

tools = [human_assistance]

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph, user_input, config)
        except EOFError:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, user_input, config)
            break
