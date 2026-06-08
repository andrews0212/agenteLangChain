import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from ddgs import DDGS

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


@tool
def search(query: str) -> str:
    """Busca información en la web usando DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
    return "\n".join(r["body"] for r in results)


tools = [search]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = "Eres un asistente útil. Haz UNA sola búsqueda y responde con eso. No hagas más de una búsqueda."


def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

inputs = {"messages": [HumanMessage(content="Dame las 3 noticias más importantes de tecnología hoy")]}
for chunk in app.stream(inputs, stream_mode="updates", config={"recursion_limit": 5}):
    if "tools" in chunk:
        for msg in chunk["tools"]["messages"]:
            print(f"[Buscando...] {msg.name}")
    if "agent" in chunk:
        for msg in chunk["agent"]["messages"]:
            if not hasattr(msg, "content"):
                continue
            content = msg.content
            if isinstance(content, str) and content:
                print("\n--- Respuesta ---")
                print(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print("\n--- Respuesta ---")
                        print(block["text"])