import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from ddgs import DDGS

load_dotenv()

# --- Configuración de aprobación por herramienta ---
# Decisions: "approve" | "edit" | "reject" | "respond"
APPROVAL_CONFIG = {
    "write_file": {"allowed_decisions": ["approve", "edit", "reject", "respond"]},
    "execute_sql": {"allowed_decisions": ["approve", "reject"]},
    # "search" no aparece aquí → se ejecuta automáticamente sin aprobación
}

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


@tool
def write_file(path: str, content: str) -> str:
    """Escribe contenido en un archivo."""
    with open(path, "w") as f:
        f.write(content)
    return f"Archivo '{path}' escrito con éxito."


@tool
def execute_sql(query: str) -> str:
    """Ejecuta una consulta SQL (simulado)."""
    return f"Query ejecutada: {query}"


tools = [search, write_file, execute_sql]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = "Eres un asistente útil. Usa las herramientas disponibles cuando sea necesario."


def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def approval_node(state: MessagesState):
    """
    Para cada tool call pendiente:
      - Si no está en APPROVAL_CONFIG → aprobación automática.
      - Si está → llama a interrupt() y espera la decisión humana.

    Decisiones posibles:
      {"action": "approve"}
      {"action": "edit", "edited_args": {...}}
      {"action": "reject"}
      {"action": "respond", "content": "mensaje custom"}
    """
    last_msg = state["messages"][-1]
    approved_calls = []
    extra_messages = []

    for tc in last_msg.tool_calls:
        config = APPROVAL_CONFIG.get(tc["name"])

        if config is None:
            # Sin aprobación requerida
            approved_calls.append(tc)
            continue

        # Pausa y espera decisión del humano
        decision = interrupt({
            "message": f"Tool execution pending approval: {tc['name']} with args={tc['args']}",
            "tool_name": tc["name"],
            "args": tc["args"],
            "allowed_decisions": config["allowed_decisions"],
        })

        action = decision.get("action")

        if action == "approve":
            approved_calls.append(tc)

        elif action == "edit" and "edit" in config["allowed_decisions"]:
            approved_calls.append({**tc, "args": decision["edited_args"]})

        elif action == "reject":
            extra_messages.append(ToolMessage(
                content=f"Operación '{tc['name']}' rechazada por el usuario.",
                tool_call_id=tc["id"],
            ))

        elif action == "respond" and "respond" in config["allowed_decisions"]:
            extra_messages.append(ToolMessage(
                content=decision["content"],
                tool_call_id=tc["id"],
            ))

    # Reemplaza el AIMessage original conservando su id (el reducer de MessagesState lo actualiza)
    updated_msg = AIMessage(
        id=last_msg.id,
        content=last_msg.content,
        tool_calls=approved_calls,
    )
    return {"messages": [updated_msg] + extra_messages}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "approval"
    return END


def after_approval(state: MessagesState):
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai and last_ai.tool_calls:
        return "tools"
    return "agent"


# --- Construcción del grafo ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("approval", approval_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_conditional_edges("approval", after_approval)
graph.add_edge("tools", "agent")

# checkpointer es OBLIGATORIO para que interrupt() funcione
app = graph.compile(checkpointer=InMemorySaver())


# --- Ejecución con loop de aprobación ---
def run_with_approval(user_message: str, thread_id: str = "1"):
    thread = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=user_message)]}

    while True:
        pending_interrupt = None

        for chunk in app.stream(inputs, config=thread, stream_mode="updates"):
            # Detecta si el grafo se pausó en un interrupt()
            if "__interrupt__" in chunk:
                pending_interrupt = chunk["__interrupt__"][0].value
                break

            # Salida normal
            if "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    print(f"[Ejecutando herramienta] {msg.name}")
            if "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and content:
                        print("\n--- Respuesta ---")
                        print(content)

        if pending_interrupt is None:
            break  # Grafo terminó sin interrupciones pendientes

        # Mostrar la solicitud de aprobación al humano
        print(f"\n[APROBACIÓN REQUERIDA]")
        print(f"  Herramienta : {pending_interrupt['tool_name']}")
        print(f"  Argumentos  : {pending_interrupt['args']}")
        print(f"  Decisiones  : {pending_interrupt['allowed_decisions']}")
        print()

        allowed = pending_interrupt["allowed_decisions"]
        action = input(f"Acción ({'/'.join(allowed)}): ").strip().lower()

        if action == "approve":
            decision = {"action": "approve"}
        elif action == "edit" and "edit" in allowed:
            new_args = {}
            for key in pending_interrupt["args"]:
                val = input(f"  {key} [{pending_interrupt['args'][key]}]: ").strip()
                new_args[key] = val if val else pending_interrupt["args"][key]
            decision = {"action": "edit", "edited_args": new_args}
        elif action == "reject":
            decision = {"action": "reject"}
        elif action == "respond" and "respond" in allowed:
            content = input("  Respuesta custom: ").strip()
            decision = {"action": "respond", "content": content}
        else:
            print("Acción inválida, rechazando por defecto.")
            decision = {"action": "reject"}

        # Retoma el grafo desde donde se pausó
        inputs = Command(resume=decision)


if __name__ == "__main__":
    run_with_approval("Dame las últimas noticias de tecnología")
