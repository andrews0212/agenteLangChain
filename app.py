import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
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


agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="Eres un asistente útil. Haz UNA sola búsqueda y responde con eso. No hagas más de una búsqueda.",
)

inputs = {"messages": [{"role": "user", "content": "Dame las 3 noticias más importantes de tecnología hoy"}]}
for chunk in agent.stream(inputs, stream_mode="updates", config={"recursion_limit": 5}):
    if "tools" in chunk:
        for msg in chunk["tools"]["messages"]:
            print(f"[Buscando...] {msg.name}")
    if "model" in chunk:
        for msg in chunk["model"]["messages"]:
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
