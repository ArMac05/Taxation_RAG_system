import os
import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv() # reads .env for API Key


MODEL_NAME:str = "claude-sonnet-5-5" 
TOP_k:int = 5


embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="db").get_collection("book")
llm = anthropic.Anthropic(
    default_headers={
        "anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]
    }
) # reads API_Anthropic key and workspace from environment

def retrieve(question:str, k:int=TOP_k) -> list[tuple[str, str]]:
    """
        Return the top-k (section, text) pairs for a question.
    """

    q_emb:list[list[float]] = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=[list(map(float, q)) for q in q_emb], n_results=k)
    docs:list[str] = res["documents"][0] # type: ignore[index]
    metas = res["metadatas"][0] # type: ignore[index]
    return [(str(m["section"]), d) for d, m in zip(docs, metas)] 

def ask(question:str) -> str:
    passages = retrieve(question)
    context:str = "\n\n".join(f"[{sec}]\n{text}" for sec, text in passages)
    prompt:str =f"""Answer the question only using the context below.
    Cite the section names you used. If the answer isn't in the context, say you don't know.

    Context:
    {context}

    Question: {question}"""

    msg = llm.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    parts:list[str] = [
        block.text for block in msg.content if block.type == "text"
    ]
    return "\n".join(parts)

def main() -> None:
    while True:
        q:str = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        print("\n" + ask(q))


if __name__ == "__main__":
    main()