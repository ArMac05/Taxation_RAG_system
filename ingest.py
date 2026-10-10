import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

BOOK_PATH:str = "data/your_federal_income_tax_p17.epub"

SKIP_KEYWORDS:tuple[str, ...] = ("nav", "cover", "toc")

def load_epub(path:str) -> list[dict[str, str]]:
    '''
        Load the EPUB, one entry per chapter file. Stored as a list of dictionary.
        Section and Text.
    '''
    book = epub.read_epub(path)
    sections: list[dict[str, str]] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name:str = item.get_name()
        if any(k in name.lower() for k in SKIP_KEYWORDS):
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text:str = soup.get_text(separator="\n").strip() # Removing newline
        if len(text) > 500: # skip covers, blank pages, tiny files
            sections.append({"section": item.get_name(), "text": text})
    return sections

def chunk(
        sections:list[dict[str, str]],
        size:int = 800,
        overlap:int = 100
) -> list[dict[str, str]]:
    '''
        Split each section into overlapping chunks
    '''
    chunks:list[dict[str, str]] = []

    for s in sections:
        text:str = s["text"]
        for start in range(0,len(text), size - overlap):
            piece:str = text[start:start + size].strip()
            if len(piece) > 100:
                chunks.append({"text": piece, "section": s["section"]})
    return chunks


def main() -> None:
    sections = load_epub(BOOK_PATH)
    print(f"Loaded {len(sections)} sections")

    chunks = chunk(sections)
    print(f"Created {len(chunks)} chunks")

    # Print a few chunks so you can check they look clean
    for c in chunks[:3]:
        print("-" * 40)
        print(c["section"])
        print(c["text"][:300])


    # Embed and Store
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")

    # Start fresh each time you re-run
    try:
        client.delete_collection("book")
    except Exception:
        pass
    col = client.create_collection("book")

    texts:list[str] = [c["text"] for c in chunks]
    embeddings:list[list[float]] = model.encode(
        texts, show_progress_bar=True
    ).tolist()

    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=[list(map(float, e)) for e in embeddings],
        metadatas=[{"section": c["section"]} for c in chunks]
    )

    print(f"Stored {len(chunks)} chunks in the database")

if __name__ == "__main__":
    main()
