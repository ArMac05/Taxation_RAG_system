import sys
from ingest import BOOK_PATH, load_epub

def main() -> None:
    phrase:str = " ".join(sys.argv[1:]).lower()
    for s in load_epub(BOOK_PATH):
        count:int = s["text"].lower().count(phrase)
        if count:
            print(f"{s["section"]}: {count} match(es)")

if __name__ == "__main__":
    main()