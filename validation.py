from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to existing Chroma vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
    client_settings=Settings(anonymized_telemetry=False)
)

def search_comorbidities(query: str, k: int = 3):
    """
    Searches the comorbidity vector database for relevant entries.
    """
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        print("No relevant comorbidities found.")
        return

    print(f"\nTop {len(results)} matching comorbidities for query: '{query}'")

    for i, r in enumerate(results, 1):
        print(f"\n{i}. Condition: {r.metadata.get('condition', 'Unknown')}")
        print(f"   ICD-10: {r.metadata.get('icd10', 'N/A')}")
        print(f"   Chunk ID: {r.metadata.get('chunk_id', '-')}")
        print(f"   Text: {r.page_content[:300]}...")

# Example search
search_comorbidities("hypertension")