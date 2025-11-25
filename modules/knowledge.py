import chromadb
# ... rest of your importsimport chromadb
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, collection_name="home_knowledge"):
        # Initialize Vector DB client
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        # Load embedding model (efficient, runs on CPU)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def ingest_documents(self, documents: list):
        """Takes list of text strings and stores embeddings."""
        ids = [str(i) for i in range(len(documents))]
        embeddings = self.embedder.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        print(f"✅ Ingested {len(documents)} documents into RAG.")

    def retrieve_context(self, query: str, n_results=2) -> str:
        """Finds most relevant docs for the query."""
        query_embed = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embed,
            n_results=n_results
        )
        # Flatten list of lists
        context = " ".join(results['documents'][0])
        return context