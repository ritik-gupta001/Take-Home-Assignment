"""
Semantic search over extracted clauses using local sentence embeddings
+ ChromaDB. Runs entirely locally (no extra API cost) so it's cheap to
add on top of the main pipeline.

Usage:
    store = ClauseSearchIndex()
    store.index_clauses(results)              # results = pipeline output
    store.search("early termination for breach", top_k=5)
"""

import logging
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for clause-level search


class ClauseSearchIndex:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embedder = SentenceTransformer(_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("cuad_clauses")

    def index_clauses(self, results: List[dict]) -> None:
        """
        results: pipeline output, list of dicts with
            {contract_id, termination_clause: {...}, confidentiality_clause: {...}, liability_clause: {...}}
        Indexes every non-empty extracted clause for semantic search.
        """
        ids, documents, metadatas = [], [], []

        for result in results:
            contract_id = result["contract_id"]
            for clause_type in ["termination_clause", "confidentiality_clause", "liability_clause"]:
                clause = result.get(clause_type, {})
                if isinstance(clause, dict) and clause.get("found") and clause.get("text"):
                    ids.append(f"{contract_id}__{clause_type}")
                    documents.append(clause["text"])
                    metadatas.append({"contract_id": contract_id, "clause_type": clause_type})

        if not documents:
            logger.warning("No clauses to index.")
            return

        embeddings = self.embedder.encode(documents).tolist()
        self.collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info(f"Indexed {len(documents)} clauses.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top_k clauses most semantically similar to the query."""
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "contract_id": meta["contract_id"],
                "clause_type": meta["clause_type"],
                "text": doc,
                "similarity": 1 - dist,
            })
        return hits
