import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        logger.info(f"Loading BGEM3FlagModel from {settings.BGE_M3_MODEL_PATH}...")
        try:
            self.model = BGEM3FlagModel(settings.BGE_M3_MODEL_PATH, use_fp16=False)
        except Exception as e:
            logger.error(f"Failed to load BGEM3FlagModel: {e}")
            self.model = None

        logger.info(f"Connecting to Qdrant at {settings.QDRANT_PATH}...")
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        self.collection_name = settings.QDRANT_COLLECTION
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024, # BGE-M3 dense dimension
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def _convert_lexical_weights_to_sparse_vector(self, lexical_weights: Dict[str, float]) -> models.SparseVector:
        indices = []
        values = []
        for token, weight in lexical_weights.items():
            # hash the token string to get an int - not perfect but works
            idx = abs(hash(token)) % (10**9)
            indices.append(idx)
            values.append(weight)
        return models.SparseVector(indices=indices, values=values)

    def ingest_document(self, text: str, source: str = "Unknown"):
        # TODO: Implement asynchronous batch processing for massive documents
        if not self.model:
            logger.warning("BGE-M3 model not loaded, skipping ingestion.")
            return

        logger.info(f"Ingesting document from {source}...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE, 
            chunk_overlap=settings.RAG_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(text)
        
        if not chunks:
            return

        logger.info(f"Created {len(chunks)} chunks. Generating embeddings...")
        embeddings = self.model.encode(chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        
        dense_vecs = embeddings['dense_vecs']
        lexical_weights_list = embeddings['lexical_weights']

        points = []
        for i, chunk in enumerate(chunks):
            sparse_vec = self._convert_lexical_weights_to_sparse_vector(lexical_weights_list[i])
            
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": sparse_vec
                },
                payload={
                    "text": chunk,
                    "source": source
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Successfully ingested chunks into Qdrant.")

    def hybrid_search(self, query: str, top_k: int = None) -> str:
        if top_k is None:
            top_k = settings.RAG_TOP_K
            
        if not self.model:
            return "RAG Search unavailable (Model not loaded)."

        logger.info(f"Performing hybrid search for: {query}")
        embeddings = self.model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        
        dense_query = embeddings['dense_vecs'][0].tolist()
        sparse_query = self._convert_lexical_weights_to_sparse_vector(embeddings['lexical_weights'][0])

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=top_k * 2
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values
                        ),
                        using="sparse",
                        limit=top_k * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k
            )
            
            context_chunks = []
            for point in results.points:
                context_chunks.append(point.payload.get("text", ""))
                
            return "\n\n---\n\n".join(context_chunks)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return "Failed to retrieve context."
