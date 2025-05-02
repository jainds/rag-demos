from typing import List, Dict, Union, Optional
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

class DocumentIndexer:
    """Service for creating and managing vector indices for documents."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 dimension: Optional[int] = None):
        """
        Initialize the document indexer.
        
        Args:
            model_name: Name of the sentence transformer model to use
            dimension: Dimension of vectors (if None, determined from model)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.index = None
        
    def create_index(self, documents: List[Dict[str, str]]) -> None:
        """
        Create a FAISS index from documents.
        
        Args:
            documents: List of document dictionaries with 'text' field
        """
        # Extract text from documents
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create and populate FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
    def save_index(self, index_path: Union[str, Path]) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
        """
        if self.index is None:
            raise ValueError("No index exists. Create an index first.")
            
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        
    def load_index(self, index_path: Union[str, Path]) -> None:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to the saved index
        """
        self.index = faiss.read_index(str(index_path))
        
    def search(self, 
               query: str,
               k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search the index for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("No index exists. Create or load an index first.")
            
        # Generate query embedding
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # Search index
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), 
            k
        )
        
        return distances, indices 