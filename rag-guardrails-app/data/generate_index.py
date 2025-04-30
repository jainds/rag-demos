from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    "The Eiffel Tower is a landmark in Paris, France, completed in 1889.",
    "The Great Wall of China is a series of fortifications built to protect Chinese states.",
    "The Amazon River is the largest river in the world by discharge volume."
]

# Generate embeddings
vectors = model.encode(documents)

# Build FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# Save the index
faiss.write_index(index, "vector_index.faiss")
