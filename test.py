import faiss
import numpy as np

# Sample data: 5 vectors, each with 128 dimensions
data = np.random.random((5, 128)).astype('float32')

# Create a Flat Index (brute-force search)
index = faiss.IndexFlatL2(128)  # 128 is the dimension of vectors
index.add(data)  # Add data to the index

# Query with a sample vector
query = np.random.random((1, 128)).astype('float32')
distances, indices = index.search(query, 1)

print(distances)
print(indices)
