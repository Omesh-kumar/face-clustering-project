import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors
    """
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_my_face(face_emb, my_embs, threshold=0.85):
    """
    Check if a face embedding matches any of the reference embeddings
    """
    similarities = [cosine_similarity(face_emb, ref) for ref in my_embs]
    return max(similarities) >= threshold
