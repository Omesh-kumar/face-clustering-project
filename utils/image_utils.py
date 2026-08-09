import os
from face.detector import detect_faces
from face.embeddings import get_embedding

def extract_embeddings_from_folder(folder):
    embeddings = []
    metadata = []

    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        faces = detect_faces(path)

        for face in faces:
            emb = get_embedding(face)
            embeddings.append(emb)
            metadata.append((img, path))

    return embeddings, metadata
