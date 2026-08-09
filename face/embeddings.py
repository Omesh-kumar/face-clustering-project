from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(face_img):
    face = cv2.resize(face_img, (160, 160))
    face = torch.tensor(face).permute(2, 0, 1).float()
    face = face.unsqueeze(0) / 255.0

    with torch.no_grad():
        embedding = model(face)

    return embedding.numpy().flatten()
