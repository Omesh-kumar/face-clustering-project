from facenet_pytorch import MTCNN
import cv2

mtcnn = MTCNN(keep_all=True)

def detect_faces(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(img)
    faces = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            faces.append(face)

    return faces
