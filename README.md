# Face Clustering & Personal Photo Organizer

A face-recognition pipeline that automatically sorts a folder of photos into **your solo pictures**, **your group pictures**, and **photos you're not in**, then clusters the group photos by the people in them using a custom optimization-based clustering approach (CBCC + RCPSO + ORC).

## How it works

1. **Reference embeddings** — A few reference photos of you are used to compute your "face signature" using a pretrained FaceNet model.
2. **Face detection & embeddings** — Every input image is scanned with MTCNN to detect faces, and each detected face is converted into a 512-dimensional embedding using `InceptionResnetV1` (pretrained on VGGFace2).
3. **Classification** — Each image is compared against your reference embeddings (cosine similarity) and sorted into:
   - `my_solo/` — only you are in the photo
   - `my_group/` — you + other people
   - `friends_only/` — you are not in the photo
4. **Group photo clustering** — For group photos, faces are clustered using a 3-stage custom pipeline:
   - **CBCC** (Centroid-Based Cluster... Initialization) — picks well-separated starting centroids using a gravitational-force heuristic instead of random init.
   - **RCPSO** (Refined Chaotic Particle Swarm Optimization) — a PSO-based clustering optimizer that refines centroid positions to minimize intra-cluster distance.
   - **ORC** (Outlier Removal Clustering) — removes faces that don't fit well into any cluster (e.g. false-positive detections, blurry/partial faces), then RCPSO re-optimizes on the cleaned data.

## Project structure

```
face_clustering_project/
├── app.py                   # Main pipeline entry point
├── face/
│   ├── detector.py          # MTCNN-based face detection
│   └── embeddings.py        # FaceNet embedding extraction
├── models/
│   ├── cbcc.py               # Centroid initialization
│   ├── rcpso.py               # PSO-based clustering
│   └── orc.py                 # Outlier removal
├── utils/
│   ├── distance.py           # Cosine similarity / face-matching logic
│   └── image_utils.py        # Folder → embeddings helper
├── data/
│   ├── my_reference/          # Your reference photos (put 3-5 clear solo photos here)
│   ├── input_images/          # Photos to be sorted
│   └── output/                 # Generated: my_solo/, my_group/, friends_only/
└── requirements.txt
```

## Example run

```
🔹 Step 1: Loading YOUR reference images...
   ✓ Loaded 5 reference faces

🔹 Step 2: Loading input images...
   ✓ Extracted 20 faces from images

🔹 Step 3: Classifying images into my_solo, my_group, friends_only...
   ✓ Classification done!
   My solo photos: 5
   My group photos: 2
   Friends only photos: 0

🔹 Step 4: CBCC – Smart centroid initialization for group photos...
🔹 Step 5: Initial clustering using RCPSO...
🔹 Step 6: Removing outliers (ORC)...
   ✓ Removed 12 noisy faces
🔹 Step 7: Re-optimizing clusters with clean data...
   ✓ Group clustering done!

🎉 DONE!
📂 Check results in: data/output/
   ├── my_solo/
   ├── my_group/
   └── friends_only/
```

## Setup

```bash
git clone https://github.com/Omesh-kumar/face-clustering-project.git
cd face-clustering-project
pip install -r requirements.txt
```

## Usage

1. Put 3–5 clear photos of yourself in `data/my_reference/`
2. Put the photos you want to sort in `data/input_images/`
3. Run:
   ```bash
   python app.py
   ```
   Or with custom folders / threshold:
   ```bash
   python app.py --input path/to/photos --reference path/to/your/photos --threshold 0.85
   ```
4. Check `data/output/` for the sorted results.

## Tech stack

- **Face detection:** MTCNN (`facenet-pytorch`)
- **Face embeddings:** InceptionResnetV1 pretrained on VGGFace2
- **Clustering:** Custom CBCC + RCPSO + ORC pipeline (NumPy)
- **Similarity metric:** Cosine similarity

## Notes / limitations

- Classification threshold for "is this my face" is currently a fixed cosine similarity of `0.85` — this can be tuned in `app.py` depending on your reference photo quality.
- MTCNN can occasionally produce false-positive face detections on busy/group photos; the ORC stage is designed to filter most of these out during clustering.

## Author

Omesh Kumar — BS Computer Science, Sindh Madressatul Islam University (SMIU)
[GitHub](https://github.com/Omesh-kumar) · [LinkedIn](https://linkedin.com/in/omesh-kumar-295971266)
