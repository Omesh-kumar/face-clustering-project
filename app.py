"""
Face Clustering & Personal Photo Organizer
--------------------------------------------
Sorts a folder of photos into my_solo/, my_group/, and friends_only/
based on reference photos of you, then clusters group photos by the
people in them using a custom CBCC + RCPSO + ORC pipeline.

Usage:
    python app.py
    python app.py --input data/input_images --reference data/my_reference --output data/output
"""

import os
import shutil
import argparse
import logging
import numpy as np

from utils.image_utils import extract_embeddings_from_folder
from utils.distance import is_my_face

from models.cbcc import CBCC
from models.orc import OutlierRemovalClustering
from models.rcpso import RCPSO

# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
log = logging.getLogger("face_clustering")

# ---------------- DEFAULT CONFIG ---------------- #

DEFAULT_INPUT_DIR = "data/input_images"
DEFAULT_REF_DIR = "data/my_reference"
DEFAULT_OUTPUT_DIR = "data/output"

THRESHOLD_MY_FACE = 0.85       # Cosine similarity threshold for face matching
OUTLIER_THRESHOLD = 0.85       # Threshold for ORC outlier removal
MAX_CLUSTERS = 5

# ------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort and cluster photos based on faces they contain."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR,
                        help=f"Folder of photos to sort (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--reference", default=DEFAULT_REF_DIR,
                        help=f"Folder of your reference photos (default: {DEFAULT_REF_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help=f"Where sorted results are written (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_MY_FACE,
                        help=f"Cosine similarity threshold for 'is this my face' (default: {THRESHOLD_MY_FACE})")
    return parser.parse_args()


def prepare_output_dirs(output_dir):
    solo_dir = os.path.join(output_dir, "my_solo")
    group_dir = os.path.join(output_dir, "my_group")
    friends_dir = os.path.join(output_dir, "friends_only")
    for d in [solo_dir, group_dir, friends_dir]:
        os.makedirs(d, exist_ok=True)
    return solo_dir, group_dir, friends_dir


def validate_folder(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"❌ {label} folder not found: '{path}'")
    if not os.listdir(path):
        raise RuntimeError(f"❌ {label} folder is empty: '{path}'")


def main():
    args = parse_args()

    try:
        validate_folder(args.reference, "Reference")
        validate_folder(args.input, "Input")

        log.info("\n🔹 Step 1: Loading YOUR reference images...")
        ref_embeddings, _ = extract_embeddings_from_folder(args.reference)
        ref_embeddings = np.array(ref_embeddings)

        if len(ref_embeddings) == 0:
            raise RuntimeError(
                "❌ No face detected in any reference image. "
                "Make sure your reference photos clearly show your face."
            )
        log.info(f"   ✓ Loaded {len(ref_embeddings)} reference faces")

        # -----------------------------------------------------

        log.info("\n🔹 Step 2: Loading input images...")
        X, metadata = extract_embeddings_from_folder(args.input)
        X = np.array(X)

        if len(X) == 0:
            raise RuntimeError(
                "❌ No faces detected in any input image. Check that the "
                "input folder contains valid, readable image files."
            )
        log.info(f"   ✓ Extracted {len(X)} faces from images")

        # -----------------------------------------------------
        # Step 3: Classify each image based on faces it contains
        log.info("\n🔹 Step 3: Classifying images into my_solo, my_group, friends_only...")

        solo_dir, group_dir, friends_dir = prepare_output_dirs(args.output)

        group_embeddings = []
        group_metadata = []

        # Process each UNIQUE image once. `metadata` has one row per
        # detected face, so iterating it directly would process an image
        # with N faces N times over, duplicating its embeddings.
        seen_images = {}
        for img_name, img_path in metadata:
            seen_images[img_name] = img_path

        for img_name, img_path in seen_images.items():
            embeddings_in_image = X[[i for i, m in enumerate(metadata) if m[0] == img_name]]

            my_face_count = sum(
                is_my_face(emb, ref_embeddings, threshold=args.threshold)
                for emb in embeddings_in_image
            )
            total_faces = len(embeddings_in_image)

            if my_face_count == 0:
                dst = friends_dir
            elif my_face_count == total_faces:
                dst = solo_dir
            else:
                dst = group_dir
                group_embeddings.extend(embeddings_in_image)
                group_metadata.extend([(img_name, img_path)] * len(embeddings_in_image))

            try:
                shutil.copy(img_path, os.path.join(dst, img_name))
            except OSError as e:
                log.warning(f"   ⚠️ Could not copy '{img_name}': {e}")

        log.info("   ✓ Classification done!")
        log.info(f"   My solo photos: {len(os.listdir(solo_dir))}")
        log.info(f"   My group photos: {len(os.listdir(group_dir))}")
        log.info(f"   Friends only photos: {len(os.listdir(friends_dir))}")

        # -----------------------------------------------------
        # Step 4: Run clustering on group photos if any
        if len(group_embeddings) > 0:
            X_group = np.array(group_embeddings)
            n_clusters = min(MAX_CLUSTERS, len(X_group))

            log.info("\n🔹 Step 4: CBCC – Smart centroid initialization for group photos...")
            cbcc = CBCC(n_clusters=n_clusters)
            init_centroids = cbcc.initialize(X_group)

            log.info("\n🔹 Step 5: Initial clustering using RCPSO...")
            rcpso = RCPSO(n_clusters=n_clusters)
            rcpso.fit(X_group, init_centroids=init_centroids)

            labels = rcpso.labels_
            centroids = rcpso.centroids_

            log.info("\n🔹 Step 6: Removing outliers (ORC)...")
            orc = OutlierRemovalClustering(threshold=OUTLIER_THRESHOLD)
            mask = orc.filter(X_group, labels, centroids)

            X_clean = X_group[mask]
            removed = int(np.sum(~mask))
            log.info(f"   ✓ Removed {removed} noisy faces")

            if len(X_clean) > 0:
                log.info("\n🔹 Step 7: Re-optimizing clusters with clean data...")
                rcpso.fit(X_clean, init_centroids=centroids)
                log.info("   ✓ Group clustering done!")
            else:
                log.warning("   ⚠️ All group faces were flagged as outliers; skipping re-optimization.")
        else:
            log.info("\n⚠️ No group photos to cluster.")

        # -----------------------------------------------------
        log.info("\n🎉 DONE!")
        log.info(f"📂 Check results in: {args.output}/")
        log.info("   ├── my_solo/")
        log.info("   ├── my_group/")
        log.info("   └── friends_only/")

    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        raise SystemExit(1)


# ---------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    main()
