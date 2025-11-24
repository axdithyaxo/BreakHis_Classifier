import os
import random
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("data/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast")
DEST_DIR = Path("data")
TRAIN_RATIO = 0.8
CLASSES = ["benign", "malignant"]
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# --- Seed for reproducibility ---
random.seed(42)

# --- Make train/val folders ---
for split in ["train", "val"]:
    for cls in CLASSES:
        (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def collect_patients(class_dir):
    """
    Collects all patient subdirectories (e.g., SOB_B_A_14-22549AB)
    for the given class directory.
    """
    sob_dir = class_dir / "SOB"
    patients = [p for p in sob_dir.rglob("*") if p.is_dir() and any(m in str(p) for m in MAGNIFICATIONS)]
    patient_roots = set()
    for p in patients:
        root = str(p).split("/"+MAGNIFICATIONS[0])[0]
        patient_roots.add(root)
    return list(patient_roots)

# --- Split patients by train/val ---
for cls in CLASSES:
    class_dir = SOURCE_DIR / cls
    patients = collect_patients(class_dir)
    random.shuffle(patients)
    split_idx = int(len(patients) * TRAIN_RATIO)
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]

    print(f"\n📊 {cls.upper()} — Total patients: {len(patients)}, Train: {len(train_patients)}, Val: {len(val_patients)}")

    for split, patient_list in [("train", train_patients), ("val", val_patients)]:
        dest_path = DEST_DIR / split / cls

        for patient_root in patient_list:
            for mag in MAGNIFICATIONS:
                mag_path = Path(patient_root) / mag
                if mag_path.exists():
                    for img_file in mag_path.glob("*.png"):
                        dest_file = dest_path / f"{Path(patient_root).name}_{mag}_{img_file.name}"
                        shutil.copy(img_file, dest_file)

print("\n✅ All magnifications organized by patient into train/val folders!")