# BreakHis Breast Cancer Classifier (PyTorch + ResNet‑50)

This project implements a **high‑performance breast‑cancer histopathology classifier** trained on the **BreaKHis** dataset (40×–400× magnification). The system uses **transfer learning with ResNet‑50**, modern training strategies, and full **Grad‑CAM / Grad‑CAM++ medical explainability**. A companion **Streamlit diagnostic app** provides real‑time predictions with visual heatmaps for interpretability.

---

## Features

- **ResNet‑50 fine‑tuning** on mixed‑magnification histopathology slides  
- **Advanced training pipeline:**  
  - Label smoothing  
  - Early stopping  
  - Cosine annealing LR scheduling  
  - Mixed augmentations (rotation, flips, color jitter)  
- **High performance**: Achieved *F1 = 0.96*, *AUC = 0.98* on validation  
- **Full explainability**: Grad‑CAM + Grad‑CAM++ heatmaps  
- **Streamlit medical diagnostic UI** (upload → prediction → heatmap)  
- **CAM evaluation utilities**: concentration, entropy, occlusion causal metrics  

---

## Dataset Structure (BreaKHis)

Organize the dataset into standard `ImageFolder` format:

```
data/
 ├── train/
 │   ├── benign/
 │   └── malignant/
 ├── val/
 │   ├── benign/
 │   └── malignant/
 └── test/ (optional)
     ├── benign/
     └── malignant/
```

A helper script (`organiser.py`) is included to automatically sort raw BreaKHis folders into this structure.

---

## Training the ResNet‑50 Model

Fine‑tune the model:

```bash
python cnn_resnet.py --epochs 20 --batch-size 16 --device mps
```

Key arguments:
- `--device`: `cpu`, `cuda`, or `mps` (Apple Silicon)  
- `--epochs`: number of training epochs  
- `--lr`: learning rate  
- `--data-root`: dataset root folder  
- `--save-path`: where to store `best_model.pth`

The script:
- Freezes the ResNet backbone for warm‑up  
- Unfreezes all layers for fine‑tuning  
- Tracks precision, recall, F1, AUC, and balanced accuracy  
- Saves the **best checkpoint** (early stopping)

---

## Run the Diagnostic Streamlit App

Start the UI:

```bash
streamlit run resnet_app.py
```

Features:
- Upload an image and receive **benign vs malignant** classification  
- Shows **model confidence**  
- Displays **Grad‑CAM** and **Grad‑CAM++** heatmaps side‑by‑side  
- Clean, compact interface optimized for clinical use

---

## Explainability Evaluation (Optional)

Run deeper interpretability analysis:

```bash
python evaluate_cam.py
```

Outputs:
- CAM concentration  
- Entropy  
- Occlusion‑based causal score  
- Randomization correlation  
- Saves comparison images (`Grad‑CAM vs Grad‑CAM++`)

---

## Next Steps

- Extend to multi‑class tumor subtype classification  
- Add magnification‑specific models (40×, 100×, 200×, 400×)  
- Deploy via FastAPI or TorchServe  
- Convert to ONNX / CoreML for edge deployment

---

## License

This project is open‑source and intended for educational + research use in medical AI explainability.
