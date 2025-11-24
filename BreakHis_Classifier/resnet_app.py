import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------
# Device setup
# --------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Sidebar content
st.sidebar.header(" Model Overview")
st.sidebar.markdown("**Model Backbone:** ResNet‑50 (pretrained on ImageNet‑1K)")
st.sidebar.markdown("**Fine‑tuned Dataset:** BreaKHis (40x–400x histopathology images)")
st.sidebar.markdown("**Training Notes:** Early stopping and learning rate scheduling applied for optimal convergence.")

# --------------------------
# Model Definition (ResNet)
# --------------------------
class BreastCancerResNet(nn.Module):
    def __init__(self):
        super(BreastCancerResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
)
        

    def forward(self, x):
        return self.backbone(x)

# --------------------------
# Load the model (cached)
# --------------------------
@st.cache_resource
def load_model():
    model = BreastCancerResNet()
    state_dict = torch.load("models/best_model_resnet_50.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# --------------------------
# Image transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Breast Cancer Classifier (ResNet)", layout="wide")

# --------------------------
# Sidebar: Model Info & Evaluation Results
# --------------------------

st.sidebar.markdown("###  Evaluation Metrics")
st.sidebar.markdown("""
<table style='width:100%; border-collapse: collapse;'>
  <tr><th style='text-align:left;'>Metric</th><th>Score</th></tr>
  <tr><td>Precision</td><td>0.94</td></tr>
  <tr><td>Recall</td><td>0.99</td></tr>
  <tr><td>F1-score</td><td>0.96</td></tr>
  <tr><td>AUC (ROC)</td><td>0.98</td></tr>
</table>
""", unsafe_allow_html=True)
st.sidebar.info(" Model validated on mixed magnification (40x–400x) dataset with early stopping and LR scheduling.")
st.markdown("<style> div.block-container{padding-top:1rem;} </style>", unsafe_allow_html=True)
st.title(" Breast Cancer Classification using ResNet-50")
st.write("Upload a histopathology image (40x–400x) to predict benign or malignant.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    classes = ["Benign", "Malignant"]
    confidence = probs[pred_idx].item() * 100
    label = classes[pred_idx]

    CONF_THRESHOLD = 60.0

    if confidence < CONF_THRESHOLD:
        st.markdown("### ⚠️ PREDICTION: **UNCERTAIN**")
        st.markdown(
            f"<h3 style='color:red;'>⚠️ Model confidence is only {confidence:.2f}% — please review manually.</h3>",
            unsafe_allow_html=True
        )
    else:
        color = "green" if label == "Benign" else "red"
        st.markdown(f"<h2 style='color:{color};'>PREDICTION: {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")

    # --------------------------
    # Grad-CAM Visualization
    # --------------------------

    target_layer = model.backbone.layer4[-1]  # last conv block of ResNet-50
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    # Register hooks
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_backward_hook(save_gradient)

    # Forward + backward pass
    model.zero_grad()
    output = model(img_t)
    pred_class = torch.argmax(output)
    output[0, pred_class].backward()

    # Compute Grad-CAM
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = np.uint8(0.5 * heatmap + 0.5 * img_np)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Display side by side with centered layout and equal widths
    col1, col2 = st.columns([1,1], gap="medium")
    with col1:
        st.image(img, caption="Original Image", width='stretch')
    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap", width='stretch')

    # Centered minimal guide in subtle gray text below images
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.9rem; line-height:1.5; margin-top: 10px;'>"
        "<b>Heatmap Interpretation:</b> Red/Yellow areas highlight regions the model considered important. "
        "Blue/Purple regions had lesser influence in the classification decision."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Model: ResNet-50 (pretrained on ImageNet, fine-tuned on BreaKHis)")
