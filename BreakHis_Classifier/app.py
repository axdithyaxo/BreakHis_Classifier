import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ------------------------
# Device setup
# ------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ------------------------
# Define CNN architecture
# ------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------------
# Model loading
# ------------------------
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("models/cnn_breakhis.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------
# Streamlit UI
# ------------------------
st.title("🩺 Breast Cancer Classification (CNN)")
st.write("Upload a histopathology image to classify as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    # Preprocess image
    img_t = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    classes = ['benign', 'malignant']
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx].item() * 100

    CONFIDENCE_THRESHOLD = 70.0

    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown("### ⚠️ Prediction: **UNCERTAIN**")
        st.markdown(
        f"<h3 style='color:red;'>⚠️ Model confidence is only {confidence:.2f}% — please review manually.</h3>",
        unsafe_allow_html=True)
    else:
        st.markdown(f"### 🧠 Prediction: **{pred_label.upper()}**")
        st.markdown(f"Confidence: **{confidence:.2f}%**")