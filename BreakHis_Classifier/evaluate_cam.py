import torch, numpy as np, cv2, os
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr
from math import log

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def cam_concentration(cam, top_pct=0.2):
    flat = cam.flatten()
    k = int(len(flat) * top_pct)
    idx = np.argpartition(flat, -k)[-k:]
    return flat[idx].sum() / flat.sum()

def cam_entropy(cam, eps=1e-8):
    p = cam / (cam.sum() + eps)
    return -(p * np.log(p + eps)).sum()

from torch import nn
from torchvision import models

# Define the same ResNet architecture used in cnn_resnet.py
class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
)

    def forward(self, x):
        return self.backbone(x)

def compute_gradcam(model, img_t, target_class):
    gradients, activations = [], []
    target_layer = model.backbone.layer4[-1]  # last ResNet block
    def save_grad(module, grad_input, grad_output): gradients.append(grad_output[0])
    def save_act(module, inp, out): activations.append(out)
    h1 = target_layer.register_forward_hook(save_act)
    h2 = target_layer.register_full_backward_hook(save_grad)
    model.zero_grad()
    out = model(img_t)
    out[0, target_class].backward()
    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1,2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights): cam += w*act[i]
    cam = np.maximum(cam,0); cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()
    return cam

def compute_gradcam_pp(model, img_t, target_class):
    gradients, activations = [], []
    target_layer = model.backbone.layer4[-1]
    def save_grad(module, grad_input, grad_output): gradients.append(grad_output[0])
    def save_act(module, inp, out): activations.append(out)
    h1 = target_layer.register_forward_hook(save_act)
    h2 = target_layer.register_full_backward_hook(save_grad)

    model.zero_grad()
    output = model(img_t)
    score = output[0, target_class]
    score.backward(retain_graph=True)

    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]
    grad2 = np.power(grad, 2)
    grad3 = np.power(grad, 3)
    eps = 1e-8

    weights = np.sum(grad2, axis=(1,2)) / (2*np.sum(grad2, axis=(1,2)) + np.sum(act * grad3, axis=(1,2)) + eps)
    weights = np.maximum(weights, 0)
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    h1.remove()
    h2.remove()
    return cam

def occlusion_drop(model, img_t, cam, device, frac=0.15):
    H,W = cam.shape
    k = int(H*W*frac)
    idx = np.argpartition(cam.flatten(), -k)[-k:]
    mask = np.ones(H*W); mask[idx]=0; mask=mask.reshape(H,W)
    mask = cv2.resize(mask, (224,224))
    masked = img_t.clone()
    for c in range(masked.shape[1]): masked[0, c] *= torch.from_numpy(mask.astype(np.float32)).to(device)
    with torch.no_grad():
        base = torch.softmax(model(img_t), dim=1).max().item()
        masked_conf = torch.softmax(model(masked), dim=1).max().item()
    return base - masked_conf

def cam_correlation(cam1, cam2):
    return spearmanr(cam1.flatten(), cam2.flatten()).correlation

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load the trained ResNet model
    model = ResNetClassifier().to(device)
    state_dict = torch.load("models/best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    benign_dir = "data/val/benign"
    malignant_dir = "data/val/malignant"

    benign_imgs = [os.path.join(benign_dir,f) for f in os.listdir(benign_dir)[:10]]
    malignant_imgs = [os.path.join(malignant_dir,f) for f in os.listdir(malignant_dir)[:10]]

    for label, imgs in [("Benign", benign_imgs), ("Malignant", malignant_imgs)]:
        concs, ents, drops, corrs = [],[],[],[]
        for path in imgs:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            pred = model(img_t).argmax().item()
            cam = compute_gradcam(model, img_t, pred)
            concs.append(cam_concentration(cam))
            ents.append(cam_entropy(cam))
            drops.append(occlusion_drop(model,img_t,cam,device))
            # Randomized correlation
            cam_rand = np.random.rand(*cam.shape)
            corrs.append(cam_correlation(cam, cam_rand))
            if path == imgs[0]:  # visualize only first image per class
                cam_pp = compute_gradcam_pp(model, img_t, pred)
                heatmap_std = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
                heatmap_pp = cv2.applyColorMap(np.uint8(255*cam_pp), cv2.COLORMAP_JET)
                img_np = np.array(img.resize((224,224)))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                overlay_std = np.uint8(0.5*heatmap_std + 0.5*img_np)
                overlay_pp = np.uint8(0.5*heatmap_pp + 0.5*img_np)
                comparison = np.hstack((overlay_std, overlay_pp))
                cv2.imwrite(f"{label}_gradcam_comparison.png", comparison)
                print(f"Saved Grad-CAM comparison for {label} as {label}_gradcam_comparison.png")
        print(f"\n=== {label} ===")
        print(f"Avg CAM concentration: {np.mean(concs):.3f}")
        print(f"Avg entropy: {np.mean(ents):.3f}")
        print(f"Avg occlusion drop: {np.mean(drops):.3f}")
        print(f"Randomized correlation: {np.mean(corrs):.3f}")
