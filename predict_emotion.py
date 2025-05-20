import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model
from PIL import Image
import matplotlib.pyplot as plt

# MLCA Module (same as in the notebook)
class MLCA(nn.Module):
    def __init__(self, x1_dim, x2_dim, embed_dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.x1_proj = nn.Linear(x1_dim, embed_dim)
        self.x2_proj = nn.Linear(x2_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x1, x2):
        x1 = self.x1_proj(x1)
        x2 = self.x2_proj(x2)
        B, N, C = x1.shape
        q = self.q_proj(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)
        k = k.reshape(B * self.num_heads, N, C // self.num_heads)
        v = v.reshape(B * self.num_heads, N, C // self.num_heads)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)
        out = out.reshape(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out

# SwinFace Model (same as in the notebook, with MLCA initialized properly)
class SwinFace(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', embed_dim=512, num_heads=4, num_classes=7, max_tokens=32):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=False, features_only=True)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        # Get feature dimensions for MLCA
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            C1 = features[-2].shape[1]
            C2 = features[-1].shape[1]
        self.mlca = MLCA(x1_dim=C1, x2_dim=C2, embed_dim=embed_dim, num_heads=num_heads)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        f1 = features[-2]
        f2 = features[-1]
        B, C1, H1, W1 = f1.shape
        B, C2, H2, W2 = f2.shape
        f1_flat = f1.flatten(2).transpose(1, 2)
        f2_flat = f2.flatten(2).transpose(1, 2)
        N = min(f1_flat.size(1), f2_flat.size(1), self.max_tokens)
        f1_flat = f1_flat[:, :N, :]
        f2_flat = f2_flat[:, :N, :]
        fused = self.mlca(f1_flat, f2_flat)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# Define emotion labels (aligned with Flask app)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Image transformation (same as test_transform in the notebook)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SwinFace(num_classes=7).to(device)
model_path = "models/swinface_model_93.pth"  # Adjust path as needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def predict_emotion(img_path):
    """
    Predict the emotion from an image.
    
    Args:
        img_path (str): Path to the image file.
        
    Returns:
        dict: Dictionary containing the predicted emotion and confidence scores.
              Example: {'emotion': 'Happy', 'scores': {'Angry': 0.05, 'Disgust': 0.02, ...}}
    """
    try:
        # Load and preprocess the image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():  # No Grad-CAM, so we can use torch.no_grad()
            logits = model(img_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            scores = probabilities.cpu().numpy().flatten()
            rounded_scores = [round(score, 2) for score in scores]

        # Map to emotion
        emotion = class_labels[predicted_class]
        emotion_scores = dict(zip(class_labels, rounded_scores))

        return {
            'emotion': emotion,
            'scores': emotion_scores
        }
    except Exception as e:
        return {
            'emotion': None,
            'scores': None,
            'error': str(e)
        }

# Example usage
if __name__ == "__main__":
    # Replace with the path to your image
    image_path = r"D:\Study-AI\SwinFace\images\WIN_20250518_18_26_13_Pro.jpg"
    result = predict_emotion(image_path)
    if result['emotion']:
        print(f"Predicted Emotion: {result['emotion']}")
        print("Confidence Scores:")
        for emotion, score in result['scores'].items():
            print(f"{emotion}: {score}")
    else:
        print(f"Error: {result['error']}")
        import matplotlib.pyplot as plt

    # Load and display the image
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Emotion: {result["emotion"]}')
    plt.show()