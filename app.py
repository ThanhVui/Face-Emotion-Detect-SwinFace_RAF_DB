import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, render_template, flash, redirect, url_for, Response
import warnings
import uuid
from timm import create_model
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
logging.basicConfig(level=logging.INFO)

print("Starting Flask app...")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MLCA Module
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

# SwinFace Model
class SwinFace(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', embed_dim=512, num_heads=4, num_classes=7, max_tokens=32):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=False, features_only=True)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            C1 = features[-2].shape[1]
            C2 = features[-1].shape[1]
        self.mlca = MLCA(x1_dim=C1, x2_dim=C2, embed_dim=embed_dim, num_heads=num_heads)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

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

# Hook class for Grad-CAM
class Hook:
    def __init__(self):
        self.forward_out = None
        self.backward_out = None
    def register_hook(self, module):
        self.hook_f = module.register_forward_hook(self.forward_hook)
        self.hook_b = module.register_full_backward_hook(self.backward_hook)
    def forward_hook(self, module, input, output):
        self.forward_out = output
    def backward_hook(self, module, grad_in, grad_out):
        self.backward_out = grad_out[0]
    def unregister_hook(self):
        self.hook_f.remove()
        self.hook_b.remove()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the model
model = SwinFace(num_classes=7).to(device)
model_path = "models/swinface_model_93.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Register Grad-CAM hook
final_layer = list(model.backbone._modules.values())[-1]
hook = Hook()
hook.register_hook(final_layer)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Text settings for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (154, 1, 254)  # Neon pink in BGR
thickness = 2
line_type = cv2.LINE_AA
transparency = 0.4

def detect_emotion(pil_crop_img):
    try:
        if not isinstance(pil_crop_img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(pil_crop_img)}")
        img_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
        logits = model(img_tensor)  # Removed torch.no_grad()
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        predicted_class_idx = predicted_class.item()
        model.zero_grad()
        one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_().to(device)
        one_hot_output[0][predicted_class_idx] = 1
        logits.backward(gradient=one_hot_output, retain_graph=True)
        gradients = hook.backward_out
        feature_maps = hook.forward_out
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        scores = probabilities.cpu().numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        return rounded_scores, cam
    except Exception as e:
        logging.error(f"Error in detect_emotion: {e}")
        return None, None

def plot_heatmap(x, y, w, h, cam, pil_crop_img, image):
    try:
        cam = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        roi = image[y:y+h, x:x+w, :]
        overlay = heatmap * transparency + roi / 255 * (1 - transparency)
        overlay = np.clip(overlay, 0, 1)
        image[y:y+h, x:x+w, :] = np.uint8(255 * overlay)
    except Exception as e:
        logging.error(f"Error in plot_heatmap: {e}")

def update_max_emotion(rounded_scores):
    max_index = np.argmax(rounded_scores)
    return class_labels[max_index]

def print_max_emotion(x, y, max_emotion, image):
    org = (x, y - 15)
    cv2.putText(image, max_emotion, org, font, font_scale, font_color, thickness, line_type)

def print_all_emotion(x, y, w, rounded_scores, image):
    org = (x + w + 10, y)
    for index, value in enumerate(class_labels):
        emotion_str = f'{value}: {rounded_scores[index]:.2f}'
        y_offset = org[1] + (index * 30)
        cv2.putText(image, emotion_str, (org[0], y_offset), font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(image, use_mediapipe=True):
    faces = {}
    emotion_scores_dict = {}
    try:
        if use_mediapipe:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
                results = face_detection.process(image)
                if not results.detections:
                    logging.info("MediaPipe: No faces detected.")
                    faces, emotion_scores_dict = detect_bounding_box_opencv(image)
                    return faces, emotion_scores_dict
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        logging.info(f"Skipping invalid bounding box for face_{i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        continue
                    faces[f'face_{i}'] = {'facial_area': [x1, y1, x2, y2]}
                    logging.info(f"Drawing bounding box for face_{i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pil_crop_img = Image.fromarray(image[y1:y2, x1:x2])
                    rounded_scores, cam = detect_emotion(pil_crop_img)
                    if rounded_scores is None or cam is None:
                        logging.warning(f"Skipping face_{i} due to failed emotion detection")
                        continue
                    emotion_scores_dict[f'face_{i}'] = rounded_scores
                    max_emotion = update_max_emotion(rounded_scores)
                    plot_heatmap(x1, y1, x2 - x1, y2 - y1, cam, pil_crop_img, image)
                    print_max_emotion(x1, y1, max_emotion, image)
                    print_all_emotion(x1, y1, x2 - x1, rounded_scores, image)
        return faces, emotion_scores_dict
    except Exception as e:
        logging.error(f"Error in detect_bounding_box: {e}")
        return {}, {}

def detect_bounding_box_opencv(image):
    faces = {}
    emotion_scores_dict = {}
    try:
        model_file = "models/opencv_face_detector_uint8.pb"
        config_file = "models/opencv_face_detector.pbtxt"
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            logging.warning("OpenCV DNN model files not found. Skipping fallback.")
            return faces, emotion_scores_dict
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        h, w, _ = image.shape
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    logging.info(f"Skipping invalid OpenCV bounding box for face_{i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    continue
                faces[f'face_{i}'] = {'facial_area': [x1, y1, x2, y2]}
                logging.info(f"Drawing bounding box for face_{i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pil_crop_img = Image.fromarray(image[y1:y2, x1:x2])
                rounded_scores, cam = detect_emotion(pil_crop_img)
                if rounded_scores is None or cam is None:
                    logging.warning(f"Skipping face_{i} due to failed emotion detection")
                    continue
                emotion_scores_dict[f'face_{i}'] = rounded_scores
                max_emotion = update_max_emotion(rounded_scores)
                plot_heatmap(x1, y1, x2 - x1, y2 - y1, cam, pil_crop_img, image)
                print_max_emotion(x1, y1, max_emotion, image)
                print_all_emotion(x1, y1, x2 - x1, rounded_scores, image)
        return faces, emotion_scores_dict
    except Exception as e:
        logging.error(f"Error in detect_bounding_box_opencv: {e}")
        return {}, {}

def process_frame(frame, frame_id, results):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, emotion_scores_dict = detect_bounding_box(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results is not None:
            frame_results = {'frame_id': frame_id, 'faces': []}
            for face_id, data in faces.items():
                if face_id in emotion_scores_dict:
                    x1, y1, x2, y2 = data['facial_area']
                    emotion_scores = dict(zip(class_labels, [round(score, 2) for score in emotion_scores_dict[face_id]]))
                    max_emotion = max(emotion_scores, key=emotion_scores.get)
                    frame_results['faces'].append({
                        'face_id': face_id,
                        'emotion_scores': emotion_scores,
                        'max_emotion': max_emotion
                    })
            if frame_results['faces']:
                results.append(frame_results)
        return frame_bgr
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")
        return frame

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            if image is None:
                flash('Invalid image file')
                return redirect(request.url)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, emotion_scores_dict = detect_bounding_box(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            output_filename = f'processed_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, image_bgr)
            logging.info(f"Saved processed image to {output_path}")
            results = []
            for face_id, data in faces.items():
                if face_id in emotion_scores_dict:
                    scores = emotion_scores_dict[face_id]
                    emotion_scores = dict(zip(class_labels, [round(score, 2) for score in scores]))
                    max_emotion = max(emotion_scores, key=emotion_scores.get)
                    results.append({
                        'face_id': face_id,
                        'emotion_scores': emotion_scores,
                        'max_emotion': max_emotion
                    })
            hook.unregister_hook()
            return render_template('result.html', output_image=output_filename, results=results)
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + '.mp4'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                flash('Invalid video file')
                return redirect(request.url)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_filename = f'processed_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            results = []
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, frame_id, results if frame_id % 30 == 0 else None)
                out.write(processed_frame)
                frame_id += 1
            cap.release()
            out.release()
            hook.unregister_hook()
            return render_template('video_result.html', output_video=output_filename, results=results)
    return render_template('video_upload.html')

@app.route('/camera')
def camera_feed():
    return render_template('camera.html')

def generate_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, 0, None)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/video_stream')
def video_stream():
    return Response(generate_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)