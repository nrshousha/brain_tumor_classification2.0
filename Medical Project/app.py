import os

# ── Standard ──────────────────────────────────────────────────────────────────
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

# ── PyTorch (ResNet50) ────────────────────────────────────────────────────────
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50

# ── TensorFlow (EfficientNetB0 / kaggle_model) ────────────────────────────────
import tensorflow as tf

# =============================================================================
app = Flask(__name__)

# Labels for each model (order must match how each model was trained)
LABELS_PYTORCH = ['None', 'Meningioma', 'Glioma', 'Pituitary']
LABELS_TF      = ['Glioma', 'No Tumor', 'Meningioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"

# ── Load PyTorch Model ────────────────────────────────────────────────────────
print("[INFO] Loading PyTorch model...")
resnet_model = resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(n_inputs, 2048), SELU(), Dropout(p=0.4),
    Linear(2048, 2048),     SELU(), Dropout(p=0.4),
    Linear(2048, 4),        LogSigmoid()
)
for _, child in resnet_model.named_children():
    for _, param in child.named_parameters():
        param.requires_grad = True

resnet_model.to(device)
resnet_model.load_state_dict(load('assets/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.eval()
print("[INFO] PyTorch model loaded ✓")

# ── Load TensorFlow Model ─────────────────────────────────────────────────────
print("[INFO] Loading TensorFlow model...")
tf_model = tf.keras.models.load_model('assets/kaggle_model.h5')
print("[INFO] TensorFlow model loaded ✓")

# =============================================================================
# Prediction functions
# =============================================================================

def predict_pytorch(image_bytes):
    """Run inference using PyTorch ResNet50 model."""
    transform = Compose([Resize((512, 512)), ToTensor()])
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    y_hat = resnet_model(tensor.to(device))
    class_id = int(argmax(y_hat.data, dim=1))
    return LABELS_PYTORCH[class_id]


def predict_tensorflow(image_bytes):
    """Run inference using TensorFlow EfficientNetB0 model.
    
    Preprocessing must match TFModel_Training.py exactly:
    - cv2.imread() loads images in BGR format
    - No normalization applied (raw pixel values 0-255)
    """
    import cv2
    # Decode image bytes → numpy array (BGR like cv2, no normalization)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)          # BGR, [0-255]
    img_bgr = cv2.resize(img_bgr, (150, 150))                 # same size as training
    arr = np.array(img_bgr, dtype=np.float32)                 # NO /255 - model trained on raw values
    arr = np.expand_dims(arr, axis=0)                         # shape: (1, 150, 150, 3)
    preds = tf_model.predict(arr, verbose=0)
    class_id = int(np.argmax(preds, axis=1)[0])
    return LABELS_TF[class_id]


# =============================================================================
# Route
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()

    pytorch_result = predict_pytorch(img_bytes)
    tf_result      = predict_tensorflow(img_bytes)

    return jsonify({
        'pytorch_pred': pytorch_result,   # shown under PyTorch logo
        'tf_pred':      tf_result         # shown under TensorFlow logo
    })


if __name__ == '__main__':
    app.run(port=5000)
