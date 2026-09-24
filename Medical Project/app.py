import os
import setuptools  # Provides distutils shim for Python 3.12+
from io import BytesIO

# ── Standard & Web Framework ──────────────────────────────────────────────────
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import nltk
from nltk.chat.util import Chat, reflections

# ── PyTorch (ResNet50) ────────────────────────────────────────────────────────
import torch
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50

# ── TensorFlow (EfficientNetB0 / kaggle_model) ────────────────────────────────
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras
import tensorflow as tf
import cv2

# =============================================================================
# Flask Application Setup
# =============================================================================
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='template', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "neuroai_secure_session_key_2026"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Labels for each model (order must match how each model was trained)
LABELS_PYTORCH = ['None', 'Meningioma', 'Glioma', 'Pituitary']
LABELS_TF      = ['Glioma', 'No Tumor', 'Meningioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"

# ── Load PyTorch Model ────────────────────────────────────────────────────────
print("[INFO] Loading PyTorch model...")
resnet_model = resnet50(weights=None)

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
resnet_path = os.path.join(os.path.dirname(__file__), 'assets', 'bt_resnet50_model.pt')
if os.path.exists(resnet_path):
    resnet_model.load_state_dict(load(resnet_path, map_location=DEVICE(device)))
    resnet_model.eval()
    print("[INFO] PyTorch model loaded ✓")
else:
    print(f"[WARN] PyTorch weights not found at {resnet_path}")

# ── Load TensorFlow Model ─────────────────────────────────────────────────────
print("[INFO] Loading TensorFlow model...")
tf_model_path = os.path.join(os.path.dirname(__file__), 'assets', 'kaggle_model.h5')
if os.path.exists(tf_model_path):
    tf_model = keras.models.load_model(tf_model_path)
    print("[INFO] TensorFlow model loaded ✓")
else:
    tf_model = None
    print(f"[WARN] TensorFlow model not found at {tf_model_path}")

# ── Setup NeuroCare Chatbot ───────────────────────────────────────────────────
try:
    nltk.download('punkt', quiet=True)
except Exception:
    pass

chatbot_pairs = [
    (r"(.*)(hello|hi|hey)(.*)", ["Hello! I'm Snow, your NeuroCare companion. How can I help you today?"]),
    (r"(.*)(your name|who are you)(.*)", ["My name is Snow, your AI health and recovery assistant."]),
    (r"(.*)(side effects|symptoms)(.*)", ["The side effects of a brain tumor depend on its size, location, and rate of growth. Common symptoms include persistent headaches, nausea/vomiting, concentration issues, balance difficulties, or vision changes."]),
    (r"(.*)(emergency|severe)(.*)", ["In case of an emergency, sudden severe symptoms, or loss of consciousness, call emergency services immediately or visit the nearest emergency room."]),
    (r"(.*)(activity|exercise|walk)(.*)", ["Engage in gentle walking aiming for 30–60 minutes daily if cleared by your physician. Adjust according to comfort level."]),
    (r"(.*)(recovery|rehab)(.*)", ["Recovery involves follow-up appointments, medication management, physical/occupational therapy, cognitive rehabilitation, and consistent medical monitoring."]),
    (r"(.*)(seizure|epileptic)(.*)", ["During a seizure: keep the person away from sharp objects, turn them on their side to maintain an open airway, and seek immediate emergency medical care."]),
    (r"(.*)(fatigue|tired)(.*)", ["Fatigue is common during treatment. Ensure sufficient rest, stay hydrated, and discuss chronic fatigue with your healthcare team."]),
    (r"(.*)(headache|unbalanced|dizzy)(.*)", ["Persistent headaches or dizziness warrant medical review. Note their timing, intensity, and any triggers to share with your physician."]),
    (r"(.*)(glioma)(.*)", ["Gliomas originate in the glial cells of the brain or spine. Treatment typically involves surgical resection, radiation therapy, and chemotherapy tailored by your neuro-oncologist."]),
    (r"(.*)(meningioma)(.*)", ["Meningiomas arise from the meninges surrounding the brain. Many are benign and slow-growing; treatment may involve active surveillance or surgical removal."]),
    (r"(.*)(pituitary)(.*)", ["Pituitary tumors form in the pituitary gland affecting hormone balance. Many respond well to endoscopic surgery, medication, or targeted radiation."]),
    (r"(.*)(thank you|thanks)(.*)", ["You're very welcome! If you have more questions or need guidance, I am always here to assist."]),
]

chatbot = Chat(chatbot_pairs, reflections)

# =============================================================================
# Prediction Functions
# =============================================================================

def predict_pytorch(image_bytes):
    """Run inference using PyTorch ResNet50 model."""
    try:
        transform = Compose([Resize((512, 512)), ToTensor()])
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            y_hat = resnet_model(tensor.to(device))
            class_id = int(argmax(y_hat.data, dim=1))
        return LABELS_PYTORCH[class_id]
    except Exception as e:
        print(f"[ERROR PyTorch] {e}")
        return "Unknown"


def predict_tensorflow(image_bytes):
    """Run inference using TensorFlow EfficientNetB0 model."""
    if tf_model is None:
        return "Model Unavailable"
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (150, 150))
        arr = np.array(img_bgr, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        preds = tf_model.predict(arr, verbose=0)
        class_id = int(np.argmax(preds, axis=1)[0])
        return LABELS_TF[class_id]
    except Exception as e:
        print(f"[ERROR TensorFlow] {e}")
        return "Unknown"


# =============================================================================
# Routes
# =============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'bt_image' not in request.files:
            return redirect(request.url)
        
        f = request.files['bt_image']
        if f.filename == '':
            return redirect(request.url)
        
        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext in ALLOWED_EXTENSIONS:
            file_bytes = f.read()
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(save_path, 'wb') as out_f:
                out_f.write(file_bytes)

            pytorch_res = predict_pytorch(file_bytes)
            tf_res      = predict_tensorflow(file_bytes)

            session['pred_label'] = [pytorch_res, tf_res]
            session['filename']   = filename
            return redirect(url_for('pred_page'))

    return render_template('index.html')


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    if pred is None or f_name is None:
        return redirect(url_for('index'))
    return render_template('pred.html', pred=pred, f_name=f_name)


@app.route('/empty_page')
def empty_page():
    filename = session.get('filename', None)
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Only remove if it was an uploaded file, not sample images
        if os.path.exists(file_path) and not filename.startswith(('glioma_', 'Te-', 'pituitary_')):
            try:
                os.remove(file_path)
            except Exception:
                pass
    session.pop('pred_label', None)
    session.pop('filename', None)
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict_api():
    """REST API endpoint for programmatic classification."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    img_bytes = file.read()

    pytorch_result = predict_pytorch(img_bytes)
    tf_result      = predict_tensorflow(img_bytes)

    return jsonify({
        'pytorch_pred': pytorch_result,
        'tf_pred':      tf_result
    })


@app.route('/chat', methods=['POST'])
def chat():
    """NeuroCare chatbot response endpoint."""
    user_input = request.form.get("user_input", "")
    response = chatbot.respond(user_input)
    if not response:
        response = "I understand your query. For personalized diagnostic and treatment advice, please consult your healthcare team."
    return jsonify({"response": response})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NeuroAI Server Running: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
