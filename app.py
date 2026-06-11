from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging
from pydub import AudioSegment
from scipy.spatial.distance import cosine

# ----------------------------
# Parameters
# ----------------------------
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
MAX_LEN = 128

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("sound_cnn_final1.h5")
print("✅ CNN Model loaded successfully!")

class_labels = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]
# Pre-warm librosa/numba at startup to avoid timeout on first request
import warnings
warnings.filterwarnings("ignore")
print("⏳ Pre-warming librosa...")
_dummy = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)
_ = librosa.feature.melspectrogram(y=_dummy, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=N_MELS)
_ = librosa.feature.mfcc(y=_dummy, sr=SAMPLE_RATE, n_mfcc=13)
print("✅ Librosa pre-warmed!")
# ----------------------------
# Folder setup
# ----------------------------
UPLOAD_FOLDER = "uploads"
CUSTOM_SOUNDS_DIR = "custom_sounds"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_SOUNDS_DIR, exist_ok=True)

# ----------------------------
# Convert any format → WAV
# ----------------------------
def convert_to_wav(input_path):
    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(wav_path, format="wav")
    return wav_path

# ----------------------------
# Compute MFCC for similarity
# ----------------------------
def compute_mfcc(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def find_best_custom_match(file_path, user_id):
    best_score = 1.0
    best_match = None
    user_files = [f for f in os.listdir(CUSTOM_SOUNDS_DIR) if f.startswith(user_id)]
    if not user_files:
        return None

    try:
        target_mfcc = compute_mfcc(file_path)
    except Exception as e:
        print(f"❌ Failed to compute MFCC for input: {e}")
        return None

    for f in user_files:
        ref_path = os.path.join(CUSTOM_SOUNDS_DIR, f)
        try:
            ref_mfcc = compute_mfcc(ref_path)
            score = cosine(target_mfcc, ref_mfcc)
            if score < best_score:
                best_score = score
                best_match = f
        except Exception as e:
            print(f"⚠️ Error comparing {f}: {e}")

    if best_score < 0.25:
        print(f"🔔 Custom match found: {best_match} (similarity={1-best_score:.2f})")
        return best_match
    return None

# ----------------------------
# Preprocessing for CNN
# ----------------------------
def preprocess_audio_for_prediction(audio_path):
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    else:
        pad_width = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, pad_width), "constant")

    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_LEN:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_LEN]
    spectrogram = mel_db[np.newaxis, ..., np.newaxis]
    return spectrogram

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Upload custom sounds
# ----------------------------

@app.route("/upload_custom_sound", methods=["POST"])
def upload_custom_sound():
    if 'file' not in request.files or 'user_id' not in request.form or 'name' not in request.form:
        return jsonify({"error": "Missing file, user_id, or name"}), 400

    user_id = request.form['user_id']
    name = secure_filename(request.form['name'])
    file = request.files['file']

    # Save temporary file
    temp_path = os.path.join(CUSTOM_SOUNDS_DIR, secure_filename(file.filename))
    file.save(temp_path)

    try:
        # ✅ If already WAV, skip conversion
        if temp_path.lower().endswith(".wav"):
            final_path = os.path.join(CUSTOM_SOUNDS_DIR, f"{user_id}_{name}.wav")
            os.rename(temp_path, final_path)
        else:
            wav_path = convert_to_wav(temp_path)
            final_path = os.path.join(CUSTOM_SOUNDS_DIR, f"{user_id}_{name}.wav")
            os.rename(wav_path, final_path)
            os.remove(temp_path)

        logging.info(f"🎙️ Custom sound saved: {final_path}")
        return jsonify({"status": "success", "file": final_path})
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Predict route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['file']
    user_id = request.form.get('user_id', 'default_user')
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    wav_path = convert_to_wav(filepath)
    logging.info(f"📁 Received: {filename}")

    important_sounds = ["dog_bark", "car_horn", "siren"]

    match = find_best_custom_match(wav_path, user_id)
    if match:
        label = match.split('_', 1)[-1].replace('.wav', '')
        if label in important_sounds:
            return jsonify({
                "type": "custom",
                "class": label,
                "confidence": 1.0
            })
        else:
            return jsonify({"type": "ignored", "class": None, "confidence": 0.0})

    try:
        spectrogram = preprocess_audio_for_prediction(wav_path)
        preds = model.predict(spectrogram)
        predicted_class_index = int(np.argmax(preds))
        predicted_label = class_labels[predicted_class_index]
        confidence = float(np.max(preds))

        if predicted_label in important_sounds:
            return jsonify({
                "type": "model",
                "class": predicted_label,
                "confidence": round(confidence, 3)
            })
        else:
            return jsonify({"type": "ignored", "class": None, "confidence": 0.0})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Test endpoint
# ----------------------------
@app.route("/test", methods=["GET"])
def test():
    return "✅ Flask server running fine!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
