import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import onnxruntime as ort
import time
import winsound
import threading
import os
import wave
import struct
import math

# ==== Parâmetros ====
WINDOW_SIZE = 45
MODEL_PATH = "./create_labels/code/documentos/model_lstm_3_45_euclidean.onnx"

# ==== Inicializar MediaPipe ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==== Índices dos keypoints usados ====
FACE_IDX = {
    "mouth_outer":   [61, 40, 37, 0, 267, 270, 291],
    "mouth_inner":   [78, 95, 14, 317, 308],
    "left_eye":      [33, 160, 158, 133, 153, 144, 145],
    "right_eye":     [362, 385, 387, 263, 373, 380, 381],
    "left_brow":     [70, 105, 107],
    "right_brow":    [336, 334, 300],
    "nose":          [1, 2, 168],
    "chin":          [152]
}
all_landmark_indices = sorted(set(idx for group in FACE_IDX.values() for idx in group))

# ==== ONNX Runtime ====
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ==== Janela de features ====
window = deque(maxlen=WINDOW_SIZE)

# ==== SOM (Microsleep) com WAV contínuo em loop ====
TONE_FS = 44100
TONE_SEC = 1.0

MICROSLEEP_FREQ_HZ = 1200
MICROSLEEP_WAV = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "microsleep_tone.wav")
microsleep_playing = False
microsleep_lock = threading.Lock()

# ==== Bocejo (2 apitos mais fortes: 2 tons agudos) ====
YAWN_FREQ_1_HZ = 1700   # 1º beep (agudo)
YAWN_FREQ_2_HZ = 2000   # 2º beep (mais agudo, mais “cortante”)
YAWN_BEEP1_MS  = 180
YAWN_BEEP2_MS  = 230
YAWN_GAP_MS    = 90
yawn_playing = False
yawn_lock = threading.Lock()
yawn_stop_event = threading.Event()

def ensure_tone_wav(path, freq, fs=TONE_FS, seconds=TONE_SEC):
    if os.path.exists(path):
        return
    num_samples = int(fs * seconds)
    amp = 32767
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        for n in range(num_samples):
            sample = int(amp * math.sin(2 * math.pi * freq * (n / fs)))
            wf.writeframes(struct.pack('<h', sample))

def purge_sound():
    """Libera imediatamente qualquer PlaySound pendente (Windows)."""
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass

# === Som: Microsleep (contínuo em loop) ===
def start_microsleep_beep():
    global microsleep_playing
    with microsleep_lock:
        if microsleep_playing:
            return
        ensure_tone_wav(MICROSLEEP_WAV, MICROSLEEP_FREQ_HZ, TONE_FS, TONE_SEC)
        winsound.PlaySound(MICROSLEEP_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        microsleep_playing = True
        # prioridade: interrompe bocejo se estiver tocando
        stop_yawn_beep()
        print("[ALERTA] Microsleep detectado - apito contínuo iniciado")

def stop_microsleep_beep():
    global microsleep_playing
    with microsleep_lock:
        if not microsleep_playing:
            return
        winsound.PlaySound(None, 0)   # para loop
        purge_sound()                 # libera device
        microsleep_playing = False
        print("[INFO] Microsleep terminou - apito parado")

# === Som: Bocejo (2 beeps mais fortes) ===
def start_yawn_beep():
    global yawn_playing
    with yawn_lock:
        if yawn_playing or microsleep_playing:
            return
        yawn_playing = True
        yawn_stop_event.clear()

    def _pattern():
        global yawn_playing
        try:
            purge_sound()
            time.sleep(0.01)

            print("[ALERTA] (SOM) Bocejo -> Beep 1 (1700Hz)")
            if not yawn_stop_event.is_set():
                winsound.Beep(YAWN_FREQ_1_HZ, YAWN_BEEP1_MS)

            if not yawn_stop_event.is_set():
                time.sleep(YAWN_GAP_MS / 1000.0)

            print("[ALERTA] (SOM) Bocejo -> Beep 2 (2000Hz)")
            if not yawn_stop_event.is_set():
                winsound.Beep(YAWN_FREQ_2_HZ, YAWN_BEEP2_MS)
        finally:
            with yawn_lock:
                yawn_playing = False

    threading.Thread(target=_pattern, daemon=True).start()

def stop_yawn_beep():
    with yawn_lock:
        if not yawn_playing:
            return
        yawn_stop_event.set()
        print("[INFO] Alerta bocejo interrompido")

# ==== Estado de bocejo ====
yawn_start = None
yawn_hold_until_release = False
YAWN_CONFIRM_SECONDS = 4.0
YAWN_WINDOW_SECONDS = 60.0

yawn_confirms = deque(maxlen=10)
last_yawn_alert_time = 0.0

# ==== Webcam ====
cap = cv2.VideoCapture(0)

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

# ==== FUNÇÃO PARA EAR ====
def eye_aspect_ratio(landmarks, eye_idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_idx[:6]]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

left_eye_idx = 33
right_eye_idx = 263

current_label = None
confidence = 0.0  # nível de confiabilidade

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    keypoints = []
    EAR_value = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ==== CALCULAR EAR ====
        EAR_left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
        EAR_right = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        EAR_value = (EAR_left + EAR_right) / 2.0

        cv2.putText(image, f"EAR: {EAR_value:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cx, cy = landmarks[1].x, landmarks[1].y
        if left_eye_idx < len(landmarks) and right_eye_idx < len(landmarks):
            scale = euclidean(landmarks[left_eye_idx], landmarks[right_eye_idx])
        else:
            scale = 1.0
        if scale < 1e-5:
            scale = 1.0
        for idx in all_landmark_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = (lm.x - cx) / scale
                y = (lm.y - cy) / scale
                keypoints.extend([x, y])
                px = int(lm.x * image.shape[1])
                py = int(lm.y * image.shape[0])
                cv2.circle(image, (px, py), 2, (0, 255, 0), -1)
            else:
                keypoints.extend([0.0, 0.0])
    else:
        keypoints = [0.0, 0.0] * len(all_landmark_indices)

    window.append(keypoints)

    if len(window) == WINDOW_SIZE:
        x_input = np.array([window], dtype=np.float32)
        prediction = session.run(None, {input_name: x_input})[0][0]

        if prediction.shape[-1] > 1:
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))  # pega maior probabilidade
        else:
            predicted_class = int(prediction > 0.5)
            confidence = float(prediction if predicted_class == 1 else 1 - prediction)

        label_str = ["alerta", "bocejo", "microsleep"][predicted_class]

        if label_str != current_label:
            if label_str in ["bocejo", "microsleep"]:
                print(f"[LOG] Label detectada: {label_str}")
            current_label = label_str
    else:
        label_str = "Carregando..."
        confidence = 0.0

    cv2.putText(image, f"Estado: {label_str} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    cv2.imshow("Webcam - Microsleep + Bocejo + EAR", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stop_microsleep_beep()
stop_yawn_beep()
cap.release()
cv2.destroyAllWindows()
