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
            # Libera qualquer PlaySound antes de Beep
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
yawn_start = None                # início para contar 4s
yawn_hold_until_release = False  # confirma 1x por episódio
YAWN_CONFIRM_SECONDS = 4.0
YAWN_WINDOW_SECONDS = 60.0

# Histórico de confirmações (últimos 60s)
yawn_confirms = deque(maxlen=10)   # timestamps dos bocejos confirmados
last_yawn_alert_time = 0.0         # timestamp do 2º bocejo do último alerta

# ==== Webcam ====
cap = cv2.VideoCapture(0)

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

left_eye_idx = 33
right_eye_idx = 263

current_label = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    keypoints = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
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
        predicted_class = int(np.argmax(prediction)) if prediction.shape[-1] > 1 else int(prediction > 0.5)
        label_str = ["alerta", "bocejo", "microsleep"][predicted_class]

        if label_str != current_label:
            if label_str in ["bocejo", "microsleep"]:
                print(f"[LOG] Label detectada: {label_str}")
            current_label = label_str

        now = time.time()

        # --- MICROSLEEP (som contínuo em loop) ---
        if label_str == "microsleep":
            start_microsleep_beep()
            stop_yawn_beep()  # prioridade
        else:
            stop_microsleep_beep()

        # --- BOCEJO: confirmar >= 4s contínuos ---
        if label_str == "bocejo":
            if not yawn_hold_until_release:
                if yawn_start is None:
                    yawn_start = now
                elif (now - yawn_start) >= YAWN_CONFIRM_SECONDS:
                    yawn_confirms.append(now)
                    yawn_hold_until_release = True
                    yawn_start = None
                    print("[EVENTO] Bocejo confirmado (>=4s)")
        else:
            yawn_start = None
            yawn_hold_until_release = False

        # manter confirmações dos últimos 60s
        while yawn_confirms and (now - yawn_confirms[0]) > YAWN_WINDOW_SECONDS:
            yawn_confirms.popleft()

        # --- Gatilho: 2 bocejos confirmados em <=60s (ambos depois do último alerta) ---
        if len(yawn_confirms) >= 2:
            prev, last = yawn_confirms[-2], yawn_confirms[-1]
            if prev > last_yawn_alert_time and last > last_yawn_alert_time:
                if (last - prev) <= YAWN_WINDOW_SECONDS:
                    if not microsleep_playing:
                        start_yawn_beep()   # purge + 2 beeps fortes
                    last_yawn_alert_time = last

    else:
        label_str = "Carregando..."

    cv2.putText(image, f"Estado: {label_str}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    cv2.imshow("Webcam - Microsleep + Bocejo (2 beeps)", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# Encerramento
stop_microsleep_beep()
stop_yawn_beep()
cap.release()
cv2.destroyAllWindows()
