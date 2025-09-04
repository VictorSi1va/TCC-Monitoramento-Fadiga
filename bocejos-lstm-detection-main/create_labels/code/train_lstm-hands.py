import json
import glob
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping


# ========== Parâmetros ==========
WINDOW_SIZE = 45
STEP = 3

# ========== Carregar todos os JSONs ==========
json_files = glob.glob("../jsons/*.json")
data = []

for path in json_files:
    with open(path, "r") as f:
        content = json.load(f)
        data.extend(content)

print(f"Total de frames carregados: {len(data)}")

# ========== Agrupar por vídeo ==========
videos = defaultdict(list)
for frame in data:
    videos[frame['video_id']].append(frame)

# Ordenar por frame_number
for vid in videos:
    videos[vid] = sorted(videos[vid], key=lambda x: x['frame_number'])

# ========== Converter frame para vetor de features ==========
def frame_to_features(frame):
    keypoint_vector = []

    # Face
    keypoints = frame.get('keypoints', {})
    for kp_id in sorted(keypoints.keys(), key=lambda x: int(x)):
        keypoint_vector.extend(keypoints[kp_id])

    # Hands
    hands = frame.get('hands', {})
    for side in ["left", "right"]:
        hand_points = hands.get(side, {})
        for i in range(21):
            coords = hand_points.get(str(i), [0.0, 0.0])
            keypoint_vector.extend(coords)

    return keypoint_vector

# ========== Função para extrair janelas ==========
def extract_windows(frames):
    X, y = [], []
    for i in range(0, len(frames) - WINDOW_SIZE + 1, STEP):
        window = frames[i:i + WINDOW_SIZE]
        x_window = [frame_to_features(f) for f in window]
        label = window[len(window) // 2]['driver_state']  # usar label do frame central
        X.append(x_window)
        y.append(label)
    return X, y

# ========== Dividir por vídeo ==========
video_ids = list(videos.keys())
train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

X_train, y_train, X_test, y_test = [], [], [], []

for vid in train_ids:
    x, y = extract_windows(videos[vid])
    X_train.extend(x)
    y_train.extend(y)

for vid in test_ids:
    x, y = extract_windows(videos[vid])
    X_test.extend(x)
    y_test.extend(y)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ========== Distribuição das classes ==========
train_counts = Counter(y_train)
test_counts = Counter(y_test)

print("\nDistribuição de labels no treino:")
for label in sorted(train_counts.keys()):
    print(f"Label {label}: {train_counts[label]}")

print("\nDistribuição de labels no teste:")
for label in sorted(test_counts.keys()):
    print(f"Label {label}: {test_counts[label]}")

# ========== Calcular class weights ==========
# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))

# print("\nPesos por classe:", class_weights)

# ========== Early stopping ==========
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ========== Criar modelo ==========
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ========== Treinar ==========
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_test, y_test),
          # class_weight=class_weights,
          callbacks=[early_stop])

# ========== Avaliar ==========
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nLoss no conjunto de teste:", test_loss)
print("Acurácia no conjunto de teste:", test_accuracy)

# ========== Prever ==========
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\nPrimeiras previsões:", y_pred_classes[:10])
print("Primeiras labels reais:", y_test[:10])

# ========== Relatório de classificação ==========
from sklearn.metrics import classification_report

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_classes, target_names=["Alerta", "Bocejando", "Microsleep"]))

# ========== Exportar modelo ==========
model.export('model_tensorflow')