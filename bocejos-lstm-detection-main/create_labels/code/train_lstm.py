import json
from collections import defaultdict, Counter
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from collections import Counter
import os

# ========== Parâmetros ==========
WINDOW_SIZE = 45
STEP = 3 # frames entre janelas

# ========== Carregar JSONs ==========
json_files = glob.glob("../jsons/*.json")
data = []
for path in json_files:
    with open(path, "r") as f:
        content = json.load(f)
        data.extend(content)

# print(f"Total de frames carregados: {len(data)}")

# ========== Agrupar frames por vídeo ==========
videos = defaultdict(list)
for frame in data:
    videos[frame['video_id']].append(frame)

# Ordenar os frames
for vid in videos:
    videos[vid] = sorted(videos[vid], key=lambda x: x['frame_number'])

# ========== Função para extrair vetor de features ==========
def frame_to_features(frame):
    keypoints = frame['keypoints']
    keypoint_vector = []
    for kp_id in sorted(keypoints.keys(), key=lambda x: int(x)):
        keypoint_vector.extend(keypoints[kp_id])
    return keypoint_vector




# # ========== Divisão por vídeo ==========


# forced_train_videos = [
#     "Expressoes-faciais-1_keypoints_1_euclidean_normalized2",
#     # "Expressoes-faciais-falando-1_keypoints_1_euclidean_normalized2",
#     # "microssono-bocejo-lado-direito-1_keypoints_1_euclidean_normalized2",
#     # "microssono-bocejo-lado-direito-2_keypoints_1_euclidean_normalized2",
#     # "microssono-bocejo-lado-esquerdo-1_keypoints_1_euclidean_normalized2",
#     # "microssono-bocejo-lado-esquerdo-2_keypoints_1_euclidean_normalized2"
# ]

# # ========== Divisão por vídeo ==========
# video_ids = list(videos.keys())
# print(video_ids)
# # Remove os que estão na lista forçada antes do split
# ids_para_split = [vid for vid in video_ids if vid not in forced_train_videos]
# train_ids, test_ids = train_test_split(ids_para_split, test_size=0.2, random_state=42)

# # Adiciona os vídeos forçados ao treino
# train_ids.extend(forced_train_videos)
# train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)


# ========== Intervals-only: trechos específicos de vídeos para TREINO ==========
# Formato: { video_id: [ ((m_ini, s_ini), (m_fim, s_fim)), ... ] }
forced_train_intervals = {
    "Expressoes-faciais-2_keypoints_1_euclidean_normalized2": [
       ((0, 0), (0, 20)),  # do 0:00 ao 0:20
    ],
    #"Expressoes-faciais-1_keypoints_1_euclidean_normalized2": [
    #    ((0, 0), (0, 20)),
    #],
    #"Expressoes-faciais-falando-1_keypoints_1_euclidean_normalized2": [
    #     ((0, 0), (1, 10)),
    #],
    #"microssono-bocejo-lado-direito-1_keypoints_1_euclidean_normalized2": [
    #     ((0, 0), (0, 50)),
    #],
    #"microssono-bocejo-lado-esquerdo-1_keypoints_1_euclidean_normalized2": [
    #     ((0, 0), (2, 00)),
    #],
    #"microssono-bocejo-lado-direito-2_keypoints_1_euclidean_normalized2": [ #DEU BOM
    #    ((0, 0), (1, 00)),
    #],
    #"microssono-bocejo-lado-esquerdo-2_keypoints_1_euclidean_normalized2": [ #DEU BOM
    #    ((0, 0), (1, 00)),
    #],
}


# FPS usado para converter número do frame -> tempo (segundos)
FPS = 30  # ajuste conforme seu dataset

def _frame_in_intervals(frame_number: int, intervals, fps: int) -> bool:
    """True se o frame cai em algum intervalo (mm:ss) informado; se intervals vazio/None, True."""
    if not intervals:
        return True
    t = frame_number / fps  # em segundos
    for (start_mmss, end_mmss) in intervals:
        start_total = start_mmss[0] * 60 + start_mmss[1]
        end_total   = end_mmss[0] * 60 + end_mmss[1]
        if start_total <= t <= end_total:
            return True
    return False

def _filter_frames_by_intervals(frames, intervals_map, fps: int):
    """Se o vídeo tiver intervals definidos, retorna só os frames dentro desses intervals."""
    if not frames:
        return frames
    vid = frames[0]['video_id']
    if vid in intervals_map and intervals_map[vid]:
        allowed = intervals_map[vid]
        return [f for f in frames if _frame_in_intervals(f['frame_number'], allowed, fps)]
    return frames

# ========== Divisão por vídeo ==========
video_ids = list(videos.keys())
print(video_ids)

# vídeos que serão sempre treino por estarem no forced_train_intervals
forced_train_ids = [vid for vid in forced_train_intervals.keys() if vid in video_ids]

# remove os forçados antes do split
ids_para_split = [vid for vid in video_ids if vid not in forced_train_ids]
train_ids, test_ids = train_test_split(ids_para_split, test_size=0.2, random_state=42)

# adiciona os forçados ao treino
train_ids.extend(forced_train_ids)
train_ids = list(dict.fromkeys(train_ids))  # remove duplicatas
test_ids  = [v for v in test_ids if v not in forced_train_ids]









# ========== Função para gerar janelas ==========

# label considerada a do meio
def extract_windows(frames):
    X, y = [], []
    for i in range(0, len(frames) - WINDOW_SIZE + 1, STEP):
        window = frames[i:i + WINDOW_SIZE]
        x_window = [frame_to_features(f) for f in window]
        label = window[len(window) // 2]['driver_state']
        X.append(x_window)
        y.append(label)
    return X, y

# # label considerada a que mais aparecer
# def extract_windows(frames):
#     X, y = [], []
#     for i in range(0, len(frames) - WINDOW_SIZE + 1, STEP):
#         window = frames[i:i + WINDOW_SIZE]
#         x_window = [frame_to_features(f) for f in window]

#         # Extrair todas as labels da janela
#         labels_in_window = [f['driver_state'] for f in window]
        
#         # Obter a label mais comum
#         most_common_label = Counter(labels_in_window).most_common(1)[0][0]
        
#         X.append(x_window)
#         y.append(most_common_label)
#     return X, y






# ========== Montar conjuntos ==========
X_train, y_train, X_test, y_test = [], [], [], []

# for vid in train_ids:
#     x, y = extract_windows(videos[vid])
#     X_train.extend(x)
#     y_train.extend(y)

# for vid in test_ids:
#     x, y = extract_windows(videos[vid])
#     X_test.extend(x)
#     y_test.extend(y)


for vid in train_ids:
    frames = videos[vid]
    frames = _filter_frames_by_intervals(frames, forced_train_intervals, FPS)
    x, y = extract_windows(frames)
    X_train.extend(x)
    y_train.extend(y)

for vid in test_ids:
    frames = videos[vid]
    x, y = extract_windows(frames)
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

# ========== Ver distribuição ==========
print("\nDistribuição de labels no treino:")
train_counts = Counter(y_train)
for label in sorted(train_counts):
    print(f"Label {label}: {train_counts[label]}")

print("\nDistribuição de labels no teste:")
test_counts = Counter(y_test)
for label in sorted(test_counts):
    print(f"Label {label}: {test_counts[label]}")

# ========== Class weights ==========
# classes = np.unique(y_train)

# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(class_weights))
# print("\nPesos por classe:", class_weights)

# ========== Modelo ==========
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ========== Early stopping ==========
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ========== Treinamento ==========
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_test, y_test),
          # class_weight=class_weights,
          callbacks=[early_stop])

# ========== Avaliação ==========
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nLoss no conjunto de teste:", test_loss)
print("Acurácia no conjunto de teste:", test_accuracy)

# ========== Previsões ==========
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\nPrimeiras previsões:", y_pred_classes[:10])
print("Primeiras labels reais:", y_test[:10])

# ========== Relatório ==========
from sklearn.metrics import classification_report

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred_classes, target_names=["Alerta", "Bocejando", "Microsleep"]))

# ========== Exportação ==========
model.export('model_tensorflow')