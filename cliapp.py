import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import json
import argparse
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from ResEmoteNet import ResEmoteNet
import torch.nn.functional as F
from hook import Hook

def get_pose_score(landmarks):
    if landmarks is None:
        return 0.0
    left = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    if left.visibility > 0.5 and right.visibility > 0.5:
        return 1.0
    return 0.5

def emotion_to_score(emotion_probs):
    probs = torch.softmax(emotion_probs, dim=0)
    weights = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5]).to(probs.device)
    return float((emotion_probs * weights).sum().item())

def detect_emotion(model, hook, pil_crop_img, transform, device):
    vid_fr_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
    logits = model(vid_fr_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    predicted_class_idx = predicted_class.item()

    one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_()
    one_hot_output[0][predicted_class_idx] = 1
    logits.backward(one_hot_output, retain_graph=True)

    gradients = hook.backward_out
    feature_maps = hook.forward_out

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
    cam = cam.clamp(min=0).squeeze()

    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()

    scores = probabilities.cpu().detach().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores, cam

def update_max_emotion(rounded_scores):
    class_labels = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    max_index = np.argmax(rounded_scores)
    max_emotion = class_labels[max_index]
    return max_emotion

def process_frame(model_retinaface, model, hook, frame, deepsort, pose, transform, device, fps, frame_num, interest_history, skip_frames):
    # Пропуск кадров по шагу
    if frame_num % skip_frames != 0:
        return frame, interest_history

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model_retinaface.predict_jsons(rgb_frame) 

    if not faces:
        return frame, interest_history

    crops, det_boxes = [], []
    bbs = []
    for face in faces:
        if face['bbox'] == []:
            continue
        x1, y1, x2, y2 = face['bbox']
        face_crop = rgb_frame[int(y1):int(y2), int(x1):int(x2)]
        if face_crop.size == 0:
            continue
        pil_crop = Image.fromarray(face_crop).convert('RGB')
        crops.append(pil_crop)
        det_boxes.append([x1, y1, x2 - x1, y2 - y1])  
        bbs.append(([x1, y1, x2 - x1, y2 - y1], face['score'], 0))

    if not crops:
        return frame, interest_history

    emotion_probs = predict_batch(model, crops, device, transform)
    features = emotion_probs.cpu().numpy()
    tracks = deepsort.update_tracks(raw_detections=bbs, embeds=features, frame=frame)
    pose_result = pose.process(rgb_frame)
    pose_score = get_pose_score(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0.0
    try:
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            emo_prob = emotion_probs[i]
            emotion_score = emotion_to_score(emo_prob)
            interest = 0.6 * emotion_score + 0.4 * pose_score
            time_sec = frame_num / fps

            # Сохраняем интерес без нормализации
            interest_history[track_id].append({"frame": time_sec, "interest": interest})

            # Отображение интереса на кадре
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} Int:{interest:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except:
        print("Ошибка в обработке треков")
    return frame, interest_history

def plot_interest(interest_history, output_path="interest_plot.png"):
    plt.figure(figsize=(12, 6))
    for track_id, values in interest_history.items():
        times = [v["frame"] for v in values]
        interests = [v["interest"] for v in values]
        plt.plot(times, interests, label=f"Track {track_id}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Interest Level")
    plt.title("Interest Level Over Time per Track")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def normalize_interests(interest_history):
    all_interests = []
    
    # Собираем все интересы для нормализации
    for track_id, values in interest_history.items():
        for value in values:
            all_interests.append(value["interest"])

    global_min = min(all_interests)
    global_max = max(all_interests)

    # Нормализуем интересы
    for track_id, values in interest_history.items():
        for value in values:
            value["interest"] = (value["interest"] - global_min) / (global_max - global_min)  # Нормализация в [0, 1]
    
    return interest_history, global_min, global_max

def save_interest_json(interest_history, output_file="interest_by_track.json"):
    with open(output_file, "w") as f:
        json.dump(interest_history, f, indent=2)

def predict_batch(model, crops, device, transform):
    """Прогоняет список изображений через модель для предсказания эмоций."""
    model.eval()
    inputs = []
    
    # Преобразуем все изображения в тензоры и добавляем в список
    for crop in crops:
        img_tensor = transform(crop).unsqueeze(0).to(device)
        inputs.append(img_tensor)
    
    # Объединяем все изображения в один батч
    inputs_batch = torch.cat(inputs, dim=0).to(device)
    
    # Прогоняем через модель
    with torch.no_grad():
        outputs = model(inputs_batch)
    
    return outputs

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    model_retinaface = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    model_retinaface.eval()
    
    print("Загружаем ResEmoteNet...")
    model = ResEmoteNet().to(device)
    checkpoint = torch.load('fer2013_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    final_layer = model.conv3
    hook = Hook()
    hook.register_hook(final_layer)

    print("Инициализируем MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    print("Инициализируем DeepSORT...")
    deepsort = DeepSort(max_age=70)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    interest_history = defaultdict(list)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    skip_frames = args.skip_frames
    print(f"Обрабатываем видео (Всего кадров: {total_frames})...")
    
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                
                frame, interest_history = process_frame(model_retinaface, model, hook, frame, deepsort, pose, transform, device, fps, frame_num, interest_history, skip_frames)

                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

                pbar.update(1)

        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Нормализация интереса...")
    interest_history, global_min, global_max = normalize_interests(interest_history)

    print("Сохраняем результаты...")
    save_interest_json(interest_history)
    plot_interest(interest_history)

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help="Путь к видеофайлу")
    parser.add_argument('--skip_frames', type=int, default=1, help="Количество кадров для пропуска (по умолчанию: 1)")
    args = parser.parse_args()

    main(args)
