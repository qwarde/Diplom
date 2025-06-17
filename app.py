import os
import cv2
import torch
import numpy as np
from datetime import datetime
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем бэкенд, который не требует GUI
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_socketio import SocketIO, emit
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from retinaface.pre_trained_models import get_model
from ResEmoteNet import ResEmoteNet
import torch.nn.functional as F
from hook import Hook
from tqdm import tqdm
from threading import Thread
import logging
from flask_cors import CORS


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Необходимо для flash-сообщений
socketio = SocketIO(app, async_mode='threading')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# Инициализация моделей (выполняется один раз при запуске приложения)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Используется устройство: {device}")

# Загрузка моделей
logger.info("Загружаем ResEmoteNet...")
model = ResEmoteNet().to(device)
checkpoint = torch.load('fer2013_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Инициализация хука
final_layer = model.conv3
hook = Hook()
hook.register_hook(final_layer)

# Инициализация MediaPipe Pose
logger.info("Инициализируем MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Инициализация DeepSORT
logger.info("Инициализируем DeepSORT...")
deepsort = DeepSort(max_age=70)

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Инициализация RetinaFace
logger.info("Инициализируем RetinaFace...")
model_retinaface = get_model("resnet50_2020-07-20", max_size=2048, device="cuda" if torch.cuda.is_available() else "cpu")
model_retinaface.eval()

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

def predict_batch(model, crops, device, transform):
    """Прогоняет список изображений через модель для предсказания эмоций."""
    model.eval()
    inputs = []
    
    for crop in crops:
        img_tensor = transform(crop).unsqueeze(0).to(device)
        inputs.append(img_tensor)
    
    inputs_batch = torch.cat(inputs, dim=0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs_batch)
    
    return outputs

def process_frame(model_retinaface, model, hook, frame, deepsort, pose, transform, device, fps, frame_num, interest_history):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model_retinaface.predict_jsons(rgb_frame)

    if not faces:
        return frame, interest_history

    crops, det_boxes = [], []
    bbs = []
    for face in faces:
        if not face['bbox']:
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

    try:
        emotion_probs = predict_batch(model, crops, device, transform)
        features = emotion_probs.cpu().numpy()
        tracks = deepsort.update_tracks(raw_detections=bbs, embeds=features, frame=frame)
        pose_result = pose.process(rgb_frame)
        pose_score = get_pose_score(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0.0

        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            if i < emotion_probs.size(0):
                emo_prob = emotion_probs[i]
                emotion_score = emotion_to_score(emo_prob)
                interest = 0.6 * emotion_score + 0.4 * pose_score
                interest_history[track_id].append({"frame": frame_num, "interest": interest})
    except Exception as e:
        logger.error(f"Error processing frame {frame_num}: {str(e)}")

    return frame, interest_history

def normalize_interests(interest_history):
    all_interests = []
    
    for track_id, values in interest_history.items():
        for value in values:
            all_interests.append(value["interest"])

    if not all_interests:
        return interest_history, 0, 1

    global_min = min(all_interests)
    global_max = max(all_interests)

    if global_max == global_min:
        return interest_history, global_min, global_max

    for track_id, values in interest_history.items():
        for value in values:
            value["interest"] = (value["interest"] - global_min) / (global_max - global_min)
    
    return interest_history, global_min, global_max

def save_results(interest_history, video_path):
    interest_history, global_min, global_max = normalize_interests(interest_history)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем график
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"plot_{base_name}_{timestamp}.png")
    plt.figure(figsize=(10, 6))
    for track_id, values in interest_history.items():
        frames = [v["frame"] for v in values]
        interests = [v["interest"] for v in values]
        plt.plot(frames, interests, label=f"Track {track_id}")
    plt.xlabel("Frame Number")
    plt.ylabel("Interest Score")
    plt.title("Interest over Time")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    # Сохраняем данные в JSON
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], f"data_{base_name}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "interest_history": interest_history,
            "global_min": global_min,
            "global_max": global_max
        }, f)
    
    return plot_path, json_path

def process_video_task(video_path, frame_skip):
    try:
        plot_path, json_path = process_video(video_path, frame_skip)
        socketio.emit('processing_complete', {
            'status': 'success',
            'plot_path': plot_path,
            'json_path': json_path
        })
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        socketio.emit('processing_complete', {
            'status': 'error',
            'message': str(e)
        })
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def process_video(video_path, frame_skip=1):
    interest_history = defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = (total_frames + frame_skip - 1) // frame_skip
    
    try:
        frame_pos = 0
        for frame_num in tqdm(range(total_frames), desc="Processing video"):
            ret = cap.grab()
            if not ret:
                break
                
            if frame_num % frame_skip == 0:
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break
                
                try:
                    frame, interest_history = process_frame(
                        model_retinaface, model, hook, frame, deepsort, 
                        pose, transform, device, fps, frame_num, interest_history
                    )
                except Exception as e:
                    logger.error(f"Error processing frame {frame_num}: {str(e)}")
                    continue
                
                frame_pos += 1
                if frame_pos % 10 == 0 or frame_pos == processed_frames:
                    socketio.emit('progress_update', {
                        'current': frame_pos,
                        'total': processed_frames,
                        'percent': int((frame_pos / processed_frames) * 100),
                        'frame_num': frame_num
                    })
                
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise
    finally:
        cap.release()
    
    if interest_history:
        plot_path, json_path = save_results(interest_history, video_path)
        return plot_path, json_path
    else:
        raise ValueError("No faces detected in the video")

@app.route('/process', methods=['POST'])
def process_video_endpoint():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    try:
        frame_skip = int(request.form.get('frame_skip', 2))
        if frame_skip < 1:
            frame_skip = 1
        
        # Создаем папку для загрузок, если её нет
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = f"{uuid.uuid4()}.{video.filename.split('.')[-1].lower()}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        
        # Запускаем обработку в отдельном потоке
        Thread(target=process_video_task, args=(video_path, frame_skip)).start()
        
        return jsonify({
            "status": "processing_started", 
            "video_id": filename,
            "message": "Video processing started successfully"
        })
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file selected')
            return redirect(request.url)
            
        video = request.files['video']
        if video.filename == '':
            flash('No video selected')
            return redirect(request.url)
        
        try:
            frame_skip = int(request.form.get('frame_skip', 1))
            if frame_skip < 1:
                frame_skip = 1
            
            filename = f"{uuid.uuid4()}.{video.filename.split('.')[-1]}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(video_path)
            
            # Запускаем обработку в отдельном потоке
            Thread(target=process_video_task, args=(video_path, frame_skip)).start()
            
            return render_template('processing.html')
            
        except Exception as e:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            flash(f'Error: {str(e)}')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/result')
def result():
    # Получаем все файлы графиков
    plot_files = sorted(
        [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith('plot_')],
        key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)),
        reverse=True
    )
    
    if not plot_files:
        flash('No analysis results found. Please analyze a video first.', 'warning')
        return redirect(url_for('index'))
    
    # Берем самый свежий график
    latest_plot = plot_files[0]
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_plot)
    
    # Проверяем существование файла
    if not os.path.exists(plot_path):
        flash('Analysis graph file not found.', 'error')
        return redirect(url_for('index'))
    
    # Ищем соответствующий JSON файл
    base_name = latest_plot.split('_', 1)[1].rsplit('.', 1)[0]
    json_file = f"data_{base_name}.json"
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_file)
    
    return render_template(
        'result.html',
        plot_path=url_for('static', filename=latest_plot),
        json_path=url_for('static', filename=json_file) if os.path.exists(json_path) else None,
        now=datetime.now().year
    )

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'favicon.ico'))

@socketio.on('connect')
def handle_connect():
    emit('progress_update', {'status': 'connected'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, debug=True, use_reloader=False)