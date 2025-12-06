import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pathlib
from flask import current_app


class WeaponDetector:
    
    _model = None
    _model_path = None
    
    WEAPON_MAPPING = {
        'sword': 'Kiếm',
        'spear': 'Thương',
        'stick': 'Côn',
    }
    
    @classmethod
    def _get_model_path(cls):
        if cls._model_path is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_file_dir, 'best.pt')
            cls._model_path = model_path
        return cls._model_path
    
    @classmethod
    def _load_model(cls):
        if cls._model is None:
            model_path = cls._get_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            cls._model = YOLO(model_path)
        return cls._model
    
    @classmethod
    def detect_from_video(cls, video_path: str) -> dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        model = cls._load_model()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"[WeaponDetector] Không thể đọc frame đầu tiên từ video", flush=True)
            return {'detected_weapon': None, 'confidence': 0.0, 'detection_count': 0, 'total_samples': 1}
        
        import sys
        print(f"[WeaponDetector] Đang chạy detection trên frame đầu tiên...", flush=True)
        results = model(frame, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id].lower()
                    
                    print(f"[WeaponDetector] Detected: class_id={cls_id}, name='{cls_name}', confidence={conf:.2%}", flush=True)
                    
                    weapon_name = cls._map_weapon_name(cls_name)
                    if weapon_name:
                        detections.append({
                            'weapon': weapon_name,
                            'confidence': conf
                        })
                        print(f"[WeaponDetector] Mapped '{cls_name}' -> '{weapon_name}'", flush=True)
                    else:
                        print(f"[WeaponDetector] WARNING: No mapping for class name '{cls_name}' (class_id={cls_id})", flush=True)
        
        if not detections:
            print(f"[WeaponDetector] Không phát hiện vũ khí nào trong frame đầu tiên", flush=True)
            sys.stdout.flush()
            return {'detected_weapon': None, 'confidence': 0.0, 'detection_count': 0, 'total_samples': 1}
        
        best = max(detections, key=lambda x: x['confidence'])
        print(f"[WeaponDetector] Vũ khí được chọn: {best['weapon']} (confidence: {best['confidence']:.2%})", flush=True)
        sys.stdout.flush()
        
        return {
            'detected_weapon': best['weapon'],
            'confidence': best['confidence'],
            'detection_count': 1,
            'total_samples': 1
        }
    
    @classmethod
    def detect_from_image(cls, image_path: str) -> dict:
        if image_path.lower().endswith(".jfif"):
            jpg_path = str(pathlib.Path(image_path).with_suffix(".jpg"))
            img = Image.open(image_path)
            img.save(jpg_path, "JPEG")
            image_path = jpg_path
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        model = cls._load_model()
        results = model(image_path, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id].lower()
                    
                    weapon_name = cls._map_weapon_name(cls_name)
                    if weapon_name:
                        detections.append({
                            'weapon': weapon_name,
                            'confidence': conf
                        })
        
        if not detections:
            return {'detected_weapon': None, 'confidence': 0.0, 'detection_count': 0}
        
        best = max(detections, key=lambda x: x['confidence'])
        return {
            'detected_weapon': best['weapon'],
            'confidence': best['confidence'],
            'detection_count': len(detections)
        }
    
    @classmethod
    def _map_weapon_name(cls, detected_name: str) -> str:
        detected_name = detected_name.lower().strip()
        return cls.WEAPON_MAPPING.get(detected_name)
