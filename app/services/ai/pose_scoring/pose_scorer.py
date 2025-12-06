import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from scipy.signal import savgol_filter, resample
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
import gc
from flask import current_app


class PoseScorer:
    
    _pose_model = None
    _model_name = "yolov8n-pose.pt"
    
    SMOOTH_WINDOW = 11
    SMOOTH_POLY = 3
    
    @classmethod
    def _load_pose_model(cls):
        import sys
        if cls._pose_model is None:
            print(f"[PoseScorer] Loading YOLO pose model: {cls._model_name}...", flush=True)
            sys.stdout.flush()
            cls._pose_model = YOLO(cls._model_name)
            print(f"[PoseScorer] YOLO pose model loaded successfully", flush=True)
            sys.stdout.flush()
        return cls._pose_model
    
    @classmethod
    def normalize_pose(cls, kpts):
        k = np.array(kpts).reshape(-1, 3)
        xy = k[:, :2]
        center = np.mean(xy, axis=0)
        xy = xy - center
        
        scale = np.linalg.norm(xy)
        if scale < 1e-6:
            scale = 1.0
        
        xy = xy / scale
        return xy.flatten().astype(np.float32)
    
    @classmethod
    def smooth_savgol(cls, seq):
        if len(seq) < cls.SMOOTH_WINDOW:
            return seq
        
        out = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            out[:, i] = savgol_filter(seq[:, i], cls.SMOOTH_WINDOW, cls.SMOOTH_POLY)
        return out
    
    @classmethod
    def smooth_ema(cls, data, alpha=0.25):
        out = data.copy()
        for i in range(1, len(data)):
            out[i] = alpha * out[i] + (1 - alpha) * out[i - 1]
        return out
    
    @classmethod
    def extract_template_from_video(cls, video_path: str) -> np.ndarray:
        import sys
        print(f"[PoseScorer] extract_template_from_video: {video_path}", flush=True)
        sys.stdout.flush()
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"[PoseScorer] Loading pose model...", flush=True)
        sys.stdout.flush()
        model = cls._load_pose_model()
        print(f"[PoseScorer] Pose model loaded", flush=True)
        sys.stdout.flush()
        
        print(f"[PoseScorer] Opening video: {video_path}", flush=True)
        sys.stdout.flush()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[PoseScorer] Video info: {total_frames} frames, {fps:.2f} fps", flush=True)
        sys.stdout.flush()
        
        frames = []
        frame_num = 0
        
        print(f"[PoseScorer] Starting frame processing...", flush=True)
        sys.stdout.flush()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"[PoseScorer] Processing frame {frame_num}/{total_frames}...", flush=True)
                sys.stdout.flush()
            
            res = model(frame, verbose=False)
            if len(res[0].keypoints) == 0:
                continue
            
            k = res[0].keypoints[0].data.cpu().numpy().flatten()
            
            if np.sum(k == 0) > 10:
                continue
            
            frames.append(cls.normalize_pose(k))
            
            del k, res
            gc.collect()
        
        cap.release()
        
        print(f"[PoseScorer] Processed {frame_num} frames, extracted {len(frames)} valid pose frames", flush=True)
        sys.stdout.flush()
        
        if len(frames) == 0:
            raise ValueError("No valid pose frames found in video")
        
        print(f"[PoseScorer] Converting to numpy array...", flush=True)
        sys.stdout.flush()
        frames = np.array(frames, dtype=np.float32)
        
        print(f"[PoseScorer] Applying smoothing...", flush=True)
        sys.stdout.flush()
        frames = cls.smooth_savgol(frames)
        frames = cls.smooth_ema(frames)
        
        print(f"[PoseScorer] Template extraction complete. Shape: {frames.shape}", flush=True)
        sys.stdout.flush()
        
        return frames
    
    @classmethod
    def load_teacher_template(cls, template_path: str) -> np.ndarray:
        import sys
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        print(f"[PoseScorer] Loading template from: {template_path}", flush=True)
        sys.stdout.flush()
        
        teacher_raw = np.load(template_path)
        print(f"[PoseScorer] Template raw shape: {teacher_raw.shape}", flush=True)
        sys.stdout.flush()
        
        if teacher_raw.shape[1] == 51:
            print(f"[PoseScorer] Template has 51 dims, applying normalization...", flush=True)
            teacher = np.apply_along_axis(cls.normalize_pose, 1, teacher_raw)
        else:
            print(f"[PoseScorer] Template already normalized (34 dims), using as-is", flush=True)
            teacher = teacher_raw
        
        print(f"[PoseScorer] Template final shape: {teacher.shape}", flush=True)
        sys.stdout.flush()
        
        return teacher
    
    @classmethod
    def align_length(cls, teacher_kpts: np.ndarray, student_kpts: np.ndarray) -> np.ndarray:
        aligned = resample(teacher_kpts, len(student_kpts), axis=0)
        gc.collect()
        return aligned
    
    @classmethod
    def compute_jitter_mse(cls, seq: np.ndarray) -> float:
        diff = np.diff(seq, axis=0)
        jitter = np.mean(diff ** 2)
        return jitter
    
    @classmethod
    def compare_templates(cls, teacher: np.ndarray, student: np.ndarray) -> tuple:
        L = min(len(teacher), len(student))
        T = teacher[:L]
        S = student[:L]
        
        cosine_sim = float(np.mean([1 - cosine(t, s) for t, s in zip(T, S)]))
        
        dtw_shape, _ = fastdtw(T, S, dist=euclidean, radius=10)
        
        T_d = np.diff(T, axis=0)
        S_d = np.diff(S, axis=0)
        dtw_tempo, _ = fastdtw(T_d, S_d, dist=euclidean, radius=10)
        
        dtw_dist = float(dtw_shape * 0.7 + dtw_tempo * 0.3)
        
        jitter_mse = cls.compute_jitter_mse(S)
        
        gc.collect()
        return cosine_sim, dtw_dist, jitter_mse
    
    @classmethod
    def evaluate(cls, cosine_sim: float, dtw_dist: float, jitter: float) -> dict:
        accuracy_score = max(0, min(50, cosine_sim * 50))
        
        speed_score = 30 * math.exp(-dtw_dist / 1000)
        speed_score = max(0, min(30, speed_score))
        
        if jitter < 0.002:
            stability_score = 20
        elif jitter < 0.005:
            stability_score = 15
        elif jitter < 0.01:
            stability_score = 10
        else:
            stability_score = 5
        
        total_score = accuracy_score + speed_score + stability_score
        
        feedback = []
        if cosine_sim < 0.6:
            feedback.append("Tư thế chưa khớp nhiều với giáo viên.")
        if dtw_dist >= 600:
            feedback.append("Nhịp chuyển động chưa khớp bài mẫu – cần giữ đúng tốc độ ở từng pha ra đòn.")
        if jitter >= 0.005:
            feedback.append("Keypoint dao động mạnh – cần giữ tư thế vững hơn.")
        elif jitter >= 0.001:
            feedback.append("Tư thế hơi rung, cố gắng giữ ổn định hơn một chút.")
        
        if not feedback:
            feedback = ["Tốt! Tư thế và nhịp rất gần với giáo viên."]
        
        return {
            'total_score': round(total_score, 2),
            'accuracy_score': round(accuracy_score, 2),
            'speed_score': round(speed_score, 2),
            'stability_score': round(stability_score, 2),
            'feedback': feedback,
            'metrics': {
                'cosine_similarity': round(cosine_sim, 4),
                'dtw_distance': round(dtw_dist, 2),
                'jitter_mse': round(jitter, 6)
            }
        }
    
    @classmethod
    def score_video(cls, student_video_path: str, teacher_template_path: str) -> dict:
        import sys
        print(f"[PoseScorer] Đang trích xuất pose từ video học viên...", flush=True)
        sys.stdout.flush()
        student_template = cls.extract_template_from_video(student_video_path)
        print(f"[PoseScorer] Đã trích xuất {len(student_template)} frames từ video học viên", flush=True)
        
        print(f"[PoseScorer] Đang tải teacher template...", flush=True)
        teacher_template = cls.load_teacher_template(teacher_template_path)
        print(f"[PoseScorer] Teacher template có {len(teacher_template)} frames", flush=True)
        
        print(f"[PoseScorer] Đang căn chỉnh độ dài...", flush=True)
        teacher_aligned = cls.align_length(teacher_template, student_template)
        print(f"[PoseScorer] Đã căn chỉnh: teacher {len(teacher_aligned)} frames, student {len(student_template)} frames", flush=True)
        
        print(f"[PoseScorer] Đang so sánh templates...", flush=True)
        cosine_sim, dtw_dist, jitter = cls.compare_templates(teacher_aligned, student_template)
        print(f"[PoseScorer] Kết quả so sánh:", flush=True)
        print(f"  - Cosine Similarity: {cosine_sim:.4f}", flush=True)
        print(f"  - DTW Distance: {dtw_dist:.2f}", flush=True)
        print(f"  - Jitter MSE: {jitter:.6f}", flush=True)
        
        print(f"[PoseScorer] Đang tính điểm...", flush=True)
        sys.stdout.flush()
        result = cls.evaluate(cosine_sim, dtw_dist, jitter)
        
        return result
