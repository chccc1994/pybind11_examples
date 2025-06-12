# search.py

import os
import cv2
import glob
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from numpy import random
from extracter_fgx import FeatureExtractor
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

class MultiPersonSearcher:

    _instance = None  # 单例缓存

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def __init__(self,
                 weights='yolo11l.pt',
                 query_folder='query_person',
                 reid_config='model/ft_ResNet50/opts.yaml',
                 device='0' if torch.cuda.is_available() else 'cpu',
                 match_threshold=1.0,
                 conf_thres=0.25,
                 show_pca=False,
                 rotate=False):

        self.device = torch.device(f'cuda:{device}' if device != 'cpu' else 'cpu')
        self.model = YOLO(weights)
        self.model.to(self.device)

        self.conf_thres = conf_thres
        self.show_pca = show_pca
        self.rotate = rotate

        self.extractor = FeatureExtractor(reid_config, device=self.device)

        self.match_threshold = match_threshold

        self.names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.target_features = {}
        self.load_query_images(query_folder)

    def load_query_images(self, root_folder):
        print(f"Loading query images from {root_folder}...")
        person_dirs = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

        for person_id_str in person_dirs:
            person_path = os.path.join(root_folder, person_id_str)
            image_paths = sorted(glob.glob(os.path.join(person_path, "*.jpg")))

            if not image_paths:
                print(f"Warning: No images found in {person_path}")
                continue

            try:
                pid = (person_id_str)
                feats = []

                for img_path in image_paths:
                    img = Image.open(img_path).convert('RGB')
                    feat = self.extractor.extract_feature(img)
                    feats.append(feat)
                    print(f"  Loaded image for person ID {pid}: {img_path}")

                self.target_features[pid] = feats  # 每人是一组特征向量列表
                print(f"Stored {len(feats)} features for person ID {pid}")

            except Exception as e:
                print(f"Error processing {person_path}: {e}")

        print(f"Loaded {len(self.target_features)} query persons.")
        self.fit_reference_pca()

    def fit_reference_pca(self):
        print("Fitting PCA on reference features...")
        ref_feats = []
        self.pca_labels = []

        for pid, feats in self.target_features.items():
            for j, f in enumerate(feats):
                ref_feats.append(f)
                self.pca_labels.append(f"{pid}_{j}")

        self.pca_mat = np.stack(ref_feats)
        self.pca = PCA(n_components=2)
        self.pca_result = self.pca.fit_transform(self.pca_mat)
        print(f"PCA fitted on {len(self.pca_labels)} features.")
        
    def visualize_feature_space_embed(self, query_feats):
        fig, ax = plt.subplots(dpi=500)
        x_all, y_all = self.pca_result[:, 0], self.pca_result[:, 1]

        # 画蓝色 target 点
        ax.scatter(x_all, y_all, c='blue', label='targets', s=10)
        for label, x, y in zip(self.pca_labels, x_all, y_all):
            ax.text(x, y, label, fontsize=5, alpha=0.4)

        # 即使 query_feats 为空，也保持图像绘制流程
        if query_feats:
            query_reduced = self.pca.transform(query_feats)
            for xq, yq in query_reduced:
                ax.scatter(xq, yq, c='red', s=40, label='query')

        ax.set_title("PCA Feature Space", fontsize=10)
        ax.axis('off')
        fig.tight_layout()

        # 渲染为 numpy 图像
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8).copy()
        plt.close(fig)

        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    def detect_persons(self, frame):
        """返回每个人的框、置信度和类别id"""
        results = self.model.predict(source=frame, verbose=False, rect=True, conf=self.conf_thres)[0]
        boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf.cpu().numpy().flatten()[0])
                if self.names[cls_id] != 'person':
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls_id": cls_id
                })
        return boxes

    def extract_person_features(self, frame, boxes):
        """输入图像和boxes，返回裁剪图像和特征向量"""
        crops = []
        feats = []
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                feats.append(None)
                crops.append(None)
                continue
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            feat = self.extractor.extract_feature(pil_crop)
            feats.append(feat)
            crops.append(crop)
        return crops, feats

    def match_ids(self, feats, matched_ids=None):
        """输入特征列表，输出每个人匹配到的ID和距离"""
        if matched_ids is None:
            matched_ids = set()
        match_results = []
        for feat in feats:
            if feat is None:
                match_results.append((None, None))  # 没有特征
                continue
            best_id, min_dist = None, float('inf')
            for pid, ref_feats in self.target_features.items():
                if pid in matched_ids:
                    continue
                dists = [np.linalg.norm(feat - ref_feat) for ref_feat in ref_feats]
                min_pid_dist = min(dists)
                if min_pid_dist < min_dist and min_pid_dist < self.match_threshold:
                    min_dist = min_pid_dist
                    best_id = pid
            match_results.append((best_id, min_dist))
        return match_results

    def process_frame(self, frame, frame_id=None):
        """主流程：先检测，再匹配，最后画图"""
        boxes = self.detect_persons(frame)
        matched_ids = set()
        crops, feats = self.extract_person_features(frame, boxes)
        match_results = self.match_ids(feats, matched_ids=matched_ids)

        annotator = Annotator(frame)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['bbox']
            conf = box['conf']
            best_id, min_dist = match_results[i]
            if best_id is not None:
                matched_ids.add(best_id)
                label = f"ID:{best_id}"
                color = tuple(self.colors[hash(best_id) % len(self.colors)])
            else:
                label = f"unmatched conf:{conf:.2f}"
                color = (128, 128, 128)
            annotator.box_label([x1, y1, x2, y2], label, color=color)

        if self.show_pca:
            query_feats = [f for f in feats if f is not None]
            result_pca = self.visualize_feature_space_embed(query_feats)
        else:
            result_pca = None
        
        return annotator.result(), result_pca

    def search_video(self, source, save_path='output', view=True, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"Cannot open video: {source}"

        video_name = os.path.basename(source)
        save_file = os.path.join(save_path, video_name)
        save_pca_file = os.path.join(save_path, "pca_" + video_name)

        os.makedirs(save_path, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        num_frames_to_process = end_frame - start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame

        if self.rotate:
            writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (h, w))
        else:
            writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        sample_img = self.visualize_feature_space_embed([])
        h_pca, w_pca = sample_img.shape[:2]
        pca_writer = cv2.VideoWriter(
            save_pca_file,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w_pca, h_pca)
        )

        print(f"Processing video from frame {start_frame} to {end_frame}...")

        with tqdm(total=num_frames_to_process, desc="Frames Processed") as pbar:
            while frame_id < end_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[Warning] Failed to read frame {frame_id}")
                    break
                
                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                result_frame, result_pca = self.process_frame(frame, frame_id=frame_id)
                writer.write(result_frame)

                if self.show_pca:
                    #pca_writer.write(result_pca)
                    print(0)

                if view:
                    cv2.imshow('Result', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_id += 1
                pbar.update(1)

        cap.release()
        writer.release()
        pca_writer.release()
        cv2.destroyAllWindows()
        print(f"Finished. Saved to: {save_path}")

    def match_id_from_crop(self, img):
        """
        输入：已裁剪的人像图片（PIL.Image 或 numpy BGR）
        输出：{'id': best_id, 'min_dist': 距离}，匹配不上时id为None
        """
        # 支持PIL和opencv/numpy格式
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        feat = self.extractor.extract_feature(pil_img)
        best_id, min_dist = None, float('inf')
        for pid, ref_feats in self.target_features.items():
            dists = [np.linalg.norm(feat - ref_feat) for ref_feat in ref_feats]
            min_pid_dist = min(dists)
            if min_pid_dist < min_dist and min_pid_dist < self.match_threshold:
                min_dist = min_pid_dist
                best_id = pid

        return best_id

def main(image_path=None):

    searcher = MultiPersonSearcher.get_instance(
        weights='yolo11n.pt',
        query_folder='query_person/yanggao',
        reid_config='model/ft_ResNet50/opts.yaml',
        match_threshold=1.0,
        device='0' if torch.cuda.is_available() else 'cpu',
        conf_thres=0.5,
        show_pca=False,
        rotate=False
    )

    crop_img = cv2.imread(image_path)
    result = searcher.match_id_from_crop(crop_img)
    print("识别结果:", result)
    

# if __name__ == '__main__':
#     for _ in range(3):
#         main("./123.jpg" )  # 替换为实际的裁剪图像路径
