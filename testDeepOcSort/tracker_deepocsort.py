import cv2
import numpy as np
from ultralytics import RTDETR
from boxmot import create_tracker
import time
from collections import defaultdict
import os

class DeepOCSortTrackingSystem:
    def __init__(self, model_path, reid_weights='osnet_x0_25_msmt17.pt', device='cuda:0'):
        # Ruta al modelo RTDETR
        self.model = RTDETR(model_path)
        self.tracker_type = 'deepocsort'
        
        print("Inicializando DeepOC-SORT a través de BoxMOT con RT-DETR...")
        self.tracker = create_tracker(
            tracker_type=self.tracker_type,
            tracker_config=None,
            reid_weights=reid_weights,
            device=device,
            half=False
        )
        
        # Ajustes de hiperparámetros de DeepOC-SORT para mejorar la robustez
        trk = self.tracker if not hasattr(self.tracker, 'tracker') else self.tracker.tracker
        if hasattr(trk, 'max_age'):
            trk.max_age = 150
        if hasattr(trk, 'det_thresh'):
            trk.det_thresh = 0.4
            
        self.trajectories = defaultdict(list)
        self.colors = {}

    def get_color(self, obj_id):
        if obj_id not in self.colors:
            np.random.seed(int(obj_id) * 123)
            self.colors[obj_id] = tuple([int(c) for c in np.random.randint(50, 255, 3)])
        return self.colors[obj_id]

    def reset(self):
        self.trajectories.clear()
        self.colors.clear()

    def process_source(self, source, save_path=None, show=False):
        cap = cv2.VideoCapture(source)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        out = None
        if save_path:
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
        frame_count = 0
        start_time = time.time()
        
        total_detections = 0
        unique_ids = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  -> Procesando frame {frame_count}...", end='\r')
            
            # Inferencias con RT-DETR
            results = self.model(frame, classes=[0], verbose=False) # 0 es ropa
            dets = results[0].boxes.data.cpu().numpy()
            
            total_detections += len(dets)
            
            # Actualizar tracker Deep-OC-SORT
            if len(dets) > 0:
                try:
                    tracks = self.tracker.update(dets, frame)
                except Exception as e:
                    # Ignore the error message 'niteray is not positive definite' from Matrix inversion. 
                    # This happens when detection geometry leads to unstable kalman matrices.
                    tracks = np.empty((0, 8))
            else:
                tracks = np.empty((0, 8))
                
            # Renderizado visual
            track_vis = frame.copy()
            
            for t in tracks:
                x1, y1, x2, y2, obj_id, conf, cls_id, ind = t
                obj_id = int(obj_id)
                unique_ids.add(obj_id)
                
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self.trajectories[obj_id].append((cx, cy))
                
                color = self.get_color(obj_id)
                
                # Dibujar trayectoria
                pts = self.trajectories[obj_id]
                cv2.polylines(track_vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=3)
                cv2.circle(track_vis, pts[-1], 6, color, -1)
                
                # Bounding box
                cv2.rectangle(track_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Etiqueta
                label = f'ID: {obj_id} (D-OC-SORT)'
                cv2.putText(track_vis, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(track_vis, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Agregar información de estado genérico
            cv2.putText(track_vis, f'Prendas Unicas Trackeadas: {len(unique_ids)}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(track_vis, f'Prendas Unicas Trackeadas: {len(unique_ids)}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if out:
                out.write(track_vis)
            
            if show:
                cv2.imshow(f"Tracker: DeepOC-SORT", cv2.resize(track_vis, (1024, 768)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        if out:
            out.release()
        
        cv2.destroyAllWindows()
        
        end_time = time.time()
        avg_fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        return {
            "total_detections": total_detections,
            "unique_ids": len(unique_ids),
            "avg_fps": avg_fps,
            "total_frames": frame_count
        }
