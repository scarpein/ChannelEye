import os
from ultralytics import YOLO

# 모델 저장 경로 설정
MODEL_DIR = os.path.join(os.getcwd(), "models")  # 현재 디렉토리 내 models 폴더 사용
os.makedirs(MODEL_DIR, exist_ok=True)  # 디렉토리가 없으면 생성

def load_yolo_model(model_name="yolov8n.pt"):
    """YOLO 모델을 로드하고 다운로드 경로를 models 폴더로 지정"""
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # 모델이 없으면 다운로드
    if not os.path.exists(model_path):
        print(f"🔄 YOLO 모델 다운로드 중: {model_name}")
        model = YOLO(model_name)  # YOLO 모델 다운로드
        # model.export(format="torchscript")  # 모델 저장 (선택적)
        os.rename(model_name, model_path)  # 다운로드된 모델을 models 폴더로 이동
    else:
        print(f"✅ 기존 모델 로드: {model_path}")

    return YOLO(model_path, verbose=False)

def detect_objects(model, frame):
    """YOLO를 사용하여 객체 감지"""
    results = model(frame, verbose=False)[0]
    
    objects = []
    for d in results.boxes:
        class_id = int(d.cls.item())
        bbox = tuple(map(int, d.xyxy[0].tolist()))
        objects.append((model.names[class_id], bbox))
    
    return objects
