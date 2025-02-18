import cv2
import numpy as np
from utils.video_utils import play_video
from utils.detection import load_yolo_model, detect_objects

# YouTube 재생 목록
PLAYLIST = [
    "https://www.youtube.com/live/dEgCz22Ayt0",
    "https://www.youtube.com/live/wnU-iYLZ7YU",
    "https://www.youtube.com/live/MWhYlaQr_DA",
]

# 영상 캡처 설정
WIDTH, HEIGHT = 640, 360
FRAME_SIZE = WIDTH * HEIGHT * 3  # RGB 3채널 (bgr24 포맷)

def main():
    """메인 실행 함수"""
    # YOLO 모델 로드 (사전 학습된 COCO 데이터셋 기반 YOLOv8)
    # ----------------------------------------------------------------------------------------
    # 모델	크기	속도 (FPS)	정확도 (mAP@50)	매개변수(Params)	FLOPs
    # ----------------------------------------------------------------------------------------
    # YOLOv8n (Nano)	가장 작음	🏎️ 매우 빠름	  🎯 낮음 (~37.3)	3.2M	8.7 BFLOPs
    # YOLOv8s (Small)	작음	    🚀 빠름	        🎯 중간 (~44.9)	11.2M	28.6 BFLOPs
    # YOLOv8m (Medium)	중간	    ⚡ 중간 속도	   🎯 높음 (~50.2)	25.9M	78.9 BFLOPs
    # YOLOv8l (Large)	큼	        🐢 느림	       🎯 매우 높음 (~52.3)	43.7M	165.2 BFLOPs
    # YOLOv8x (X-Large)	가장 큼	🐌 매우 느림	    🎯 최상 (~53.9)	68.2M	257.8 BFLOPs
    # ----------------------------------------------------------------------------------------
    model = load_yolo_model("yolov8n.pt") # 가장 빠른 모델 (YOLOv8n)
    # model = load_yolo_model("yolov8s.pt") # 작은 모델 (YOLOv8s)
    # model = load_yolo_model("yolov8m.pt") # 중간 크기 (YOLOv8m)
    # model = load_yolo_model("yolov8l.pt") # 큰 모델 (YOLOv8l)
    # model = load_yolo_model("yolov8x.pt") # 초대형 모델 (YOLOv8x)

    # 첫 번째 영상 실행
    current_video_index = 0
    process = play_video(PLAYLIST, current_video_index)

    while True:
        raw_frame = process.stdout.read(FRAME_SIZE)
        if not raw_frame:
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
        if frame is None or frame.size == 0:
            continue

        # 객체 감지 수행
        objects = detect_objects(model, frame)

        # 검출된 객체 화면에 표시
        for obj, bbox in objects:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YouTube Stream (YOLO Detection)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # 종료
            break
        elif key == ord("n"):  # 다음 영상
            print("🔄 다음 영상으로 이동...")
            process.kill()
            current_video_index = (current_video_index + 1) % len(PLAYLIST)
            process = play_video(PLAYLIST, current_video_index)

    process.kill()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
