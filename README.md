# 📺 AI 기반 미디어 자동 검수 시스템  

**IPTV 및 OTT 서비스의 채널 전환, 영상 품질, 전환 속도를 자동 분석하는 AI 기반 검수 시스템**  

## 📌 프로젝트 개요  
본 프로젝트는 **IPTV, OTT 등의 미디어 서비스의 검수 작업을 자동화**하기 위한 솔루션입니다.  
리모컨 신호로 채널을 전환하고, HDMI 캡처 장치로 영상을 받아 AI 모델이 자동으로 채널 전환 검출, 화면 품질 분석, 전환 지연 측정을 수행합니다.  

## ✨ 주요 기능  
✅ **채널 전환 감지**: 장면이 바뀌는 시점을 분석하거나 로고·워터마크를 검출하여 채널 전환 여부 판단  
✅ **영상 품질 분석**: 화면 깨짐, 지연, 색상 이상 등의 품질 문제 감지  
✅ **전환 지연(Delay) 측정**: 리모컨 입력 시점과 화면 변환 시점 차이를 자동 측정  
✅ **자동 리포팅**: 감지된 문제에 대한 로그 및 알림 자동 생성  

## 🛠️ 기술 스택  
| 기능                  | 사용 기술 |  
|----------------------|-----------------|  
| 채널 전환 검출       | PySceneDetect, YOLOv5/YOLOv8, Detectron2 |  
| 영상 품질 분석       | SSIM, PSNR, OpenMMLab(VQA), Detectron2 |  
| OCR(문자 인식)       | Tesseract, EasyOCR |  
| 영상 캡처 및 처리   | OpenCV, FFMPEG |  

## 🚀 설치 및 실행 방법  

### 1️⃣ **필수 패키지 설치**  
```bash
# pip install -r requirements.txt
# torch gpu관련 환경이 개인 마다 다를수 있어, requirements는 첨부하지 않았습니다.
# python main.py를 수행해보고, 필요한 라이브러리 설치하면 될 듯 합니다.
```

### 2️⃣ **프로그램 실행**  
```bash
python main.py
```

## 📊 분석 프로세스  
1️⃣ **리모컨/프로그램 제어**: 특정 채널로 전환 명령 발송 및 신호 타임스탬프 기록  
2️⃣ **영상 캡처**: HDMI 입력을 캡처 카드로 수집 및 실시간 처리  
3️⃣ **AI 분석 적용**: 장면 전환 검출(PySceneDetect), 영상 품질 평가(SSIM, PSNR), OCR 분석 수행  
4️⃣ **결과 저장 및 보고**: 채널 전환 성공 여부, 전환 시간, 품질 문제 리포트 생성  

## 📢 기대 효과  
✔️ **수작업 없이 자동화된 검수 가능**  
✔️ **정확한 채널 전환 속도 및 품질 검출**  
✔️ **영상 품질 문제를 실시간으로 탐지하여 대응 가능**  

---
