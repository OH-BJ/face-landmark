# Face Landmark (MediaPipe Webcam Demo)

[MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)로 웹캠 영상에서 **얼굴 랜드마크 478개**와 **표정 점수(blendshape) 52개**를 실시간으로 추출하고 시각화하는 파이썬 예제입니다.

## 기능

- 웹캠 실시간 처리 (`LIVE_STREAM` 모드, 비동기 추론)
- 최대 2명까지 동시 인식
- 얼굴 메쉬(tesselation) 및 부위별 윤곽선 표시
  - 얼굴 외곽, 입술, 왼쪽/오른쪽 눈·눈썹, 홍채
- 첫 번째 얼굴의 blendshape 상위 10개를 막대그래프로 표시
- FPS와 인식된 얼굴 수 표시
- 거울 모드(좌우 반전) 화면

## 폴더 구조

```
face-landmark/
├── webcam_face_landmarker.py   # 실행 스크립트
├── face_landmarker.task        # MediaPipe 모델 (float16, latest)
├── requirements.txt
└── README.md
```

## 요구 사항

- Python 3.9 ~ 3.12 (3.11.9에서 확인)
- 웹캠

## 설치

```bash
git clone https://github.com/OH-BJ/face-landmark.git
cd face-landmark
pip install -r requirements.txt
```

모델 파일(`face_landmarker.task`)이 저장소에 포함되어 있습니다. 직접 받으려면 아래 링크를 사용하세요.

```bash
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## 실행

```bash
python webcam_face_landmarker.py
```

### 단축키

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `m` | 얼굴 메쉬 표시 켜기/끄기 |
| `b` | blendshape 패널 켜기/끄기 |

## 설정

`webcam_face_landmarker.py` 상단의 상수로 조절합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CAMERA_INDEX` | `0` | 웹캠 번호. 카메라가 안 열리면 `1`로 변경 |
| `NUM_FACES` | `2` | 동시에 인식할 최대 얼굴 수 |

인식 민감도는 `FaceLandmarkerOptions`의 `min_face_detection_confidence`, `min_face_presence_confidence`, `min_tracking_confidence`(기본 0.5)로 조절할 수 있습니다.

## 동작 방식

1. OpenCV로 웹캠 프레임을 읽고 좌우 반전(거울 모드)
2. BGR → RGB 변환 후 `mp.Image`로 감싸 `detect_async()`에 전달
   - `LIVE_STREAM` 모드는 타임스탬프가 단조 증가해야 하므로 `time.monotonic()` 기반 ms 값을 사용
3. 결과는 `result_callback`으로 비동기 수신 → 가장 최근 결과를 프레임 위에 그림
4. 랜드마크 연결 정보는 `FaceLandmarksConnections`를 사용하고, 그리기는 OpenCV로 직접 처리
   (최신 mediapipe에는 `mp.solutions` 드로잉 유틸이 없음)

## 참고: 좌우 반전

화면을 거울처럼 보이게 하려고 `cv2.flip(frame, 1)`을 적용한 뒤 모델에 넣습니다. 인식 자체에는 반전이 필요 없지만, 반전된 이미지를 넣으면 `LEFT_EYE`/`RIGHT_EYE`, `eyeBlinkLeft`/`eyeBlinkRight` 같은 **좌우 결과가 실제 얼굴과 반대로 나올 수 있습니다.**
좌우 구분이 중요하다면 해당 줄을 지우고 원본 프레임을 사용하세요.

## 출력 데이터

`FaceLandmarkerResult`에는 다음이 들어 있습니다.

- `face_landmarks`: 얼굴별 478개 랜드마크 (`x`, `y`는 0~1 정규화 좌표, `z`는 상대 깊이)
- `face_blendshapes`: 얼굴별 52개 표정 계수 (`category_name`, `score`)
- `facial_transformation_matrixes`: 얼굴 변환 행렬 (이 예제에서는 비활성화)

## 라이선스 / 출처

- 모델: [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) Face Landmarker (Apache License 2.0)
- 가이드: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
