# Vision Bench

한 장의 이미지에서 OpenCV 처리 순서와 파라미터를 조절하고, JSON 설정과 독립 실행 Python 코드를 추출하는 로컬 웹 도구입니다. 카메라, DB, 외부 서비스가 필요하지 않습니다.

## 실행

```powershell
python -m pip install -r requirements.txt
python -m uvicorn workbench.server:app --host 127.0.0.1 --port 8765
```

브라우저에서 http://127.0.0.1:8765 를 엽니다. `start.ps1`로도 실행할 수 있습니다.

## 사용

1. 이미지를 업로드하거나 포함된 TIFF 샘플을 선택합니다. 현재 PNG/JPG/TIFF의 8-bit 영상을 지원합니다. 컬러는 OpenCV의 BGR→GRAY 변환으로 Mono8 처리합니다. TIFF는 첫 페이지를 읽습니다. Mono16은 자동으로 축소하지 않고 거부합니다.
2. **ROI 그리기**를 눌러 드래그하거나 ROI 단계의 X/Y/너비/높이를 입력합니다. X/Y는 원본 픽셀 좌표입니다. 너비·높이 0은 현재 입력의 끝까지를 뜻합니다.
3. **처리 단계 추가**로 필터, 밝기 보정, Threshold, Edge, Morphology, 객체 측정, 판정을 조합합니다. 단계 클릭으로 중간 결과를 봅니다. 순서 변경, 활성화, 복제, 삭제를 지원합니다.
4. 오른쪽에서 값을 조절합니다. 잘못된 타입 연결이나 범위는 오류로 표시합니다. 자동 실행을 끄고 한 번에 수정한 다음 실행할 수도 있습니다.
5. **JSON 저장**과 **JSON 열기**로 전체 설정을 왕복합니다. 브라우저에는 최근 설정만 자동 보관하며, 이미지는 보관하지 않습니다.
6. **세팅 / 코드 추출**에서 현재 Recipe JSON, 설정표 CSV, 독립 실행 Python 파일을 다운로드합니다. Python 파일은 웹 서버 없이 OpenCV와 NumPy만으로 실행됩니다.

```powershell
python vision_pipeline.py test_image/0.tiff --output output
```

결과는 `result.png`와 `result.json`에 기록됩니다. **결과 JSON** 버튼은 마지막 실행의 실제 설정, 버전, 단계별 자동 계산값, 객체 측정, 판정을 저장합니다. 현재 편집값과 실행 결과가 다르면 재실행해야 결과를 저장할 수 있습니다.

## 처리 의미

- 모든 필터, ROI, 측정은 원본 해상도에서 실행됩니다. Preview 전송만 긴 변 1200px 이하로 축소합니다. `확대`는 원본과 같은 표시 크기이며, 큰 이미지의 Preview 자체에는 축소가 적용됩니다.
- Gaussian의 border는 reflect101, sigma=0은 OpenCV 자동 계산입니다. Morphology는 OpenCV 기본 경계 정책을 사용합니다. 재현을 위해 실행 결과에 OpenCV/NumPy 버전을 기록합니다.
- Sobel/Laplacian은 미분 절댓값을 0–255로 정규화합니다. Threshold에 연결할 수 있습니다. Canny Edge는 영역 Mask와 구분하므로 Contour/Blob 면적 측정에 직접 연결하지 않습니다.
- Contour는 외곽 윤곽을 사용합니다. Area는 윤곽 면적이며 구멍은 빼지 않습니다. Blob Area는 8방향 연결 영역 픽셀 개수입니다. 원형도·Solidity는 두 방식 모두 외곽 윤곽으로 계산합니다.
- Width/Height는 축 정렬 Bounding Box, Aspect는 긴 변/짧은 변입니다. 회전 박스와 OpenCV 각도는 결과 JSON에도 포함합니다. 좌표와 중심점은 ROI 이전 원본 좌표로 변환합니다.
- Rule은 객체 수, 면적, 최소 원형도, 최대 장단변 비율을 AND로 적용합니다. 객체가 없으면 NG입니다. 기존 `opencv.py`의 Bridge/과납/납 부족 분류는 이 일반 Rule과 별개입니다.
- 파일 40MB, 24메가픽셀, 24단계, 출력 객체 1000개 제한이 있습니다. 처리 중인 한 요청이 끝나면 가장 최근 설정을 실행하고 오래된 응답은 화면에 적용하지 않습니다.
- JSON 스키마의 고정된 Operation 의미가 처리 계약입니다. 알려지지 않은 파라미터나 잘못된 연결은 거부합니다. 이 도구는 단일 사용자 로컬 사용을 전제로 합니다.

## 기능 선택 · 설명 · 수동 조절

기존 3개 패널, 이미지 Preview, 측정 표, Recipe 형식을 유지하면서 기능을 **19개 → 53개**로 확장했습니다. 카메라·학습·3D 등 OpenCV 전체 SDK를 임의 실행하는 방식이 아니라 현재 Mono8 단일 이미지 검사에 연결 가능한 기능을 제공합니다. 히스토그램·통계 그래프와 별도 A/B 화면은 추가하지 않습니다.

1. **처리 단계 추가**에서 이름·용도·파라미터를 검색하거나 분류를 선택합니다.
2. 기능을 선택하면 설명, 입력/출력 타입, 파라미터 범위·간격·기본값을 먼저 읽을 수 있습니다. 입력이 맞지 않는 기능도 설명은 볼 수 있고, 필요한 연결을 안내합니다.
3. **선택 기능 추가**로 단계에 넣습니다. 오른쪽 Parameters에서 모든 수치 값을 직접 입력하고 선택값을 바꿀 수 있습니다. 제공되는 슬라이더도 함께 사용할 수 있습니다.
4. **변경 시 자동 실행**을 끄면 MANUAL 모드입니다. 여러 값을 수정한 뒤 **▶ 실행**으로 적용합니다. 수치의 범위와 홀수 커널 간격을 지켜주세요. Otsu/Triangle처럼 알고리즘이 계산하는 임계값 자체는 자동이며 수동 임계값은 Global Threshold에서 조절합니다.

| 분류 | 선택 가능한 기능 |
| --- | --- |
| 입력 | ROI |
| 필터 | Gaussian, Median, Bilateral, Mean/Box, Unsharp, Non-local Means, 직접 3×3 Filter2D 커널, Gabor |
| 밝기 | Equalize, CLAHE, Gamma, Normalize, 밝기/대비, Top-hat, Black-hat, 배경 보정 |
| 분할 | Global, Otsu, Adaptive, Range, Triangle, Truncate/To-zero, 닫힌 Edge → Mask |
| 에지 | Canny, Sobel, Laplacian, Morphological Gradient, Scharr, Prewitt, Roberts, DoG, LoG, Harris 응답 |
| 형태학 | Mask/Edge Morphology, Grayscale Morphology |
| 마스크 정리 | 구멍 채우기, 작은 영역 제거, Distance Transform, 경계 영역 제거, 면적 범위 필터, Convex Hull, Watershed |
| 측정 | Contour, Connected Components, Hough 선분, Hough 원, Shi–Tomasi 코너, ORB, AKAZE, ROI Template Matching |
| 변환 · 판정 | 반전, 기존 OK/NG Rule |

검출 기능도 기존 경계 상자 Preview와 측정 표를 사용합니다. 선분·코너는 영역 면적 0, 원은 πr², 특징점은 지원 원 면적이며 Contour/Blob 면적과 의미가 다릅니다. 선분의 Perimeter는 닫힌 2점 윤곽의 둘레이므로 길이의 두 배이며 실제 길이와 원 반지름, 매칭 점수 등 추가 값은 결과 JSON에 포함됩니다. 위치는 ROI를 포함해 원본 좌표로 기록합니다. Template Matching의 템플릿 X/Y 입력만 현재 처리 영상 좌표 기준입니다.

Hough Circles와 AKAZE는 4메가픽셀 이하 ROI에서 실행합니다. 큰 NLM 검색 창이나 Gabor 커널은 시간이 걸리므로 작은 ROI와 MANUAL 모드로 먼저 비교하세요. Template Matching은 같은 처리 영상 안의 ROI를 참조하며 회전·크기 변화나 다른 참조 이미지 업로드를 지원하지 않습니다. Watershed는 씨앗이 분리된 경우에만 붙은 객체를 나누므로 중심 거리 비율을 조절하세요.

## 구조와 검증

- `workbench/engine.py`: UI와 독립된 OpenCV 연산·검증·측정
- `workbench/server.py`: stateless 로컬 API 및 코드/CSV 추출
- `workbench/static/`: 브라우저 UI
- `opencv.py`: 기존 실험 코드 (변경하지 않음)

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```
