import cv2
import numpy as np


# =========================
# 설정
# =========================
IMAGE_PATH = "test_image/0.tiff"

DISPLAY_SCALE = 0.4

# 검사 대상 Contour 범위
MIN_AREA = 50
MAX_AREA = 20000

# 정상 납땜 Area 범위
OK_MIN_AREA = 80
OK_MAX_AREA = 400

RING_SIZE = 15

# 판별 기준
BRIDGE_ASPECT = 2.0
EXCESS_CIRCULARITY = 0.85
OK_RECTANGULARITY = 0.65

DARK_THRESHOLD = 100
MIN_DARK_RATIO = 0.55


# =========================
# 기본 함수
# =========================
def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale))
    )


def to_gray(img):
    if img.ndim == 2:
        return img

    return cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


def nothing(x):
    pass


# =========================
# 이미지 불러오기
# =========================
original = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_UNCHANGED
)

if original is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

gray = to_gray(original)


# =========================
# 전처리 Manual Control
# =========================
cv2.namedWindow(
    "Control",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Control",
    600,
    300
)


# Threshold
cv2.createTrackbar(
    "Threshold",
    "Control",
    180,
    255,
    nothing
)


# Gaussian Kernel
# 0 -> 1x1
# 1 -> 3x3
# 2 -> 5x5
cv2.createTrackbar(
    "Gaussian",
    "Control",
    2,
    10,
    nothing
)


# Opening
# 0 = OFF
# 1 = 3x3
# 2 = 5x5
cv2.createTrackbar(
    "Opening",
    "Control",
    0,
    10,
    nothing
)


# Closing
# 0 = OFF
# 1 = 3x3
# 2 = 5x5
cv2.createTrackbar(
    "Closing",
    "Control",
    0,
    10,
    nothing
)


print("전처리 값을 조절하세요.")
print("Threshold : 이진화")
print("Gaussian  : Blur")
print("Opening   : 작은 흰색 Noise 제거")
print("Closing   : 끊어진 흰색 영역 연결")
print()
print("ENTER : 다음 단계")
print("ESC   : 종료")


while True:

    # =========================
    # Trackbar 값
    # =========================
    threshold = cv2.getTrackbarPos(
        "Threshold",
        "Control"
    )

    gaussian_value = cv2.getTrackbarPos(
        "Gaussian",
        "Control"
    )

    opening_value = cv2.getTrackbarPos(
        "Opening",
        "Control"
    )

    closing_value = cv2.getTrackbarPos(
        "Closing",
        "Control"
    )


    # =========================
    # Gaussian Blur
    # =========================
    gaussian_size = (
        gaussian_value * 2 + 1
    )

    blur = cv2.GaussianBlur(
        gray,
        (gaussian_size, gaussian_size),
        0
    )


    # =========================
    # Threshold
    # =========================
    _, binary = cv2.threshold(
        blur,
        threshold,
        255,
        cv2.THRESH_BINARY
    )


    # =========================
    # Opening
    # 작은 흰색 Noise 제거
    # =========================
    if opening_value > 0:

        opening_size = (
            opening_value * 2 + 1
        )

        opening_kernel = np.ones(
            (opening_size, opening_size),
            np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            opening_kernel
        )


    # =========================
    # Closing
    # 끊어진 흰색 영역 연결
    # =========================
    if closing_value > 0:

        closing_size = (
            closing_value * 2 + 1
        )

        closing_kernel = np.ones(
            (closing_size, closing_size),
            np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            closing_kernel
        )


    # =========================
    # Preview
    # =========================
    cv2.imshow(
        "Binary Preview",
        resize(
            binary,
            DISPLAY_SCALE
        )
    )


    key = cv2.waitKey(1) & 0xFF

    if key == 13:      # ENTER
        break

    if key == 27:      # ESC
        cv2.destroyAllWindows()
        exit()


cv2.destroyAllWindows()


print()
print("========== 전처리 설정 ==========")
print("Threshold :", threshold)
print("Gaussian  :", gaussian_size)

if opening_value > 0:
    print("Opening   :", opening_value * 2 + 1)
else:
    print("Opening   : OFF")

if closing_value > 0:
    print("Closing   :", closing_value * 2 + 1)
else:
    print("Closing   : OFF")


# =========================
# Contour 검출
# =========================
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ============================================================
# 1단계 : Area 확인
# ============================================================
area_preview = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2BGR
)

print()
print("========== Contour Area ==========")


for i, cnt in enumerate(contours):

    area = cv2.contourArea(cnt)

    x, y, w, h = cv2.boundingRect(cnt)

    print(
        f"{i+1:02d} | "
        f"Area={area:.1f} | "
        f"W={w} H={h}"
    )

    cv2.rectangle(
        area_preview,
        (x, y),
        (x+w, y+h),
        (0, 255, 255),
        2
    )

    cv2.putText(
        area_preview,
        f"A={int(area)}",
        (x, max(y-10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )


print()
print(
    f"검사 Area : "
    f"{MIN_AREA} ~ {MAX_AREA}"
)

print(
    f"정상 Area : "
    f"{OK_MIN_AREA} ~ {OK_MAX_AREA}"
)

print("ENTER : 판별 시작")
print("ESC   : 종료")


cv2.imshow(
    "Contour Area Check",
    resize(
        area_preview,
        DISPLAY_SCALE
    )
)


while True:

    key = cv2.waitKey(0) & 0xFF

    if key == 13:
        break

    if key == 27:
        cv2.destroyAllWindows()
        exit()


cv2.destroyAllWindows()


# ============================================================
# 2단계 : 실제 판별
# ============================================================
result = cv2.cvtColor(
    gray,
    cv2.COLOR_GRAY2BGR
)


for cnt in contours:

    area = cv2.contourArea(cnt)

    # 검사 대상이 아닌 객체 제외
    if area < MIN_AREA or area > MAX_AREA:
        continue


    # =========================
    # Bounding Box
    # =========================
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = (
        max(w, h)
        / max(1, min(w, h))
    )


    # =========================
    # Circularity
    # =========================
    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if perimeter == 0:
        continue

    circularity = (
        4 * np.pi * area
        / (perimeter ** 2)
    )


    # =========================
    # Rectangularity
    # =========================
    rectangularity = (
        area / (w * h)
    )


    # =========================
    # 꼭짓점 수
    # =========================
    approx = cv2.approxPolyDP(
        cnt,
        0.04 * perimeter,
        True
    )

    vertices = len(approx)


    # =========================
    # 중심 밝은 영역
    # =========================
    center_mask = np.zeros_like(
        gray
    )

    cv2.drawContours(
        center_mask,
        [cnt],
        -1,
        255,
        -1
    )


    # =========================
    # 주변 Ring
    # =========================
    ring_kernel = np.ones(
        (RING_SIZE, RING_SIZE),
        np.uint8
    )

    expanded = cv2.dilate(
        center_mask,
        ring_kernel
    )

    ring_mask = cv2.subtract(
        expanded,
        center_mask
    )


    # =========================
    # 중심 / 주변 밝기
    # =========================
    center_mean = cv2.mean(
        gray,
        mask=center_mask
    )[0]

    outer_mean = cv2.mean(
        gray,
        mask=ring_mask
    )[0]

    contrast = (
        center_mean - outer_mean
    )


    # =========================
    # 주변 어두운 비율
    # =========================
    ring_pixels = gray[
        ring_mask > 0
    ]

    if len(ring_pixels) == 0:
        continue

    dark_ratio = np.mean(
        ring_pixels < DARK_THRESHOLD
    )


    # ========================================================
    # 판별
    # ========================================================

    # Bridge
    if aspect_ratio > BRIDGE_ASPECT:

        label = "BRIDGE"
        color = (0, 0, 255)


    # 납 부족
    elif (
        area < OK_MIN_AREA
        or dark_ratio < MIN_DARK_RATIO
    ):

        label = "INSUFFICIENT"
        color = (255, 0, 255)


    # 과납
    elif (
        area > OK_MAX_AREA
        or circularity > EXCESS_CIRCULARITY
    ):

        label = "EXCESS"
        color = (0, 165, 255)


    # 정상
    elif (
        OK_MIN_AREA <= area <= OK_MAX_AREA
        and 4 <= vertices <= 6
        and rectangularity > OK_RECTANGULARITY
        and dark_ratio >= MIN_DARK_RATIO
    ):

        label = "OK"
        color = (0, 255, 0)


    else:

        label = "UNKNOWN"
        color = (255, 255, 0)


    # =========================
    # 결과 표시
    # =========================
    cv2.rectangle(
        result,
        (x, y),
        (x+w, y+h),
        color,
        2
    )

    cv2.putText(
        result,
        f"{label} A={int(area)}",
        (x, max(y-10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


    print(
        label,
        "| Area:",
        round(area, 1),
        "| Circ:",
        round(circularity, 3),
        "| Aspect:",
        round(aspect_ratio, 3),
        "| Rect:",
        round(rectangularity, 3),
        "| Dark:",
        round(float(dark_ratio), 3),
        "| Contrast:",
        round(contrast, 1),
        "| Vertex:",
        vertices
    )


# =========================
# 결과
# =========================
cv2.imshow(
    "Binary",
    resize(
        binary,
        DISPLAY_SCALE
    )
)

cv2.imshow(
    "Inspection Result",
    resize(
        result,
        DISPLAY_SCALE
    )
)

cv2.waitKey(0)
cv2.destroyAllWindows()