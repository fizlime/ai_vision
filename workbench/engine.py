"""Standalone, stateless single-image pipeline. Also embedded in Python exports."""
import base64
import json
import math
import time
import cv2
import numpy as np


def number(label, default, minimum, maximum, step=1):
    return dict(label=label, default=default, min=minimum, max=maximum, step=step, kind="number")


def choice(label, default, options):
    return dict(label=label, default=default, options=options, kind="select")


CATALOG = {}


def operation(key, label, group, params, inputs=("image",), output="image", help=""):
    CATALOG[key] = dict(id=key, label=label, group=group, params=params, inputs=list(inputs), output=output, help=help)


operation("roi", "ROI 자르기", "입력", dict(x=number("X · px", 0, 0, 100000), y=number("Y · px", 0, 0, 100000), width=number("너비 · px (0 = 끝까지)", 0, 0, 100000), height=number("높이 · px (0 = 끝까지)", 0, 0, 100000)))
operation("gaussian", "Gaussian Blur", "필터", dict(kernel=number("커널 크기", 5, 1, 51, 2), sigma=number("Sigma (0 = 자동)", 0, 0, 30, .1)))
operation("median", "Median Blur", "필터", dict(kernel=number("커널 크기", 5, 3, 51, 2)))
operation("bilateral", "Bilateral Filter", "필터", dict(diameter=number("이웃 크기", 9, 1, 31, 2), sigma_color=number("색상 Sigma", 75, 1, 200), sigma_space=number("공간 Sigma", 75, 1, 200)))
operation("equalize", "Histogram Equalization", "밝기", {})
operation("clahe", "CLAHE", "밝기", dict(clip_limit=number("Clip limit", 2, .1, 20, .1), grid=number("Tile grid", 8, 1, 32)))
operation("gamma", "Gamma", "밝기", dict(gamma=number("Gamma", 1, .1, 5, .05)), help="출력 = 255 × (입력 / 255)^gamma. 1보다 크면 어두워집니다.")
operation("normalize", "Normalize", "밝기", dict(low=number("최솟값", 0, 0, 255), high=number("최댓값", 255, 0, 255)))
polarity = choice("극성", "bright", ["bright", "dark"])
operation("threshold", "Global Threshold", "분할", dict(value=number("임계값", 180, 0, 255), polarity=polarity), output="mask")
operation("otsu", "Otsu Threshold", "분할", dict(polarity=polarity), output="mask")
operation("adaptive", "Adaptive Threshold", "분할", dict(method=choice("방식", "gaussian", ["gaussian", "mean"]), block=number("Block size", 11, 3, 101, 2), c=number("C", 2, -50, 50, .5), polarity=polarity), output="mask")
operation("range", "Range Threshold", "분할", dict(low=number("하한", 100, 0, 255), high=number("상한", 220, 0, 255)), output="mask")
operation("canny", "Canny Edge", "에지", dict(low=number("Low", 50, 0, 255), high=number("High", 150, 0, 255)), output="edge")
operation("sobel", "Sobel", "에지", dict(axis=choice("방향", "magnitude", ["x", "y", "magnitude"]), kernel=choice("커널 크기", 3, [1, 3, 5, 7])), help="미분 절댓값을 0–255로 정규화한 영상입니다. 이후 Threshold로 분할할 수 있습니다.")
operation("laplacian", "Laplacian", "에지", dict(kernel=choice("커널 크기", 3, [1, 3, 5, 7])), help="미분 절댓값을 0–255로 정규화해 표시합니다.")
operation("morphology", "Morphology", "형태학", dict(operation=choice("연산", "opening", ["erosion", "dilation", "opening", "closing"]), shape=choice("커널 모양", "rectangle", ["rectangle", "ellipse", "cross"]), kernel=number("커널 크기", 3, 1, 51, 2), iterations=number("반복", 1, 1, 10)), inputs=("mask", "edge"), output="same")
operation("contours", "Contour Analysis", "측정", dict(min_area=number("최소 면적 · px²", 100, 0, 10000000), max_area=number("최대 면적 · px² (0 = 무제한)", 0, 0, 10000000)), inputs=("mask",), output="objects", help="외곽 윤곽 기준 면적과 형상을 측정합니다. 구멍은 면적에서 제외하지 않습니다.")
operation("blobs", "Connected Components", "측정", dict(min_area=number("최소 면적 · px²", 100, 0, 10000000), max_area=number("최대 면적 · px² (0 = 무제한)", 0, 0, 10000000)), inputs=("mask",), output="objects", help="8방향 연결 영역의 실제 픽셀 개수를 면적으로 사용합니다.")
operation("rule", "검사 조건 · OK / NG", "판정", dict(min_count=number("최소 객체 수", 1, 0, 10000), max_count=number("최대 객체 수", 100, 0, 10000), min_area=number("최소 면적 · px²", 100, 0, 10000000), max_area=number("최대 면적 · px² (0 = 무제한)", 0, 0, 10000000), min_circularity=number("최소 원형도", 0, 0, 1, .01), max_aspect=number("최대 장단변 비율", 100, 1, 100, .1)), inputs=("objects",), output="objects", help="모든 객체가 조건을 만족하고 객체 수가 범위 안이면 OK입니다. 객체 0개는 NG입니다.")

# New operations use the same catalog/recipe contract as the original steps.
operation("brightness_contrast", "밝기 / 대비", "밝기", dict(brightness=number("밝기 이동량", 0, -255, 255), contrast=number("대비 배율", 1, 0, 4, .05)), help="출력 = 입력 × 대비 + 밝기. 0–255 밖의 값은 잘라냅니다.")
operation("mean", "Mean / Box Blur", "필터", dict(width=number("커널 너비", 5, 1, 51, 2), height=number("커널 높이", 5, 1, 51, 2)), help="직사각형 이웃의 평균입니다. Gaussian / Median과 경계 보존을 비교하세요.")
operation("unsharp", "Unsharp Mask", "필터", dict(kernel=number("Gaussian 커널", 5, 1, 51, 2), amount=number("강조 강도", 1, 0, 5, .1), threshold=number("최소 밝기 차이", 0, 0, 255)), help="입력 + 강도 × (입력 − Blur). 최소 차이보다 작은 변화는 강조하지 않습니다.")
hat_params = dict(kernel=number("커널 크기", 21, 1, 151, 2), shape=choice("커널 모양", "ellipse", ["rectangle", "ellipse", "cross"]))
operation("tophat", "Top-hat · 밝은 영역", "밝기", hat_params, help="입력 − Opening. 커널보다 작은 밝은 영역을 강조한 후 Threshold로 분할하세요.")
operation("blackhat", "Black-hat · 어두운 영역", "밝기", hat_params, help="Closing − 입력. 작은 어두운 영역을 밝게 강조한 후 Threshold로 분할하세요.")
operation("gradient", "Morphological Gradient", "에지", dict(kernel=number("커널 크기", 3, 1, 51, 2), shape=choice("커널 모양", "rectangle", ["rectangle", "ellipse", "cross"])), inputs=("image", "mask"), output="image", help="Dilation − Erosion. 경계 강도 영상이므로 Threshold를 거쳐 객체를 측정하세요.")
operation("invert", "영상 / 마스크 반전", "변환", {}, inputs=("image", "mask", "edge"), output="same", help="255 − 입력. 영상, 영역 Mask, Edge의 입력 타입을 유지합니다.")
operation("fill_holes", "마스크 구멍 채우기", "마스크 정리", dict(max_area=number("최대 구멍 면적 · px² (0 = 모두)", 0, 0, 10000000)), inputs=("mask",), output="mask", help="영상 경계와 연결되지 않은 배경을 채웁니다. 구멍은 4방향 연결 픽셀 면적 기준입니다.")
operation("remove_small", "작은 영역 제거", "마스크 정리", dict(min_area=number("최소 유지 면적 · px²", 50, 0, 10000000), connectivity=choice("연결 방향", 8, [4, 8])), inputs=("mask",), output="mask", help="지정 면적보다 작은 연결 영역을 Mask에서 제거하고 다음 단계에 전달합니다.")
operation("background", "배경 밝기 보정", "밝기", dict(method=choice("배경 추정", "gaussian", ["gaussian", "opening"]), mode=choice("보정 방식", "subtract", ["subtract", "divide"]), kernel=number("배경 커널 크기", 51, 3, 301, 2), strength=number("보정 강도", 1, 0, 1, .05)), help="큰 커널로 추정한 배경을 빼거나 나누고 배경 평균 밝기로 맞춥니다. 강도 0은 원본입니다.")

operation("nlm", "Non-local Means · 노이즈 제거", "필터", dict(strength=number("노이즈 제거 강도 h", 10, 0, 50, .5), template=number("비교 패치 크기", 7, 3, 15, 2), search=number("검색 창 크기", 21, 3, 41, 2)), help="비슷한 패치를 찾아 평균내는 노이즈 제거입니다. h를 높이면 노이즈와 미세 결함이 함께 사라질 수 있습니다. 검색 창이 커질수록 처리 시간이 늘어납니다.")
operation("custom_filter", "Filter2D · 직접 커널 입력", "필터", dict(**{f"k{i}": number(f"커널 {i//3+1}행 {i%3+1}열", 1 if i == 4 else 0, -20, 20, .1) for i in range(9)}, divisor=number("커널 나눗값", 1, .1, 100, .1), delta=number("출력 밝기 이동", 0, -255, 255)), help="3×3 커널의 9개 계수를 직접 입력합니다. 계수 / 나눗값으로 filter2D를 수행하고 밝기 이동을 더합니다. 음수 및 255 초과 응답은 잘립니다. 기본값은 원본 유지입니다.")
operation("gabor", "Gabor · 방향성 / 텍스처", "필터", dict(kernel=number("커널 크기", 21, 3, 101, 2), sigma=number("Gaussian Sigma", 4, .1, 30, .1), theta=number("방향 · 도", 0, 0, 180, 1), wavelength=number("줄무늬 파장 · px", 10, 1, 100, .5), aspect=number("커널 종횡비", .5, .1, 1, .05), phase=number("위상 · 도", 0, -180, 180, 1)), help="특정 방향과 주기의 줄무늬를 강조합니다. PCB 패턴과 긁힘 비교에 사용합니다. 응답 절댓값을 0–255로 정규화하므로 서로 다른 실행의 밝기는 절대 응답 크기가 아닙니다.")
for key, label in (("scharr", "Scharr"), ("prewitt", "Prewitt"), ("roberts", "Roberts")):
    operation(key, label + " · 방향 미분", "에지", dict(axis=choice("방향", "magnitude", ["x", "y", "magnitude"])), help="방향별 밝기 변화 또는 두 방향의 합성 강도를 계산합니다. 절댓값을 0–255로 정규화합니다. 객체 측정 전에 Threshold로 마스크를 만드세요.")
operation("dog", "Difference of Gaussians", "에지", dict(kernel=number("커널 크기", 21, 3, 101, 2), sigma_small=number("작은 Sigma", 1, .1, 20, .1), sigma_large=number("큰 Sigma", 3, .1, 30, .1)), help="서로 다른 Sigma의 Gaussian 영상 차이로 특정 크기의 구조를 강조합니다. 큰 Sigma는 작은 Sigma보다 커야 합니다. 차이의 절댓값을 정규화해 표시합니다.")
operation("log", "Laplacian of Gaussian", "에지", dict(kernel=number("Gaussian 커널", 5, 3, 51, 2), sigma=number("Sigma", 1, .1, 20, .1), derivative=choice("Laplacian 커널", 3, [1, 3, 5, 7])), help="Gaussian으로 노이즈를 줄인 뒤 2차 미분합니다. 미세 경계와 점 결함을 강조하며 응답 절댓값을 0–255로 정규화합니다.")
operation("triangle", "Triangle Threshold", "분할", dict(polarity=polarity), output="mask", help="밝기 분포의 꼬리를 이용해 임계값을 자동 계산합니다. 한쪽에 배경 밝기가 몰린 영상에서 Otsu와 비교하세요. 극성으로 밝은 / 어두운 객체를 선택합니다.")
operation("threshold_modes", "Threshold · Truncate / To-zero", "분할", dict(value=number("임계값", 128, 0, 255), mode=choice("방식", "truncate", ["truncate", "to_zero", "to_zero_inv"])), help="Truncate는 임계값 위를 잘라내고, To-zero는 임계값 이하를 0으로 만듭니다. 반대 방식도 선택할 수 있습니다. 이진 마스크가 아닌 밝기 영상이므로 객체 측정 전 이진화가 필요합니다.")
operation("gray_morphology", "Grayscale Morphology", "형태학", dict(operation=choice("연산", "opening", ["erosion", "dilation", "opening", "closing"]), shape=choice("커널 모양", "ellipse", ["rectangle", "ellipse", "cross"]), kernel=number("커널 크기", 5, 1, 151, 2), iterations=number("반복", 1, 1, 10)), help="이진화 전에 원본 밝기 영상에 침식 / 팽창 / Opening / Closing을 적용합니다. Opening은 작은 밝은 구조, Closing은 작은 어두운 구조를 완화합니다.")
operation("distance", "Distance Transform", "마스크 정리", dict(metric=choice("거리 방식", "L2", ["L1", "L2", "chessboard"]), mask_size=choice("거리 커널", 3, [3, 5])), inputs=("mask",), output="image", help="흰 영역 내부에서 가장 가까운 검은 픽셀까지의 거리를 계산합니다. 표시 영상은 0–255 정규화이며 직접 px 거리가 아닙니다. 굵은 영역과 중심부 분할에 사용하세요.")
operation("clear_border", "경계 접촉 영역 제거", "마스크 정리", dict(connectivity=choice("연결 방향", 8, [4, 8])), inputs=("mask",), output="mask", help="영상 또는 ROI의 테두리에 닿은 흰 연결 영역 전체를 제거합니다. 잘린 객체를 측정 대상에서 제외할 때 사용합니다.")
operation("filter_area", "면적 범위로 영역 유지", "마스크 정리", dict(min_area=number("최소 픽셀 면적", 50, 0, 10000000), max_area=number("최대 픽셀 면적 (0 = 무제한)", 0, 0, 10000000), connectivity=choice("연결 방향", 8, [4, 8])), inputs=("mask",), output="mask", help="실제 연결 픽셀 면적이 지정 범위 안인 영역만 유지합니다. Contour 측정 전에 너무 작거나 큰 영역을 제거할 수 있습니다.")
operation("convex_hull", "Convex Hull · 볼록 영역", "마스크 정리", dict(mode=choice("결합 방식", "each", ["each", "all"])), inputs=("mask",), output="mask", help="각 외곽 윤곽 또는 모든 흰 픽셀을 감싸는 볼록 껍질을 채웁니다. 오목한 부분이 메워지고 구멍도 사라집니다. 실제 결함 면적 측정 전에 사용할 때 주의하세요.")
operation("edge_regions", "닫힌 Edge → Mask", "분할", dict(min_area=number("최소 윤곽 면적", 10, 0, 10000000)), inputs=("edge",), output="mask", help="Edge의 외곽 윤곽을 흰색으로 채워 영역 마스크로 변환합니다. 먼저 Closing으로 끊긴 경계를 연결하세요. 열린 경계에서도 인위적인 영역이 생길 수 있습니다.")
operation("watershed", "Watershed · 붙은 영역 분리", "마스크 정리", dict(seed_ratio=number("중심 거리 비율", .5, .05, .95, .05), min_seed=number("최소 씨앗 픽셀 면적", 3, 1, 100000)), inputs=("mask",), output="mask", help="거리 변환의 중심부를 씨앗으로 삼아 붙은 흰 영역을 나눕니다. 분리 경계를 검게 표시한 마스크가 출력됩니다. 씨앗 자체가 연결되어 있으면 분리되지 않으며 씨앗이 없는 영역은 유지합니다.")
operation("hough_lines", "Hough Lines · 선분 검출", "측정", dict(votes=number("최소 투표 수", 30, 1, 1000), min_length=number("최소 길이 · px", 30, 1, 10000), max_gap=number("허용 끊김 · px", 10, 0, 1000), theta=number("각도 간격 · 도", 1, .1, 10, .1), max_objects=number("최대 선분 수", 100, 1, 1000)), inputs=("edge", "mask"), output="objects", help="HoughLinesP로 직선을 검출합니다. 기존 객체 표에 중심, 경계 상자, 방향을 표시합니다. 선은 면적 0이며 닫힌 윤곽의 둘레 칸은 선 길이의 두 배입니다. Canny 다음에 연결하세요.")
operation("hough_circles", "Hough Circles · 원 검출", "측정", dict(dp=number("누산기 해상도 비율", 1, 1, 4, .1), min_distance=number("최소 중심 거리 · px", 20, 1, 10000), canny_high=number("내부 Canny 상한", 100, 1, 500), votes=number("중심 투표 임계값", 30, 1, 500), min_radius=number("최소 반지름 · px", 0, 0, 2000), max_radius=number("최대 반지름 (0 = 자동)", 0, 0, 2000), max_objects=number("최대 원 수", 100, 1, 1000)), output="objects", help="원형 에지를 검출합니다. Gaussian / Median으로 전처리하면 안정적입니다. 표의 면적은 검출 반지름으로 계산한 πr²이며 실제 연결 픽셀 면적이 아닙니다. 4메가픽셀 초과 영상은 ROI로 줄여주세요.")
operation("corners", "Shi–Tomasi · 코너 검출", "측정", dict(max_objects=number("최대 코너 수", 100, 1, 1000), quality=number("최소 품질 비율", .01, .001, 1, .001), min_distance=number("최소 코너 거리 · px", 10, 1, 1000), block=number("이웃 크기", 3, 3, 31, 2)), output="objects", help="두 방향의 변화가 강한 코너를 찾습니다. 정렬 기준점과 패턴 위치 확인에 사용합니다. 기존 표에는 점 좌표와 1×1 경계 상자를 표시하며 영역 면적은 0입니다.")
operation("harris", "Harris · 코너 응답", "에지", dict(block=number("이웃 크기", 3, 2, 31), aperture=choice("미분 커널", 3, [3, 5, 7]), k=number("Harris k", .04, .01, .2, .005)), help="Harris 코너 응답의 양수 부분을 0–255로 정규화합니다. 이후 Threshold로 강한 코너 위치를 영역으로 분할할 수 있습니다.")
operation("orb", "ORB · 특징점 검출", "측정", dict(max_objects=number("최대 특징점 수", 300, 1, 1000), scale=number("피라미드 배율", 1.2, 1.05, 2, .05), levels=number("피라미드 층 수", 8, 1, 12), fast_threshold=number("FAST 임계값", 20, 0, 255), edge=number("경계 제외 폭", 31, 0, 100), patch=number("패치 크기", 31, 3, 101, 2)), output="objects", help="다중 크기의 FAST 기반 특징점을 검출합니다. 위치, 방향, 지원 영역 크기를 기존 표에 표시합니다. 면적은 특징점 지원 원의 면적이며 결함 영역 면적이 아닙니다. 이미지 간 매칭은 수행하지 않습니다.")
operation("akaze", "AKAZE · 특징점 검출", "측정", dict(threshold=number("검출 임계값", .001, .00001, .1, .00001), octaves=number("Octave 수", 4, 1, 8), layers=number("Octave당 층 수", 4, 1, 8), max_objects=number("최대 특징점 수", 300, 1, 1000)), output="objects", help="비선형 스케일 공간에서 특징점을 검출합니다. ORB와 위치 안정성을 비교하세요. 면적은 특징점 지원 원의 면적입니다. 매칭은 수행하지 않으며 4메가픽셀 초과 영상은 ROI를 사용하세요.")
operation("template_match", "Template Matching · ROI 패턴 찾기", "측정", dict(x=number("템플릿 X · 현재 영상 좌표", 0, 0, 100000), y=number("템플릿 Y · 현재 영상 좌표", 0, 0, 100000), width=number("템플릿 너비 · px", 32, 1, 2000), height=number("템플릿 높이 · px", 32, 1, 2000), method=choice("매칭 방식", "ccoeff", ["ccoeff", "ccorr", "sqdiff"]), score=number("최소 유사도", .8, 0, 1, .01), min_distance=number("중복 억제 거리 · px", 16, 1, 2000), exclude_source=choice("원래 템플릿 위치 제외", "yes", ["yes", "no"]), max_objects=number("최대 검출 수", 50, 1, 1000)), output="objects", help="현재 처리 영상의 지정 ROI를 템플릿으로 사용해 같은 영상의 반복 패턴을 찾습니다. 정규화 매칭이며 SQDIFF는 1−차이를 유사도로 사용합니다. 회전 / 크기 변화는 대응하지 않습니다. 평탄한 템플릿은 검출하지 않습니다. 좌표는 현재 영상 기준입니다.")

# Explanations for original operations are shown both before and after adding a step.
for key, description in {
    "roi": "원본 좌표 기준으로 검사 영역을 자릅니다. 너비 / 높이 0은 현재 영상 끝까지입니다. 이미지에서 ROI 그리기로 선택하거나 좌표를 직접 입력할 수 있습니다.",
    "gaussian": "Gaussian 가중 평균으로 노이즈를 완화합니다. 커널이 커질수록 작은 경계가 흐려집니다. Sigma 0은 커널 크기에서 자동 결정합니다.",
    "median": "이웃 밝기의 중앙값을 사용합니다. 점 형태의 소금 / 후추 노이즈 제거에 유용하며 커널이 커지면 작은 결함도 사라질 수 있습니다.",
    "bilateral": "거리와 밝기 차이를 함께 고려해 경계를 보존하며 평활화합니다. 색상 Sigma는 Mono8 밝기 차이를 뜻합니다. 이웃 크기와 Sigma를 높일수록 강하게 평활화합니다.",
    "equalize": "전체 밝기 분포를 재배치해 대비를 높입니다. 노이즈도 함께 커질 수 있습니다. CLAHE와 비교하세요. 히스토그램 그래프는 표시하지 않습니다.",
    "clahe": "타일별로 대비를 조절하되 Clip limit으로 과도한 증폭을 제한합니다. Grid는 타일 개수이며 커널 픽셀 크기가 아닙니다.",
    "normalize": "현재 영상의 최소 / 최대 밝기를 지정한 출력 범위로 선형 변환합니다. 장면에 따라 같은 입력 밝기의 출력이 달라질 수 있습니다.",
    "threshold": "한 개의 수동 임계값으로 흰 객체와 검은 배경을 나눕니다. Bright는 임계값보다 밝은 픽셀, Dark는 임계값 이하 픽셀을 흰색으로 만듭니다.",
    "otsu": "밝기 분포에서 두 부류를 나누는 임계값을 자동 계산합니다. 계산된 임계값은 실행 값으로 표시합니다. 수동 조절이 필요하면 Global Threshold를 사용하세요.",
    "adaptive": "각 픽셀 주변의 Mean 또는 Gaussian 가중 평균에서 C를 뺀 값을 임계값으로 사용합니다. Block size는 홀수입니다. 조명 편차가 있는 영상에서 비교하세요.",
    "range": "하한 이상, 상한 이하 밝기만 흰색으로 선택합니다. 양 끝값을 포함하며 지정 범위 밖은 배경이 됩니다.",
    "canny": "노이즈 완화 후 두 임계값으로 연결된 경계를 추적합니다. Low는 약한 경계, High는 강한 경계 기준입니다. Edge 출력이므로 영역 측정에는 닫힌 Edge → Mask가 필요합니다.",
    "morphology": "이진 Mask / Edge에 침식, 팽창, Opening, Closing을 적용합니다. Opening은 작은 흰 노이즈를 제거하고 Closing은 작은 검은 틈을 메웁니다. 커널 모양, 크기, 반복 횟수를 조절하세요.",
}.items():
    CATALOG[key]["help"] = description

# Optional parameter hints keep the existing recipe values and export format unchanged.
PARAM_HINTS = {
    "kernel": "커널의 픽셀 크기입니다. 값이 클수록 더 넓은 주변 영역을 사용합니다. 홀수만 입력하세요.",
    "min_area": "이 값 이상인 면적을 유지하거나 허용합니다. 기능 설명에서 픽셀 면적 / 윤곽 면적 기준을 확인하세요.",
    "max_area": "이 값 이하인 면적을 허용합니다. 0은 상한을 적용하지 않습니다.",
    "connectivity": "4는 상하좌우, 8은 대각선까지 연결된 것으로 봅니다.",
    "polarity": "Bright는 밝은 객체, Dark는 어두운 객체를 흰색 전경으로 만듭니다.",
    "iterations": "같은 형태학 연산을 반복할 횟수입니다.",
    "max_objects": "검출 결과 수의 상한입니다. 기존 측정 결과 표에 표시됩니다.",
}
for spec in CATALOG.values():
    for key, meta in spec["params"].items():
        meta["help"] = PARAM_HINTS.get(key, "")
        if spec["id"] == "fill_holes" and key == "max_area": meta["help"] = "이 픽셀 면적 이하의 내부 구멍을 채웁니다. 0은 모든 내부 구멍을 채웁니다."


def validate(recipe):
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 1:
        raise ValueError("지원하는 Recipe schema_version은 1입니다.")
    if not isinstance(recipe.get("name"), str) or not recipe["name"].strip() or len(recipe["name"]) > 120:
        raise ValueError("Recipe 이름은 1–120자여야 합니다.")
    nodes = recipe.get("nodes")
    if not isinstance(nodes, list) or len(nodes) > 24:
        raise ValueError("단계는 최대 24개입니다.")
    seen, kind, clean = set(), "image", []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("올바르지 않은 단계입니다.")
        nid, key = node.get("id"), node.get("type")
        if not isinstance(nid, str) or not nid or len(nid) > 80 or nid in seen:
            raise ValueError("단계 ID는 중복 없는 문자열이어야 합니다.")
        seen.add(nid)
        if not isinstance(key, str) or key not in CATALOG:
            raise ValueError(f"지원하지 않는 기능: {key}")
        spec = CATALOG[key]
        enabled = node.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("enabled는 boolean이어야 합니다.")
        raw = node.get("params", {})
        if not isinstance(raw, dict) or set(raw) - set(spec["params"]):
            raise ValueError(f"{spec['label']}: 알 수 없는 파라미터입니다.")
        params = {}
        for keyp, meta in spec["params"].items():
            value = raw.get(keyp, meta["default"])
            if meta["kind"] == "select":
                if isinstance(value, bool) or value not in meta["options"]:
                    raise ValueError(f"{spec['label']}: {keyp} 선택값이 잘못되었습니다.")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not meta["min"] <= value <= meta["max"]:
                    raise ValueError(f"{spec['label']}: {meta['label']} 범위를 확인하세요.")
                if meta["step"] >= 1:
                    if int(value) != value or (value - meta["min"]) % meta["step"]:
                        raise ValueError(f"{spec['label']}: {meta['label']} 간격을 확인하세요.")
                    value = int(value)
            params[keyp] = value
        if key in ("range", "canny", "normalize") and params["low"] > params["high"]:
            raise ValueError(f"{spec['label']}: 하한은 상한보다 클 수 없습니다.")
        if "min_area" in params and "max_area" in params and params["max_area"] and params["min_area"] > params["max_area"]:
            raise ValueError("최소 면적은 최대 면적보다 클 수 없습니다.")
        if key == "rule" and params["min_count"] > params["max_count"]:
            raise ValueError("최소 객체 수는 최대 객체 수보다 클 수 없습니다.")
        if key == "nlm" and params["template"] > params["search"]:
            raise ValueError("NLM 검색 창은 비교 패치 이상이어야 합니다.")
        if key == "dog" and params["sigma_small"] >= params["sigma_large"]:
            raise ValueError("큰 Sigma는 작은 Sigma보다 커야 합니다.")
        if key == "hough_circles" and params["max_radius"] and params["min_radius"] > params["max_radius"]:
            raise ValueError("최소 반지름은 최대 반지름보다 클 수 없습니다.")
        if enabled:
            if kind not in spec["inputs"]:
                raise ValueError(f"{spec['label']}: 입력 {kind}에 연결할 수 없습니다. 필요한 타입: {', '.join(spec['inputs'])}")
            kind = kind if spec["output"] == "same" else spec["output"]
        clean.append(dict(id=nid, type=key, enabled=enabled, params=params))
    return dict(schema_version=1, name=recipe["name"].strip(), nodes=clean)


def decode_image(data):
    if len(data) > 40 * 1024 * 1024:
        raise ValueError("이미지는 40MB 이하여야 합니다.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. PNG / JPG / 단일 페이지 TIFF를 사용하세요.")
    if img.shape[0] * img.shape[1] > 24_000_000:
        raise ValueError("이미지는 24메가픽셀 이하여야 합니다.")
    if img.dtype != np.uint8:
        raise ValueError("현재 8-bit 이미지만 지원합니다. Mono16 이미지는 먼저 명시적으로 변환하세요.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return img


def preview(img):
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        img = cv2.resize(img, (max(1, round(w * 1200 / max(h, w))), max(1, round(h * 1200 / max(h, w)))), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Preview 변환 실패")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def measure(contour, area, offset, oid):
    x, y, w, h = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    moments = cv2.moments(contour)
    cx = moments["m10"] / moments["m00"] if moments["m00"] else x + w / 2
    cy = moments["m01"] / moments["m00"] if moments["m00"] else y + h / 2
    (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
    return dict(id=oid, area=float(area), perimeter=float(perimeter), width=w, height=h, x=x+offset[0], y=y+offset[1], cx=cx+offset[0], cy=cy+offset[1], circularity=4*math.pi*cv2.contourArea(contour)/(perimeter**2) if perimeter else 0., solidity=cv2.contourArea(contour)/hull_area if hull_area else 0., aspect_ratio=max(w,h)/max(1,min(w,h)), rotated_width=float(rw), rotated_height=float(rh), orientation=float(angle))


def response_image(values):
    return cv2.normalize(np.maximum(values, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def circle_contour(x, y, radius):
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    return np.rint(np.column_stack((x+radius*np.cos(angles), y+radius*np.sin(angles)))).astype(np.int32).reshape(-1, 1, 2)


def detection_preview(source, detections, offset):
    """Use the original object schema and bounding-box preview for every detector."""
    image, objects = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR), []
    for contour, area, extra in detections:
        obj = measure(contour, area, offset, len(objects)+1)
        obj.update(extra)
        objects.append(obj)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (100, 220, 80), 1)
        cv2.putText(image, str(obj["id"]), (x, max(12, y-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 220, 255), 1)
    return image, objects


def run_pipeline(original, recipe, include_previews=True):
    recipe = validate(recipe)
    img, offset, kind = original.copy(), [0, 0], "image"
    steps, objects, decision = [], [], None
    started = time.perf_counter()
    for node in recipe["nodes"]:
        tick = time.perf_counter()
        key, p, computed = node["type"], node["params"], {}
        if node["enabled"]:
            if key == "roi":
                x, y = p["x"] - offset[0], p["y"] - offset[1]
                w, h = p["width"] or img.shape[1]-x, p["height"] or img.shape[0]-y
                if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0] or w <= 0 or h <= 0:
                    raise ValueError("ROI가 현재 영상 범위를 벗어났습니다.")
                img = img[y:y+h, x:x+w].copy()
                offset[0] += x; offset[1] += y
                computed = dict(x=offset[0], y=offset[1], width=w, height=h)
            elif key == "gaussian": img = cv2.GaussianBlur(img, (p["kernel"],)*2, p["sigma"], borderType=cv2.BORDER_REFLECT_101)
            elif key == "median": img = cv2.medianBlur(img, p["kernel"])
            elif key == "bilateral": img = cv2.bilateralFilter(img, p["diameter"], p["sigma_color"], p["sigma_space"])
            elif key == "brightness_contrast":
                img = np.clip(np.rint(img.astype(np.float32)*p["contrast"]+p["brightness"]), 0, 255).astype(np.uint8)
            elif key == "mean": img = cv2.blur(img, (p["width"], p["height"]), borderType=cv2.BORDER_REFLECT_101)
            elif key == "nlm": img = cv2.fastNlMeansDenoising(img, None, p["strength"], p["template"], p["search"])
            elif key == "custom_filter":
                kernel = np.array([p[f"k{i}"] for i in range(9)], np.float32).reshape(3, 3)/p["divisor"]
                img = cv2.filter2D(img, -1, kernel, delta=p["delta"], borderType=cv2.BORDER_REFLECT_101)
            elif key == "gabor":
                kernel = cv2.getGaborKernel((p["kernel"],)*2, p["sigma"], math.radians(p["theta"]), p["wavelength"], p["aspect"], math.radians(p["phase"]), ktype=cv2.CV_32F)
                img = response_image(np.abs(cv2.filter2D(img, cv2.CV_32F, kernel)))
            elif key in ("scharr", "prewitt", "roberts"):
                if key == "scharr":
                    gx, gy = cv2.Scharr(img, cv2.CV_32F, 1, 0), cv2.Scharr(img, cv2.CV_32F, 0, 1)
                else:
                    kx = np.array([[-1, 0, 1]]*3 if key == "prewitt" else [[1, 0], [0, -1]], np.float32)
                    ky = np.array([[-1]*3, [0]*3, [1]*3] if key == "prewitt" else [[0, 1], [-1, 0]], np.float32)
                    gx, gy = cv2.filter2D(img, cv2.CV_32F, kx), cv2.filter2D(img, cv2.CV_32F, ky)
                img = response_image(cv2.magnitude(gx, gy) if p["axis"] == "magnitude" else np.abs(gx if p["axis"] == "x" else gy))
            elif key == "dog":
                source = img.astype(np.float32)
                a = cv2.GaussianBlur(source, (p["kernel"],)*2, p["sigma_small"])
                b = cv2.GaussianBlur(source, (p["kernel"],)*2, p["sigma_large"])
                img = response_image(np.abs(a-b))
            elif key == "log":
                blurred = cv2.GaussianBlur(img.astype(np.float32), (p["kernel"],)*2, p["sigma"])
                img = response_image(np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=p["derivative"])))
            elif key == "harris": img = response_image(cv2.cornerHarris(img.astype(np.float32), p["block"], p["aperture"], p["k"]))
            elif key == "unsharp":
                source = img.astype(np.float32)
                blurred = cv2.GaussianBlur(img, (p["kernel"],)*2, 0, borderType=cv2.BORDER_REFLECT_101).astype(np.float32)
                detail = source - blurred
                detail[np.abs(detail) < p["threshold"]] = 0
                img = np.clip(np.rint(source+p["amount"]*detail), 0, 255).astype(np.uint8)
            elif key in ("tophat", "blackhat", "gradient"):
                shape = dict(rectangle=cv2.MORPH_RECT, ellipse=cv2.MORPH_ELLIPSE, cross=cv2.MORPH_CROSS)[p["shape"]]
                op = dict(tophat=cv2.MORPH_TOPHAT, blackhat=cv2.MORPH_BLACKHAT, gradient=cv2.MORPH_GRADIENT)[key]
                img = cv2.morphologyEx(img, op, cv2.getStructuringElement(shape, (p["kernel"],)*2))
            elif key == "invert": img = cv2.bitwise_not(img)
            elif key == "fill_holes":
                background_mask = (img == 0).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(background_mask, connectivity=4)
                boundary = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
                fill = np.ones(count, dtype=bool)
                fill[boundary] = False
                fill[0] = False
                if p["max_area"]: fill &= stats[:, cv2.CC_STAT_AREA] <= p["max_area"]
                computed = dict(filled_holes=int(fill.sum()), filled_pixels=int(stats[fill, cv2.CC_STAT_AREA].sum()))
                img = img.copy()
                img[fill[labels]] = 255
            elif key == "remove_small":
                count, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=p["connectivity"])
                keep = stats[:, cv2.CC_STAT_AREA] >= p["min_area"]
                keep[0] = False
                computed = dict(removed_objects=int(count-1-keep.sum()), removed_pixels=int(stats[1:, cv2.CC_STAT_AREA][~keep[1:]].sum()))
                img = (keep[labels].astype(np.uint8)*255)
            elif key == "background":
                if p["method"] == "gaussian":
                    background = cv2.GaussianBlur(img, (p["kernel"],)*2, 0, borderType=cv2.BORDER_REFLECT_101)
                else:
                    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["kernel"],)*2))
                source, baseline = img.astype(np.float32), background.astype(np.float32)
                mean = float(baseline.mean())
                corrected = source-baseline+mean if p["mode"] == "subtract" else source/np.maximum(baseline, 1)*mean
                img = np.clip(np.rint(source+p["strength"]*(corrected-source)), 0, 255).astype(np.uint8)
                computed["background_mean"] = round(mean, 4)
            elif key == "threshold_modes":
                mode = dict(truncate=cv2.THRESH_TRUNC, to_zero=cv2.THRESH_TOZERO, to_zero_inv=cv2.THRESH_TOZERO_INV)[p["mode"]]
                _, img = cv2.threshold(img, p["value"], 255, mode)
            elif key == "distance":
                metric = dict(L1=cv2.DIST_L1, L2=cv2.DIST_L2, chessboard=cv2.DIST_C)[p["metric"]]
                img = response_image(cv2.distanceTransform(img, metric, p["mask_size"]))
            elif key in ("clear_border", "filter_area"):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=p["connectivity"])
                keep = np.ones(count, dtype=bool)
                keep[0] = False
                if key == "clear_border":
                    keep[np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))] = False
                else:
                    keep &= stats[:, cv2.CC_STAT_AREA] >= p["min_area"]
                    if p["max_area"]: keep &= stats[:, cv2.CC_STAT_AREA] <= p["max_area"]
                img = keep[labels].astype(np.uint8)*255
            elif key in ("convex_hull", "edge_regions"):
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if key == "convex_hull":
                    if p["mode"] == "all" and contours: contours = [np.concatenate(contours)]
                    contours = [cv2.convexHull(c) for c in contours]
                else: contours = [c for c in contours if cv2.contourArea(c) >= p["min_area"]]
                img = np.zeros_like(img)
                cv2.drawContours(img, contours, -1, 255, cv2.FILLED)
            elif key == "watershed":
                mask = img.copy()
                distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                seeds = (distance > p["seed_ratio"]*float(distance.max())).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
                keep = stats[:, cv2.CC_STAT_AREA] >= p["min_seed"]
                keep[0] = False
                seeds = keep[labels].astype(np.uint8)
                count, markers = cv2.connectedComponents(seeds, connectivity=8)
                if count > 1:
                    markers += 1
                    markers[(mask != 0) & (seeds == 0)] = 0
                    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
                    _, components = cv2.connectedComponents(mask, connectivity=8)
                    seeded = np.unique(components[seeds != 0])
                    img = mask.copy()
                    img[(markers == -1) & np.isin(components, seeded)] = 0
            elif key in ("hough_lines", "hough_circles", "corners", "orb", "akaze", "template_match"):
                detections = []
                if key == "hough_lines":
                    lines = cv2.HoughLinesP(img, 1, math.radians(p["theta"]), p["votes"], minLineLength=p["min_length"], maxLineGap=p["max_gap"])
                    if lines is not None:
                        for x1, y1, x2, y2 in lines[:p["max_objects"], 0]:
                            contour = np.array([(x1, y1), (x2, y2)], np.int32).reshape(-1, 1, 2)
                            detections.append((contour, 0, dict(cx=float(x1+x2)/2+offset[0], cy=float(y1+y2)/2+offset[1], orientation=math.degrees(math.atan2(float(y2-y1), float(x2-x1))), length=math.hypot(float(x2-x1), float(y2-y1)))))
                elif key == "hough_circles":
                    if img.size > 4_000_000: raise ValueError("Hough Circles는 4메가픽셀 이하 ROI에서 실행하세요.")
                    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, p["dp"], p["min_distance"], param1=p["canny_high"], param2=p["votes"], minRadius=p["min_radius"], maxRadius=p["max_radius"])
                    if circles is not None:
                        for x, y, radius in circles[0, :p["max_objects"]]:
                            detections.append((circle_contour(x, y, radius), math.pi*float(radius)**2, dict(cx=float(x)+offset[0], cy=float(y)+offset[1], radius=float(radius))))
                elif key == "corners":
                    points = cv2.goodFeaturesToTrack(img, p["max_objects"], p["quality"], p["min_distance"], blockSize=p["block"])
                    if points is not None:
                        for x, y in points[:, 0]:
                            detections.append((np.rint([[[x, y]]]).astype(np.int32), 0, dict(cx=float(x)+offset[0], cy=float(y)+offset[1])))
                elif key in ("orb", "akaze"):
                    if key == "orb":
                        detector = cv2.ORB_create(nfeatures=p["max_objects"], scaleFactor=p["scale"], nlevels=p["levels"], edgeThreshold=p["edge"], patchSize=p["patch"], fastThreshold=p["fast_threshold"])
                    else:
                        if img.size > 4_000_000: raise ValueError("AKAZE는 4메가픽셀 이하 ROI에서 실행하세요.")
                        detector = cv2.AKAZE_create(threshold=p["threshold"], nOctaves=p["octaves"], nOctaveLayers=p["layers"])
                    points = sorted(detector.detect(img, None), key=lambda pt: (-pt.response, pt.pt))[:p["max_objects"]]
                    for point in points:
                        x, y = point.pt
                        radius = point.size/2
                        detections.append((circle_contour(x, y, radius), math.pi*radius**2, dict(cx=x+offset[0], cy=y+offset[1], orientation=point.angle, response=point.response)))
                else:
                    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
                    if x+w > img.shape[1] or y+h > img.shape[0]: raise ValueError("템플릿 ROI가 현재 영상 범위를 벗어났습니다.")
                    template = img[y:y+h, x:x+w]
                    if float(template.std()) < 1e-6:
                        computed["warning"] = "평탄한 템플릿입니다. 패턴이 포함된 ROI를 지정하세요."
                    else:
                        method = dict(ccoeff=cv2.TM_CCOEFF_NORMED, ccorr=cv2.TM_CCORR_NORMED, sqdiff=cv2.TM_SQDIFF_NORMED)[p["method"]]
                        scores = cv2.matchTemplate(img, template, method)
                        if p["method"] == "sqdiff": scores = 1-scores
                        scores[~np.isfinite(scores)] = -1
                        radius = p["min_distance"]
                        if p["exclude_source"] == "yes": scores[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = -1
                        for _ in range(p["max_objects"]):
                            _, score, _, (mx, my) = cv2.minMaxLoc(scores)
                            if score < p["score"]: break
                            contour = np.array([(mx,my), (mx+w-1,my), (mx+w-1,my+h-1), (mx,my+h-1)], np.int32).reshape(-1,1,2)
                            detections.append((contour, w*h, dict(score=score, cx=mx+(w-1)/2+offset[0], cy=my+(h-1)/2+offset[1])))
                            scores[max(0,my-radius):my+radius+1, max(0,mx-radius):mx+radius+1] = -1
                img, objects = detection_preview(img, detections, offset)
                computed["object_count"] = len(objects)
            elif key == "equalize": img = cv2.equalizeHist(img)
            elif key == "clahe": img = cv2.createCLAHE(p["clip_limit"], (p["grid"],)*2).apply(img)
            elif key == "gamma": img = cv2.LUT(img, np.rint(255*(np.arange(256)/255.)**p["gamma"]).astype(np.uint8))
            elif key == "normalize": img = cv2.normalize(img, None, p["low"], p["high"], cv2.NORM_MINMAX)
            elif key in ("threshold", "otsu", "triangle"):
                flag = cv2.THRESH_BINARY if p["polarity"] == "bright" else cv2.THRESH_BINARY_INV
                threshold, img = cv2.threshold(img, p.get("value", 0), 255, flag | (cv2.THRESH_OTSU if key == "otsu" else cv2.THRESH_TRIANGLE if key == "triangle" else 0))
                computed["threshold"] = threshold
            elif key == "adaptive": img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C if p["method"] == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY if p["polarity"] == "bright" else cv2.THRESH_BINARY_INV, p["block"], p["c"])
            elif key == "range": img = cv2.inRange(img, p["low"], p["high"])
            elif key == "canny": img = cv2.Canny(img, p["low"], p["high"])
            elif key == "sobel":
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=p["kernel"])
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=p["kernel"])
                values = cv2.magnitude(gx, gy) if p["axis"] == "magnitude" else np.abs(gx if p["axis"] == "x" else gy)
                img = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            elif key == "laplacian": img = cv2.normalize(np.abs(cv2.Laplacian(img, cv2.CV_32F, ksize=p["kernel"])), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            elif key in ("morphology", "gray_morphology"):
                shape = dict(rectangle=cv2.MORPH_RECT, ellipse=cv2.MORPH_ELLIPSE, cross=cv2.MORPH_CROSS)[p["shape"]]
                op = dict(erosion=cv2.MORPH_ERODE, dilation=cv2.MORPH_DILATE, opening=cv2.MORPH_OPEN, closing=cv2.MORPH_CLOSE)[p["operation"]]
                img = cv2.morphologyEx(img, op, cv2.getStructuringElement(shape, (p["kernel"],)*2), iterations=p["iterations"])
            elif key in ("contours", "blobs"):
                mask = img
                pairs = []
                if key == "contours":
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pairs = [(c, cv2.contourArea(c), None) for c in contours]
                else:
                    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
                    for i in range(1, count):
                        area = int(stats[i, cv2.CC_STAT_AREA])
                        if area < p["min_area"] or (p["max_area"] and area > p["max_area"]): continue
                        x, y, w, h = map(int, stats[i, :4])
                        local = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
                        cs, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x,y))
                        if cs: pairs.append((max(cs, key=cv2.contourArea), area, centers[i]))
                objects = []
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for c, area, centroid in sorted(pairs, key=lambda pair: cv2.boundingRect(pair[0])[::-1]):
                    if area < p["min_area"] or (p["max_area"] and area > p["max_area"]): continue
                    if len(objects) >= 1000: raise ValueError("객체가 1,000개를 초과합니다. 최소 면적을 높여 주세요.")
                    obj = measure(c, area, offset, len(objects)+1)
                    if centroid is not None: obj.update(cx=float(centroid[0])+offset[0], cy=float(centroid[1])+offset[1])
                    objects.append(obj)
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (100,220,80), 1)
                    cv2.putText(img, str(obj["id"]), (x,max(12,y-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (80,220,255), 1)
                computed["object_count"] = len(objects)
            elif key == "rule":
                for obj in objects:
                    reasons = []
                    if obj["area"] < p["min_area"] or (p["max_area"] and obj["area"] > p["max_area"]): reasons.append("면적")
                    if obj["circularity"] < p["min_circularity"]: reasons.append("원형도")
                    if obj["aspect_ratio"] > p["max_aspect"]: reasons.append("장단변 비율")
                    obj.update(result="NG" if reasons else "OK", reasons=reasons)
                count_ok = bool(objects) and p["min_count"] <= len(objects) <= p["max_count"]
                decision = dict(result="OK" if count_ok and all(o["result"] == "OK" for o in objects) else "NG", count_ok=count_ok)
                computed.update(decision)
            kind = kind if CATALOG[key]["output"] == "same" else CATALOG[key]["output"]
        step = dict(id=node["id"], type=key, enabled=node["enabled"], kind=kind, width=img.shape[1], height=img.shape[0], offset=list(offset), ms=round((time.perf_counter()-tick)*1000,2), computed=computed, objects=json.loads(json.dumps(objects)), decision=decision)
        if include_previews: step["preview"] = preview(img)
        steps.append(step)
    result = dict(recipe=recipe, steps=steps, objects=objects, decision=decision, total_ms=round((time.perf_counter()-started)*1000,2), source=dict(width=original.shape[1], height=original.shape[0]), versions=dict(opencv=cv2.__version__, numpy=np.__version__))
    if include_previews: result["original"] = preview(original)
    return result, img
