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
        if "max_area" in params and params["max_area"] and params["min_area"] > params["max_area"]:
            raise ValueError("최소 면적은 최대 면적보다 클 수 없습니다.")
        if key == "rule" and params["min_count"] > params["max_count"]:
            raise ValueError("최소 객체 수는 최대 객체 수보다 클 수 없습니다.")
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
            elif key == "equalize": img = cv2.equalizeHist(img)
            elif key == "clahe": img = cv2.createCLAHE(p["clip_limit"], (p["grid"],)*2).apply(img)
            elif key == "gamma": img = cv2.LUT(img, np.rint(255*(np.arange(256)/255.)**p["gamma"]).astype(np.uint8))
            elif key == "normalize": img = cv2.normalize(img, None, p["low"], p["high"], cv2.NORM_MINMAX)
            elif key in ("threshold", "otsu"):
                flag = cv2.THRESH_BINARY if p["polarity"] == "bright" else cv2.THRESH_BINARY_INV
                threshold, img = cv2.threshold(img, p.get("value", 0), 255, flag | (cv2.THRESH_OTSU if key == "otsu" else 0))
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
            elif key == "morphology":
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
