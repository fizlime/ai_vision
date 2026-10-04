import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient
from workbench.engine import CATALOG, decode_image, run_pipeline, validate
from workbench.server import app


def recipe(*items):
    return dict(schema_version=1, name="검사 테스트", nodes=[dict(id=f"node-{i}", type=key, enabled=True, params=params) for i, (key, params) in enumerate(items)])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((120, 180), np.uint8)
        self.img[30:70, 50:100] = 220

    def test_crop_original_coordinates_and_measurements(self):
        r = recipe(("roi", dict(x=20,y=10,width=120,height=90)), ("roi", dict(x=40,y=20,width=70,height=60)), ("threshold", dict(value=180)), ("blobs", dict(min_area=1)))
        result, output = run_pipeline(self.img, r)
        obj = result["objects"][0]
        self.assertEqual((obj["x"],obj["y"],obj["width"],obj["height"],obj["area"]), (50,30,50,40,2000))
        self.assertEqual((obj["cx"],obj["cy"]), (74.5,49.5))
        self.assertEqual(result["steps"][1]["offset"], [40,20])
        self.assertEqual(output.shape, (60,70,3))

    def test_all_operations_and_round_trip(self):
        for key, spec in CATALOG.items():
            with self.subTest(operation=key):
                prefix = []
                if spec["inputs"] == ["objects"]: prefix = [("threshold", {}), ("contours", {})]
                elif spec["inputs"] == ["edge"]: prefix = [("canny", {})]
                elif "image" not in spec["inputs"]: prefix = [("threshold", {})]
                r = recipe(*prefix, (key, {}))
                clean = validate(r)
                a, img_a = run_pipeline(self.img, clean, False)
                b, img_b = run_pipeline(self.img, json.loads(json.dumps(clean)), False)
                np.testing.assert_array_equal(img_a,img_b)
                self.assertEqual(a["objects"], b["objects"])

    def test_matches_opencv_filter_chain(self):
        r = recipe(("gaussian",dict(kernel=5)),("threshold",dict(value=180)),("morphology",dict(operation="opening")))
        _, actual = run_pipeline(self.img,r)
        expected=cv2.GaussianBlur(self.img,(5,5),0)
        _,expected=cv2.threshold(expected,180,255,cv2.THRESH_BINARY)
        expected=cv2.morphologyEx(expected,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        np.testing.assert_array_equal(actual,expected)

    def test_validation(self):
        invalid = [recipe(("gaussian",dict(kernel=4))),recipe(("range",dict(low=200,high=10))),recipe(("contours",{})),recipe(("canny",{}),("contours",{})),recipe(("gamma",dict(gamma=float('nan')))),recipe(("median",dict(kernel=True))),recipe(("gaussian",dict(unknown=3)))]
        for r in invalid:
            with self.assertRaises(ValueError): validate(r)
        r=recipe(("threshold",{}),("contours",{}));r["nodes"][0]["enabled"]=False
        with self.assertRaises(ValueError): validate(r)

    def test_invalid_roi(self):
        with self.assertRaises(ValueError):run_pipeline(self.img,recipe(("roi",dict(x=200))))
        with self.assertRaises(ValueError):run_pipeline(self.img,recipe(("roi",dict(x=20)),("roi",dict(x=0))))

    def test_empty_is_ng_and_otsu_recorded(self):
        result,_=run_pipeline(np.zeros_like(self.img),recipe(("otsu",{}),("contours",{}),("rule",dict(min_count=0))))
        self.assertEqual(result["decision"]["result"],"NG")
        self.assertIn("threshold",result["steps"][0]["computed"])

    def test_rule_does_not_mutate_prior_steps(self):
        result,_=run_pipeline(self.img,recipe(("threshold",{}),("contours",{}),("rule",dict(min_area=9999))))
        self.assertNotIn("result",result["steps"][1]["objects"][0])
        self.assertEqual(result["objects"][0]["result"],"NG")

    def test_mono16_rejected(self):
        _,data=cv2.imencode('.png',self.img.astype(np.uint16))
        with self.assertRaises(ValueError):decode_image(data.tobytes())

    def test_brightness_clips_without_absolute_value_or_wraparound(self):
        image = np.array([[0, 20, 100, 200, 255]], np.uint8)
        _, actual = run_pipeline(image, recipe(("brightness_contrast", dict(brightness=-30, contrast=2))), False)
        np.testing.assert_array_equal(actual, [[0, 10, 170, 255, 255]])

    def test_rectangular_mean_and_unsharp_identity(self):
        image = np.arange(35, dtype=np.uint8).reshape(5, 7)
        _, actual = run_pipeline(image, recipe(("mean", dict(width=3, height=1))), False)
        self.assertEqual(actual[2, 3], int(image[2, 2:5].mean()))
        for params in (dict(amount=0), dict(amount=5, threshold=255)):
            _, actual = run_pipeline(image, recipe(("unsharp", params)), False)
            np.testing.assert_array_equal(actual, image)

    def test_hat_polarity_and_gradient_threshold_contract(self):
        image = np.full((31, 31), 100, np.uint8)
        image[7:10, 7:10] = 200
        image[21:24, 21:24] = 20
        _, bright = run_pipeline(image, recipe(("tophat", dict(kernel=7, shape="rectangle"))), False)
        _, dark = run_pipeline(image, recipe(("blackhat", dict(kernel=7, shape="rectangle"))), False)
        self.assertEqual((int(bright[8, 8]), int(bright[22, 22])), (100, 0))
        self.assertEqual((int(dark[8, 8]), int(dark[22, 22])), (0, 80))
        result, _ = run_pipeline(image, recipe(("gradient", {}), ("threshold", {}), ("contours", {})), False)
        self.assertEqual([step["kind"] for step in result["steps"]], ["image", "mask", "objects"])
        with self.assertRaises(ValueError): validate(recipe(("gradient", {}), ("contours", {})))

    def test_invert_preserves_each_input_type(self):
        for prefix, kind in (([], "image"), ([("threshold", {})], "mask"), ([("canny", {})], "edge")):
            before, image = run_pipeline(self.img, recipe(*prefix), False)
            result, inverted = run_pipeline(self.img, recipe(*prefix, ("invert", {})), False)
            np.testing.assert_array_equal(inverted, 255-image)
            self.assertEqual(result["steps"][-1]["kind"], kind)
        with self.assertRaises(ValueError): validate(recipe(("threshold", {}), ("contours", {}), ("invert", {})))

    def test_holes_use_pixel_area_and_exclude_border_background(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[2:18, 2:28] = 255
        mask[5:8, 5:8] = 0
        mask[10:15, 10:15] = 0
        result, filled = run_pipeline(mask, recipe(("threshold", {}), ("fill_holes", dict(max_area=10))), False)
        self.assertTrue(np.all(filled[5:8, 5:8] == 255))
        self.assertTrue(np.all(filled[10:15, 10:15] == 0))
        self.assertTrue(np.all(filled[0] == 0))
        self.assertEqual(result["steps"][-1]["computed"], dict(filled_holes=1, filled_pixels=9))
        _, filled = run_pipeline(mask, recipe(("threshold", {}), ("fill_holes", {})), False)
        self.assertTrue(np.all(filled[2:18, 2:28] == 255))
        for value in (0, 255):
            flat = np.full((5, 6), value, np.uint8)
            _, actual = run_pipeline(flat, recipe(("threshold", {}), ("fill_holes", {})), False)
            np.testing.assert_array_equal(actual, flat)

    def test_small_components_connectivity_and_area_boundary(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2, 2] = mask[3, 3] = 255
        mask[7:9, 7:9] = 255
        r = lambda connectivity: recipe(("threshold", {}), ("remove_small", dict(min_area=2, connectivity=connectivity)))
        result, four = run_pipeline(mask, r(4), False)
        self.assertEqual(result["steps"][-1]["computed"], dict(removed_objects=2, removed_pixels=2))
        self.assertEqual(int(np.count_nonzero(four)), 4)
        _, eight = run_pipeline(mask, r(8), False)
        np.testing.assert_array_equal(eight, mask)
        _, minimum = run_pipeline(mask, recipe(("threshold", {}), ("remove_small", dict(min_area=4))), False)
        np.testing.assert_array_equal(minimum, four)

    def test_background_neutral_flat_and_zero_strength(self):
        for method in ("gaussian", "opening"):
            for mode in ("subtract", "divide"):
                params = dict(method=method, mode=mode, kernel=7)
                for value in (0, 100, 255):
                    image = np.full((25, 25), value, np.uint8)
                    _, actual = run_pipeline(image, recipe(("background", params)), False)
                    np.testing.assert_array_equal(actual, image)
                _, actual = run_pipeline(self.img, recipe(("background", dict(**params, strength=0))), False)
                np.testing.assert_array_equal(actual, self.img)

    def test_background_reduces_smooth_lighting_variation(self):
        image = np.tile(np.linspace(50, 200, 101).astype(np.uint8), (61, 1))
        _, corrected = run_pipeline(image, recipe(("background", dict(kernel=31))), False)
        self.assertLess(float(corrected[:, 20:80].std()), float(image[:, 20:80].std())/10)

    def test_new_operation_validation_and_disabled_round_trip(self):
        for key, params in (("mean", dict(width=4)), ("tophat", dict(kernel=2)), ("unsharp", dict(amount=-1)), ("background", dict(strength=2)), ("remove_small", dict(connectivity=6)), ("fill_holes", dict(max_area=-1))):
            with self.subTest(key=key):
                with self.assertRaises(ValueError): validate(recipe((key, params)))
        with self.assertRaises(ValueError): validate(recipe(("fill_holes", {})))
        with self.assertRaises(ValueError): validate(recipe(("canny", {}), ("remove_small", {})))
        config = recipe(("brightness_contrast", dict(brightness=12)), ("mean", {}), ("threshold", {}), ("fill_holes", {}))
        config["nodes"][0]["enabled"] = False
        clean = validate(json.loads(json.dumps(config)))
        self.assertFalse(clean["nodes"][0]["enabled"])
        self.assertEqual(clean["nodes"][0]["params"]["brightness"], 12)

    def test_catalog_explanations_and_parameters(self):
        self.assertGreaterEqual(len(CATALOG), 50)
        for spec in CATALOG.values():
            with self.subTest(operation=spec["id"]):
                self.assertTrue(spec["help"].strip())
                self.assertEqual(set(spec), {"id", "label", "group", "params", "inputs", "output", "help"})
                for parameter in spec["params"].values():
                    self.assertIn("help", parameter)
                    self.assertIn("default", parameter)

    def test_custom_kernel_and_threshold_modes_match_opencv(self):
        image = np.arange(63, dtype=np.uint8).reshape(7, 9)*4
        _, identity = run_pipeline(image, recipe(("custom_filter", {})), False)
        np.testing.assert_array_equal(identity, image)
        params = {f"k{i}": 1 for i in range(9)}
        params.update(divisor=9, delta=-10)
        _, actual = run_pipeline(image, recipe(("custom_filter", params)), False)
        np.testing.assert_array_equal(actual, cv2.filter2D(image, -1, np.ones((3,3), np.float32)/9, delta=-10))
        for mode, flag in (("truncate", cv2.THRESH_TRUNC), ("to_zero", cv2.THRESH_TOZERO), ("to_zero_inv", cv2.THRESH_TOZERO_INV)):
            _, actual = run_pipeline(image, recipe(("threshold_modes", dict(mode=mode, value=100))), False)
            _, expected = cv2.threshold(image, 100, 255, flag)
            np.testing.assert_array_equal(actual, expected)

    def test_border_area_and_convex_mask_processing(self):
        mask = np.zeros((40,40), np.uint8)
        mask[0:10,5:10] = 255
        mask[15:25,15:25] = 255
        mask[30:32,30:32] = 255
        _, actual = run_pipeline(mask, recipe(("threshold", {}), ("clear_border", {})), False)
        self.assertEqual(np.count_nonzero(actual), 104)
        _, actual = run_pipeline(mask, recipe(("threshold", {}), ("filter_area", dict(min_area=50, max_area=60))), False)
        self.assertEqual(np.count_nonzero(actual), 50)
        mask = np.zeros((30,30), np.uint8)
        mask[5:25,5:9] = mask[21:25,5:25] = 255
        _, hull = run_pipeline(mask, recipe(("threshold", {}), ("convex_hull", {})), False)
        self.assertGreater(np.count_nonzero(hull), np.count_nonzero(mask))
        self.assertTrue(np.all(hull[mask != 0] == 255))

    def test_detectors_have_original_table_schema_and_roi_coordinates(self):
        canvas = np.zeros((160,200), np.uint8)
        cv2.rectangle(canvas, (60,40), (120,100), 255, -1)
        for detector in ("corners", "hough_lines"):
            nodes = [("roi", dict(x=20,y=10,width=160,height=140))]
            if detector == "hough_lines": nodes += [("canny", {})]
            nodes += [(detector, {})]
            result, preview = run_pipeline(canvas, recipe(*nodes), False)
            self.assertGreater(len(result["objects"]), 0)
            self.assertEqual(preview.shape, (140,160,3))
            for obj in result["objects"]:
                self.assertGreaterEqual(obj["cx"], 59)
                self.assertGreaterEqual(obj["cy"], 39)
                self.assertEqual(obj["area"], 0)
                self.assertTrue({"id", "width", "height", "perimeter", "orientation", "circularity", "solidity", "aspect_ratio"} <= set(obj))
        canvas[:] = 0
        cv2.circle(canvas, (100,80), 30, 255, 2)
        result, _ = run_pipeline(canvas, recipe(("hough_circles", dict(votes=15, min_radius=25, max_radius=35))), False)
        self.assertGreater(len(result["objects"]), 0)
        self.assertAlmostEqual(result["objects"][0]["area"], np.pi*result["objects"][0]["radius"]**2)

    def test_template_repetition_flat_and_bounds(self):
        rng = np.random.default_rng(10)
        pattern = rng.integers(20,235,(12,14), dtype=np.uint8)
        image = np.zeros((80,100), np.uint8)
        image[10:22,10:24] = pattern
        image[40:52,60:74] = pattern
        for method in ("ccoeff", "ccorr", "sqdiff"):
            result, _ = run_pipeline(image, recipe(("template_match", dict(x=10,y=10,width=14,height=12,method=method,score=.99,min_distance=12))), False)
            self.assertEqual(len(result["objects"]), 1)
            obj = result["objects"][0]
            self.assertEqual((obj["x"], obj["y"], obj["width"], obj["height"]), (60,40,14,12))
            self.assertGreater(obj["score"], .99)
        result, _ = run_pipeline(np.zeros_like(image), recipe(("template_match", {})), False)
        self.assertEqual(result["objects"], [])
        self.assertIn("warning", result["steps"][0]["computed"])
        with self.assertRaises(ValueError): run_pipeline(image, recipe(("template_match", dict(x=99))), False)

    def test_extended_validation_and_empty_inputs(self):
        for key, params in (("nlm", dict(template=15, search=3)), ("dog", dict(sigma_small=3,sigma_large=1)), ("hough_circles", dict(min_radius=50,max_radius=10)), ("custom_filter", dict(divisor=0)), ("filter_area", dict(min_area=100,max_area=10))):
            with self.subTest(operation=key):
                prefix = [("threshold", {})] if key == "filter_area" else []
                with self.assertRaises(ValueError): validate(recipe(*prefix, (key, params)))
        for key in ("corners", "orb", "akaze", "hough_circles", "hough_lines", "watershed", "distance", "edge_regions", "convex_hull"):
            spec = CATALOG[key]
            prefix = [] if "image" in spec["inputs"] else [("canny" if spec["inputs"] == ["edge"] else "threshold", {})]
            result, actual = run_pipeline(np.zeros((120,180), np.uint8), recipe(*prefix, (key, {})), False)
            self.assertTrue(np.isfinite(actual).all())
            self.assertEqual(result["objects"], [])

    def test_watershed_splits_touching_objects_and_preserves_seedless(self):
        mask = np.zeros((100,140), np.uint8)
        cv2.circle(mask, (45,50), 25, 255, -1)
        cv2.circle(mask, (85,50), 25, 255, -1)
        result, split = run_pipeline(mask, recipe(("threshold", {}), ("watershed", dict(seed_ratio=.8)), ("blobs", dict(min_area=50))), False)
        self.assertEqual(len(result["objects"]), 2)
        self.assertTrue(np.isfinite(split).all())
        _, unchanged = run_pipeline(mask, recipe(("threshold", {}), ("watershed", dict(min_seed=100000))), False)
        np.testing.assert_array_equal(unchanged, mask)

    def test_orb_and_akaze_detect_features_and_limit_results(self):
        rng = np.random.default_rng(55)
        image = cv2.GaussianBlur(rng.integers(0,256,(220,280),dtype=np.uint8), (3,3), 0)
        for detector, params in (("orb", dict(max_objects=15,fast_threshold=5,edge=8)), ("akaze", dict(max_objects=15,threshold=.0001))):
            result, _ = run_pipeline(image, recipe(("roi", dict(x=10,y=20,width=250,height=190)), (detector, params)), False)
            self.assertGreater(len(result["objects"]), 0)
            self.assertLessEqual(len(result["objects"]), 15)
            for obj in result["objects"]:
                self.assertGreaterEqual(obj["cx"], 10)
                self.assertGreaterEqual(obj["cy"], 20)
                self.assertGreater(obj["area"], 0)
                self.assertTrue(np.isfinite(obj["response"]))


class APITests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app)

    def test_extended_standalone_exports_match_live_engine(self):
        image = np.zeros((160,200), np.uint8)
        cv2.rectangle(image, (40,30), (100,95), 220, -1)
        cv2.circle(image, (135,80), 20, 180, 2)
        _, encoded = cv2.imencode('.png', image)
        configs = [
            recipe(("custom_filter", dict(k4=2,k1=-.25,k3=-.25,k5=-.25,k7=-.25)), ("nlm", {}), ("gabor", dict(theta=45)), ("scharr", {}), ("triangle", {}), ("filter_area", dict(min_area=5)), ("watershed", {}), ("blobs", dict(min_area=1))),
            recipe(("roi", dict(x=10,y=10,width=180,height=140)), ("canny", {}), ("hough_lines", dict(votes=10)), ("rule", dict(min_area=0))),
            recipe(("template_match", dict(x=35,y=25,width=20,height=20,exclude_source="no",max_objects=5))),
        ]
        for config in configs:
            with self.subTest(operation=config["nodes"][-1]["type"]):
                response = self.client.post('/api/run', json=dict(image=base64.b64encode(encoded).decode(), recipe=config))
                self.assertEqual(response.status_code, 200, response.text[:300])
                live = response.json()
                script = self.client.post('/api/export/python', json=config)
                self.assertEqual(script.status_code, 200)
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root/'pipeline.py').write_text(script.text, encoding='utf-8')
                    (root/'source.png').write_bytes(encoded.tobytes())
                    proc = subprocess.run([sys.executable,'-B',str(root/'pipeline.py'),str(root/'source.png'),'--output',str(root/'out')], capture_output=True,text=True)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    exported = json.loads((root/'out/result.json').read_text(encoding='utf-8'))
                    self.assertEqual(live['objects'], exported['objects'])
                    self.assertEqual(live['recipe'], exported['recipe'])
                    self.assertEqual([s['computed'] for s in live['steps']], [s['computed'] for s in exported['steps']])
                    _, expected = run_pipeline(image, config, False)
                    np.testing.assert_array_equal(expected, cv2.imread(str(root/'out/result.png'), cv2.IMREAD_UNCHANGED))

    def test_api_and_standalone_python_equivalence(self):
        source=Path('test_image/0.tiff').read_bytes()
        config=recipe(("roi",dict(x=5,y=5,width=200,height=200)),("gaussian",{}),("threshold",{}),("morphology",{}),("contours",{}))
        res=self.client.post('/api/run',json=dict(image=base64.b64encode(source).decode(),recipe=config))
        self.assertEqual(res.status_code,200,res.text[:500])
        live=res.json()
        script=self.client.post('/api/export/python',json=config)
        self.assertEqual(script.status_code,200)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'pipeline.py').write_text(script.text,encoding='utf-8');(root/'image.tiff').write_bytes(source)
            proc=subprocess.run([sys.executable,str(root/'pipeline.py'),str(root/'image.tiff'),'--output',str(root/'out')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            exported=json.loads((root/'out/result.json').read_text(encoding='utf-8'))
            self.assertEqual(live['objects'],exported['objects'])
            self.assertEqual(live['recipe'],exported['recipe'])
            _,expected=run_pipeline(decode_image(source),config,False)
            np.testing.assert_array_equal(expected,cv2.imdecode(np.frombuffer((root/'out/result.png').read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED))

    def test_samples(self):
        for name in self.client.get('/api/samples').json():
            data=self.client.get('/api/samples/'+name).content
            r=self.client.post('/api/run',json=dict(image=base64.b64encode(data).decode(),recipe=recipe(("gaussian",{}),("threshold",{}),("contours",{}))))
            self.assertEqual(r.status_code,200,r.text[:300])

    def test_exports_and_errors(self):
        config=recipe(("gaussian",{}))
        self.assertEqual(self.client.post('/api/validate',json=config).json()['nodes'][0]['params'],dict(kernel=5,sigma=0))
        res=self.client.post('/api/export/csv',json=config)
        self.assertIn('sigma,0',res.text)
        self.assertEqual(self.client.post('/api/run',json={}).status_code,422)
        self.assertEqual(self.client.post('/api/run',content='[]').status_code,400)
        self.assertEqual(self.client.post('/api/run',json={},headers={'Origin':'https://example.com'}).status_code,403)

    def test_new_pipeline_export_and_sample_equivalence(self):
        config = recipe(("roi", dict(x=405, y=99, width=240, height=240)),
                        ("brightness_contrast", dict(brightness=5)), ("mean", {}),
                        ("unsharp", {}), ("background", dict(kernel=21)),
                        ("tophat", dict(kernel=11)), ("blackhat", dict(kernel=7)),
                        ("gradient", {}), ("threshold", dict(value=10)), ("invert", {}),
                        ("fill_holes", dict(max_area=100)), ("remove_small", dict(min_area=5)),
                        ("blobs", dict(min_area=5)), ("rule", dict(min_area=5)))
        script = self.client.post('/api/export/python', json=config)
        self.assertEqual(script.status_code, 200)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'pipeline.py').write_text(script.text, encoding='utf-8')
            for sample in sorted(Path('test_image').glob('*.tiff')):
                with self.subTest(sample=sample.name):
                    source = sample.read_bytes()
                    response = self.client.post('/api/run', json=dict(image=base64.b64encode(source).decode(), recipe=config))
                    self.assertEqual(response.status_code, 200, response.text[:300])
                    live = response.json()
                    proc = subprocess.run([sys.executable, '-B', str(root/'pipeline.py'), str(sample.resolve()), '--output', str(root/'out')], capture_output=True, text=True)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    exported = json.loads((root/'out/result.json').read_text(encoding='utf-8'))
                    self.assertEqual(live['objects'], exported['objects'])
                    self.assertEqual(live['recipe'], exported['recipe'])
                    self.assertEqual([s['computed'] for s in live['steps']], [s['computed'] for s in exported['steps']])
                    _, expected = run_pipeline(decode_image(source), config, False)
                    np.testing.assert_array_equal(expected, cv2.imdecode(np.frombuffer((root/'out/result.png').read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED))
        csv = self.client.post('/api/export/csv', json=config)
        self.assertEqual(csv.status_code, 200)
        self.assertIn('background', csv.text)
        self.assertIn('fill_holes', csv.text)


if __name__=='__main__':unittest.main()
