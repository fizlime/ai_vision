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


class APITests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app)

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


if __name__=='__main__':unittest.main()
