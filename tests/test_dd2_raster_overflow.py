"""Regression checks on the actual DD2 rasterizer, without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import warnings
import cv2
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
SKELETON=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

def rasterizer():
    tree=ast.parse(PATH.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate_canvas_box')
    ns={'np':np,'cv2':cv2,'HEIGHT':200,'WIDTH':200}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(PATH),'exec'),ns)
    return lambda corners:ns['generate_canvas_box'](SimpleNamespace(box_normal=True,box_skeleton=SKELETON),np.array([corners]),[11],thickness=2)

@pytest.mark.parametrize('lo,hi',[(0,50000),(-2000000000,2000000000)])
def test_large_projected_face_keeps_front_edge_over_back_fill(lo,hi):
    # The first face is much larger than the 100 x 100 back face. Its area
    # must remain positive; wrapped int32 used to reverse the drawing order.
    corners=[[lo,lo],[0,40],[hi,hi],[40,0],[10,10],[110,10],[110,110],[10,110]]
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        canvas=rasterizer()(corners)
    assert canvas[11,10,10]==1.0
    assert canvas[11,50,50]==0.8
    assert not np.any(canvas[:11]) and not np.any(canvas[12:])

@pytest.mark.parametrize('extent,expected',[(50,0.8),(150,1.0)])
def test_regular_face_order_is_preserved(extent,expected):
    corners=[[0,0],[extent,0],[extent,extent],[0,extent],[10,10],[110,10],[110,110],[10,110]]
    canvas=rasterizer()(corners)
    assert canvas[11,10,10]==expected
