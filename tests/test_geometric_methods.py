from __future__ import annotations

import numpy as np

from hd.methods.geometric.sgi import SGIMetric
from hd.methods.geometric.dc import DCMetric
from hd.methods.geometric.cai import CAIMetric


def test_sgi_formula_basic():
    # Construct vectors so that a is closer to c than to q.
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)
    a = np.array([[0.0, 1.0]], dtype=np.float32)

    m = SGIMetric(threshold=0.0)
    out = m.compute(q=q, c=c, a=a, gt=None, centroid=np.array([1.0, 0.0], dtype=np.float32))

    assert out.scores.shape == (1,)
    assert out.flags.shape == (1,)
    assert out.scores[0] > 0.0
    assert out.flags[0] is False


def test_dc_formula_basic():
    q = np.array([[0.0, 0.0]], dtype=np.float32)
    gt = np.array([[1.0, 0.0]], dtype=np.float32)
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)

    m = DCMetric(threshold=0.9)
    out = m.compute(q=q, c=c, a=a, gt=gt, centroid=np.array([0.0, 1.0], dtype=np.float32))
    assert out.scores[0] > 0.99
    assert out.flags[0] is False


def test_cai_basic():
    # a equals c => dist_a_c=0; if centroid is far => cai positive
    a = np.array([[0.0, 1.0]], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    centroid = np.array([1.0, 0.0], dtype=np.float32)

    m = CAIMetric(threshold=0.0)
    out = m.compute(q=q, c=c, a=a, gt=None, centroid=centroid)
    assert out.scores[0] > 0.0
    assert out.flags[0] is False
