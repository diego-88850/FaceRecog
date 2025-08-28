import numpy as np
from backend.index import IndexState, nearest, auth_compare

def test_nearest_and_auth():
    # user 1: two vecs near [1,0,...]; user 2: vecs near [0,1,...]
    e1a = np.zeros(512, dtype=np.float32); e1a[0] = 1
    e1b = np.zeros(512, dtype=np.float32); e1b[0] = 0.9
    e2a = np.zeros(512, dtype=np.float32); e2a[1] = 1
    M = np.vstack([e1a, e1b, e2a])
    owners = np.array([1,1,2], dtype=np.int32)
    state = IndexState(M, owners, {1: np.array([0,1]), 2: np.array([2])})
    q = np.zeros(512, dtype=np.float32); q[0] = 1.0
    uid, score = nearest(q, state)
    assert uid == 1 and score > 0.9
    score_auth = auth_compare(1, q, state)
    assert score_auth > 0.9