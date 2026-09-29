"""SHA-256 commit-reveal for DR baselines (paper Eq. commit, Appendix B)."""
import hashlib
import json
import os
import time
import numpy as np


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def commit(event_id, t_commit, window, baseline, model_bytes, feature_bytes, nonce=None):
    rec = {"event_id": event_id, "t_commit": t_commit, "window": [int(w) for w in window],
           "baseline_mw": [round(float(x), 4) for x in baseline],
           "model_sha256": sha(model_bytes), "features_sha256": sha(feature_bytes),
           "nonce": nonce or os.urandom(32).hex()}
    return sha(canonical(rec)), rec


def verify(c, rec, model_bytes, feature_bytes, recompute=None, atol=1e-3):
    ok = (sha(canonical(rec)) == c and rec["model_sha256"] == sha(model_bytes)
          and rec["features_sha256"] == sha(feature_bytes))
    ok = ok and len(rec["baseline_mw"]) == len(rec["window"])
    if ok and recompute is not None:
        b, ref = np.asarray(recompute(), float), np.asarray(rec["baseline_mw"], float)
        ok = b.shape == ref.shape and bool(np.allclose(b, ref, atol=atol))
    return ok


def benchmark(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    model, feats = rng.random(26).tobytes(), rng.random(24 * 60).tobytes()
    base = rng.random(4) * 100
    t0 = time.perf_counter()
    for i in range(n):
        c, rec = commit(f"ev{i}", "2025-07-01T12:00:00+04:00", range(19, 23), base, model, feats)
        assert verify(c, rec, model, feats)
    micros = (time.perf_counter() - t0) / n * 1e6
    tampered = dict(rec, baseline_mw=[x * 1.1 for x in rec["baseline_mw"]])
    assert not verify(c, tampered, model, feats)
    return micros, len(canonical(rec))
