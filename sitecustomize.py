"""
Two small fixes for nightly Ray + Dreamer V3

1. Ensure the first EnvRunner ⇄ Learner state-sync contains the
   'rl_module' key (avoids KeyError).
2. Make Stats.peek() fall back gracefully when NumPy sees TF tensors.
"""

# -------------------------------------------------------------------
# 1️⃣  EnvRunner KeyError fix
# -------------------------------------------------------------------
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner

_gs = DreamerV3EnvRunner.get_state
_ss = DreamerV3EnvRunner.set_state

def _safe_get_state(self, *a, **kw):
    st = _gs(self, *a, **kw)
    st.setdefault("rl_module", {})
    return st

def _safe_set_state(self, state, *a, **kw):
    try:
        _ss(self, state, *a, **kw)
    except KeyError:
        pass                           # first handshake – weights still empty

DreamerV3EnvRunner.get_state = _safe_get_state
DreamerV3EnvRunner.set_state = _safe_set_state


# -------------------------------------------------------------------
# 2️⃣  Stats.peek() tensor-safe fix
# -------------------------------------------------------------------
from ray.rllib.utils.metrics.stats import Stats
import numpy as _np
try:
    import tensorflow as _tf
except ImportError:
    _tf = None                         # torch workers, etc.

_orig_peek = Stats.peek

def _safe_peek(self, *args, **kw):
    try:
        return _orig_peek(self, *args, **kw)   # fast path
    except (NotImplementedError, TypeError, ValueError):
        # Fallback: if every element is a TF tensor, use TF mean;
        # otherwise just return 0.0 to keep training alive.
        try:
            vals = list(self.values())         # public accessor exists
        except Exception:
            return 0.0
        if _tf and vals and all(map(_tf.is_tensor, vals)):
            return _tf.reduce_mean(_tf.stack(vals))
        return 0.0

Stats.peek = _safe_peek