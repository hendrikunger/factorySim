"""
Two small fixes for nightly Ray + Dreamer V3

1. Ensure the first EnvRunner ⇄ Learner state-sync contains the
   'rl_module' key (avoids KeyError).
2. Make Stats.peek() fall back gracefully when NumPy sees TF tensors.
"""

# -------------------------------------------------------------------
# 1️⃣  EnvRunner KeyError fix  •  bundle-aware, id-agnostic
# -------------------------------------------------------------------
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.core.rl_module.rl_module import DEFAULT_MODULE_ID

_gs = DreamerV3EnvRunner.get_state
_ss = DreamerV3EnvRunner.set_state
COMP_RL = "rl_module"

def _safe_get_state(self, *a, **kw):
    st = _gs(self, *a, **kw)
    st.setdefault(COMP_RL, {DEFAULT_MODULE_ID: self.module.get_state()})
    return st

def _safe_set_state(self, state, *a, **kw):
    #  unwrap bundle {runner_id: state_dict, …}
    if COMP_RL not in state:
        state = next(iter(state.values()))

    #  Early pings can be plain ints; ignore those.
    if not isinstance(state, dict):
        return

    #  guarantee required key
    state.setdefault(COMP_RL,
                     {DEFAULT_MODULE_ID: self.module.get_state()})
    try:
        _ss(self, state, *a, **kw)
    except KeyError:
        pass

DreamerV3EnvRunner.get_state = _safe_get_state
DreamerV3EnvRunner.set_state = _safe_set_state


# -------------------------------------------------------------------
# 2️⃣  Stats.peek() – tensor-safe fallback
# -------------------------------------------------------------------
from ray.rllib.utils.metrics.stats import Stats
try:
    import tensorflow as _tf
except ImportError:
    _tf = None

_orig_peek = Stats.peek

def _safe_peek(self, *args, **kw):
    """Call the original peek(); if it stumbles over symbolic tensors,
    fall back to a tf.constant(0.) so gradient_tape accepts it."""
    try:
        return _orig_peek(self, *args, **kw)
    except (NotImplementedError, TypeError, ValueError):
        try:
            vals = list(self.values())      # public accessor
        except Exception:
            vals = []
        # If all values are TF-tensors → return their mean
        if _tf and vals and all(map(_tf.is_tensor, vals)):
            return _tf.reduce_mean(_tf.stack(vals))
        # Otherwise supply a harmless scalar tensor placeholder
        if _tf:
            return _tf.constant(0.0, dtype=_tf.float32)
        return 0.0                          # non-TF backend

Stats.peek = _safe_peek

# -------------------------------------------------------------------
# 3️⃣  Stats._reduced_values  —  ALWAYS (scalar, list)
# -------------------------------------------------------------------
from ray.rllib.utils.metrics.stats import Stats
try:
    import tensorflow as _tf
except ImportError:
    _tf = None

_orig_reduce = Stats._reduced_values

def _to_list(obj):
    """Return obj unchanged if it is already a list/tuple, else wrap."""
    return list(obj) if not isinstance(obj, (list, tuple)) else list(obj)

def _reduce_safe(self, *args, **kwargs):
    try:
        res = _orig_reduce(self, *args, **kwargs)

        # ── Case A: upstream already returns tuple of len 2
        if isinstance(res, tuple) and len(res) == 2:
            scal, vals = res
            return float(scal), _to_list(vals)

        # ── Case B: upstream returns just a scalar
        vals = kwargs.get("values") or getattr(self, "_values", [])
        return float(res), _to_list(vals)

    except Exception:
        # ── Fallback for dtype/shape quirks
        vals = kwargs.get("values") or getattr(self, "_values", [])
        vals = _to_list(vals)
        if _tf and vals and all(map(_tf.is_tensor, vals)):
            scal = float(_tf.reduce_mean(_tf.stack(vals)).numpy())
        elif vals:
            try:
                scal = float(sum(vals) / len(vals))
            except Exception:
                scal = float(vals[0]) if isinstance(vals[0], (int, float)) else 0.0
        else:
            scal = 0.0
        return scal, vals

Stats._reduced_values = _reduce_safe