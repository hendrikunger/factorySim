# """
# Two small fixes for nightly Ray + Dreamer V3

# 1. Ensure the first EnvRunner ⇄ Learner state-sync contains the
#    'rl_module' key (avoids KeyError).
# 2. Make Stats.peek() fall back gracefully when NumPy sees TF tensors.
# """

# # -------------------------------------------------------------------
# # 1️⃣  EnvRunner KeyError fix  •  bundle-aware, id-agnostic
# # -------------------------------------------------------------------
# from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
# from ray.rllib.core.rl_module.rl_module import DEFAULT_MODULE_ID

# _gs = DreamerV3EnvRunner.get_state
# _ss = DreamerV3EnvRunner.set_state
# COMP_RL = "rl_module"

# def _safe_get_state(self, *a, **kw):
#     st = _gs(self, *a, **kw)
#     st.setdefault(COMP_RL, {DEFAULT_MODULE_ID: self.module.get_state()})
#     return st

# def _safe_set_state(self, state, *a, **kw):
#     #  unwrap bundle {runner_id: state_dict, …}
#     if COMP_RL not in state:
#         state = next(iter(state.values()))

#     #  Early pings can be plain ints; ignore those.
#     if not isinstance(state, dict):
#         return

#     #  guarantee required key
#     state.setdefault(COMP_RL,
#                      {DEFAULT_MODULE_ID: self.module.get_state()})
#     try:
#         _ss(self, state, *a, **kw)
#     except KeyError:
#         pass

# DreamerV3EnvRunner.get_state = _safe_get_state
# DreamerV3EnvRunner.set_state = _safe_set_state


# from ray.rllib.utils.metrics.stats import Stats, force_list
# import numpy as _np
# try:
#     import tensorflow as _tf
# except ImportError:
#     _tf = None

# _orig_reduce = Stats._reduced_values

# def _safe_reduced_values(self, *args, **kwargs):
#     """
#     Return ((scalar, throughput), values_list) under *all* circumstances.
#     Never touch NumPy with TF tensors.
#     """
#     try:
#         out = _orig_reduce(self, *args, **kwargs)
#         # Nightly Ray already returns the correct 2-tuple – just forward it.
#         if (isinstance(out, tuple) and len(out) == 2
#                 and isinstance(out[0], tuple)):
#             return out
#         # It returned a flatter shape -> wrap once.
#         return (out[0] if isinstance(out, tuple) else out, None), \
#                (out[1] if isinstance(out, tuple) else
#                 list(getattr(self, "_values", [])))
#     except Exception:
#         vals = list(getattr(self, "_values", []))      # may be empty
#         if _tf and vals and all(map(_tf.is_tensor, vals)):
#             scalar = float(_tf.reduce_mean(_tf.stack(vals)))
#         elif vals:
#             try:
#                 scalar = float(_np.mean(vals))
#             except Exception:
#                 scalar = float(vals[0]) if isinstance(vals[0], (int, float)) else 0.0
#         else:
#             scalar = 0.0
#         return (scalar, None), vals

# Stats._reduced_values = _safe_reduced_values



# _orig_peek = Stats.peek
# def _peek_scalar(self, *a, **kw):
#     out = _orig_peek(self, *a, **kw)
#     # Ray core sometimes returns (scalar, throughput)
#     if isinstance(out, tuple):
#         return out[0]
#     return out          # already scalar

# Stats.peek = _peek_scalar  



# _orig_reduce_fn = Stats.reduce






# from typing import Tuple, Sequence

# def _reduce_scalar(self: Stats, *, compile: bool = True):
#     """Safe replacement that never returns tuples to the caller."""
#     scalar, values = self._reduced_values(compile=compile)
#     # _reduced_values() may give a bare scalar -> normalise
#     if isinstance(scalar, (tuple, list)):
#         # was (scalar, values) from our earlier patch
#         scalar, values = scalar                   # first item
#     # book-keeping: store both pieces internally, but…
#     self._reduce_history.append(force_list([scalar, values]))
#     # …hand only the scalar to outer RLlib code
#     return scalar

# Stats.reduce = _reduce_scalar




"""
Hot-fixes for Ray 2.48 + DreamerV3 until the next nightly.

1.  DreamerV3EnvRunner.get/set_state
    – always transfer an `rl_module` entry.

2.  Stats.*    – ALWAYS hand *scalars* to RLlib and
                 guarantee the object passed to `.copy()`
                 is itself copy-able.
"""
# ------------------------------------------------------------------ #
# 1️⃣  EnvRunner ↔ Learner handshake -------------------------------- #
# ------------------------------------------------------------------ #
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.core.rl_module.rl_module import DEFAULT_MODULE_ID

_RL_KEY = "rl_module"
_orig_get, _orig_set = (DreamerV3EnvRunner.get_state,
                        DreamerV3EnvRunner.set_state)

def _get_state_safe(self, *a, **kw):
    st = _orig_get(self, *a, **kw)
    st.setdefault(_RL_KEY, {DEFAULT_MODULE_ID: self.module.get_state()})
    return st

def _set_state_safe(self, state, *a, **kw):
    # unwrap bundle {runner_id: state_dict}
    if _RL_KEY not in state and isinstance(state, dict):
        state = next(iter(state.values()))
    if not isinstance(state, dict):
        return                                            # early ping
    state.setdefault(_RL_KEY, {DEFAULT_MODULE_ID: self.module.get_state()})
    try:
        _orig_set(self, state, *a, **kw)
    except KeyError:
        pass                                             # first contact

DreamerV3EnvRunner.get_state = _get_state_safe
DreamerV3EnvRunner.set_state = _set_state_safe

# ------------------------------------------------------------------ #
# 2️⃣  Stats helpers ------------------------------------------------ #
# ------------------------------------------------------------------ #
from ray.rllib.utils.metrics.stats import Stats, force_list

# -- a)  _reduced_values  → ((scalar, throughput), values)
_orig_rv = Stats._reduced_values
def _rv_safe(self: Stats, *a, **kw):
    try:
        val, vals = _orig_rv(self, *a, **kw)
        #  upstream may return scalar OR (scalar, throughput)
        if not isinstance(val, tuple):
            val = (val, None)
        return (val, vals)
    except Exception:
        # no stored values yet → safe defaults
        return ((0.0, None), [])
Stats._reduced_values = _rv_safe

# -- b)  peek  → only the scalar -----------------------------------
Stats.peek = lambda self, **kw: _rv_safe(self, **kw)[0][0]

# -- c)  reduce  → scalar returned, copy-safe history ---------------
def _reduce_scalar(self: Stats, *, compile: bool = True):
    (scalar, throughput), _vals = _rv_safe(self, compile=compile)
    # store *list* so .copy() is available
    self._reduce_history.append(force_list([scalar]))
    return scalar
Stats.reduce = _reduce_scalar