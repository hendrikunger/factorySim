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