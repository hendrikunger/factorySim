#%%
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString

#%%
line1 = LineString([(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
line2 = LineString([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
line1.intersects(line2)

#%%
output = shapely.intersection(line1, line2)
print(output)
# %%
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import numpy as np
logger = MetricsLogger()

# Log n simple float values under the "loss" key. By default, all logged values
# under that key are averaged over once `reduce()` is called.
logger.log_value("Klaus", 1, window=13, reduce="mean")
logger.log_value("Klaus", 2)
logger.log_value("Klaus", 3)
# Peek at the current (reduced) value of "Klaus":
logger.peek("Klaus")  # 0.002

# %%
logger = MetricsLogger()
ema = 0.01

# Log some (EMA reduced) values.
key = ("some", "nested", "key", "sequence")
logger.log_value(key, 2.0, window=10, reduce="mean")
logger.log_value(key, 3.0)

# Expected reduced value:
expected_reduced = (1.0 - ema) * 2.0 + ema * 3.0

print(f"Expected reduced value: {expected_reduced}, peek: {logger.peek(key)}")

# %%
np.ones((10,2,3))

# %%
import numpy as np
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
logger = MetricsLogger()




logger.log_value("loss", np.float64(0.001), reduce=None, clear_on_reduce=False)
logger.log_value("loss", np.float64(0.002), reduce=None, clear_on_reduce=False)
logger.peek("loss")

# %%
dd= {"test": np.float64(0.001), "accuracy": np.float64(0.9)}
dd
#%%

logger.log_dict(dd, reduce=None, clear_on_reduce=False)
logger.log_dict({"test": np.float64(0.005), "accuracy": np.float64(0.3)}, reduce=None, clear_on_reduce=False)

result = logger.reduce()
result["test"]
# %%
from ray.rllib.utils.metrics.stats import Stats
import numpy as np
stats = Stats(reduce=None, clear_on_reduce=True)

stats.push(np.float64(0.001))
stats.push(np.float64(0.002))
stats.push(np.float64(0.003))
stats.push(np.float64(0.004))

stats

# %%
stats.peek()
# %%
stats.reduce()
# %%
stats
# %%
