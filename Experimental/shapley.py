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
