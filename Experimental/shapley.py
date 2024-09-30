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
