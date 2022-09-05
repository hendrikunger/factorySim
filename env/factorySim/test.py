
from itertools import combinations


p = {"a": 1, "b": 2, "c": 3}

for i in combinations(p.values(), 2):
    print(i)

print(list(p))