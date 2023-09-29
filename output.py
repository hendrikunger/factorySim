import time

for i in range(10):
    print(f"{i} - Hello, world!")
    time.sleep(0.5)


#$(date +%F_%H%M%S)

#python -m output 2>&1 | tee  "$(date +%F_%H%M%S).log" 