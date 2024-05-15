import os
import multiprocessing
import queue
import yaml
import numpy as np
from env.factorySim.factorySimEnv import FactorySimEnv

class Worker:
    def __init__(self, env_config):
        self.env = FactorySimEnv( env_config = env_config)
        self.env.reset()
       
        

    def process_action(self, action):
        #print(action)
        for index, (x, y, r) in enumerate(zip(action[::3], action[1::3], action[2::3])):
            self.env.factory.update(index, xPosition=x, yPosition=y, rotation=r)

        self.env.tryEvaluate()
        return  self.env.currentMappedReward, self.env.info

def worker_main(task_queue, result_queue, env_name):
    worker = Worker(env_name)
    while True:
        try:
            task = task_queue.get(timeout=3)  # Adjust timeout as needed
            if task is None:
                break
            result = worker.process_action(task)
            result_queue.put(result)
        except queue.Empty:
            continue

def main():

    num_workers = 24
    rng = np.random.default_rng(42)
    

    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation", "2.ifc")
    f_config['evaluation_config']["env_config"]["inputfile"] = ifcpath

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Create worker processes
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker_main, args=(task_queue, result_queue, f_config['evaluation_config']["env_config"]))
        p.start()
        workers.append(p)

    # Enqueue initial tasks (e.g., "reset" command or actions)
    num_tasks = 200  # Example number of tasks
    for _ in range(num_tasks):
        task_queue.put(rng.uniform(low=-1, high=1, size=3*5)) 



    result = {}
    # Collect results
    for i in range(num_tasks):
        result[f"{i}"] = result_queue.get()

    import json
    def convert(o):
        if isinstance(o, np.generic): return o.item()  
        raise TypeError
    with open('result.json', 'w') as fp:
        json.dump(result, fp, default=convert, indent=4, sort_keys=True)

    # Signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for p in workers:
        p.join()


if __name__ == '__main__':
    main()
