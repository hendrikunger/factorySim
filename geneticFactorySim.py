import os
import multiprocessing
import queue
import yaml
import numpy as np
from env.factorySim.factorySimEnv import FactorySimEnv
from deap import base, creator, tools
from deap.tools.support import HallOfFame


NUMGERATIONS = 300
NUMTASKS = 12
NUMPOP = 300

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.4

class Worker:
    def __init__(self, env_config):
        self.env = FactorySimEnv( env_config = env_config)
        self.env.reset()
       

    def process_action(self, action, render=False):
        #print(action)
        for index, (x, y, r) in enumerate(zip(action[::3], action[1::3], action[2::3])):
            self.env.factory.update(index, xPosition=x, yPosition=y, rotation=r)

        self.env.tryEvaluate()
        if render:
            self.env.render_mode = "human"
            self.env.render()
            self.env.render_mode = "rgb_array"
        return  self.env.currentMappedReward, self.env.info

def worker_main(task_queue, result_queue, env_name):
    worker = Worker(env_name)
    while True:
        try:
            task = task_queue.get(timeout=3)  # Adjust timeout as needed
            if task is None:
                break
            result = worker.process_action(task[1], task[2])
            result_queue.put((task[0], result))
        except queue.Empty:
            continue#

def print_list(list, title="List"):
    print(f"---{title}---")
    for i in list:
        print(i)

def main():

    rng = np.random.default_rng(42)


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator 
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    toolbox.register("attr_float", rng.uniform, -1, 1)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_float, 3*5)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # register the crossover operator
    toolbox.register("mate", tools.cxUniform, indpb=0.33)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.1, low=-1.0, up=1.0, indpb=0.33)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    hall = HallOfFame(10)

    # create an initial population of 300 individuals 
    pop = toolbox.population(n=NUMPOP)


    

    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    ifcpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation", "1", "2.ifc")
    f_config['evaluation_config']["env_config"]["inputfile"] = ifcpath

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Create worker processes
    workers = []
    for i in range(NUMTASKS):
        config = f_config['evaluation_config']["env_config"].copy()
        config["prefix"] = str(i)+"_"
        #config["randomSeed"] = f_config['evaluation_config']["env_config"]["randomSeed"] + i
        print(config)
        p = multiprocessing.Process(target=worker_main, args=(task_queue, result_queue, config))
        p.start()
        workers.append(p)


    print("Start of evolution")



# --- EVOLUTION ---

    # Evaluate the entire population

    # Enqueue initial tasks (e.g., "reset" command or actions)
    num_tasks = len(pop) 
    for index, individual in enumerate(pop):
        task_queue.put((index,individual,False)) 

    # Collect results
    for _ in range(num_tasks):
        output = result_queue.get()
        pop[output[0]].fitness.values = (output[1][0],)

    hall.update(pop)
    
    print("  Evaluated %i individuals" % len(pop))
    print("  Best fitness is ", hall[0].fitness.values)

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 1 and g < NUMGERATIONS:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if rng.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if rng.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        num_tasks = len(invalid_ind) 
        for index, individual in enumerate(invalid_ind):
            task_queue.put((index,individual,False)) 
        
        # Collect results
        for _ in range(num_tasks):
            output = result_queue.get()
            invalid_ind[output[0]].fitness.values = (output[1][0],)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        #Update hall of fame
        hall.update(pop)

        print("  Evaluated %i individuals" % len(pop))
        print("  Best fitness is ", hall[0].fitness.values)


    print("-- End of (successful) evolution --")

    result = {}

    print("Hall of fame:")
    for i ,ind in enumerate(hall):
        print(f"{i+1} - {ind.fitness.values} - {ind}")
        result[i] = {"fitness": ind.fitness.values, "individual": ind}
        task_queue.put((i,ind,True)) 

    while not result_queue.empty():
        result_queue.get()





# --- Result Processing ---
    import json
    def convert(o):
        if isinstance(o, np.generic): return o.item()  
        raise TypeError
    with open('result.json', 'w') as fp:
        json.dump(result, fp, default=convert, indent=4, sort_keys=True)

    # Signal workers to exit
    for _ in range(NUMTASKS):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for p in workers:
        p.join()


if __name__ == '__main__':
    main()
