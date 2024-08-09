import os
import multiprocessing
import queue
import yaml
import argparse
import numpy as np
from env.factorySim.factorySimEnv import FactorySimEnv
from deap import base, creator, tools
from deap.tools.support import HallOfFame
from tqdm import tqdm
import random
from supabase import create_client, Client
from pathlib import Path
from datetime import datetime
import ifcopenshell
from pprint import pp


# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual

NUMMACHINES = 5
CXPB, MUTPB = 0.5, 0.3
ETA = 0.9

parser = argparse.ArgumentParser()

parser.add_argument("--num-workers", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", 2)))  #multiprocessing.cpu_count()
parser.add_argument("--num-generations", type=int, default=5) 
parser.add_argument("--num-population", type=int, default=100) 
parser.add_argument(
    "--problemID",
    type=int,
    default=1,
    help="Which - in the list of evaluation environments to use. Default is 1.",
)

class Worker:
    def __init__(self, env_config, starting_time):
        self.env = FactorySimEnv( env_config = env_config)
        self.env.reset()
        self.starting_time = starting_time
        inputPath = env_config["inputfile"]
        self.problem_id= os.path.splitext(os.path.basename(inputPath))[0]
        self.outputPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Output", self.problem_id)
        os.makedirs(self.outputPath, exist_ok=True)
       

    def process_action(self, action, render=False, generation=None):
        #print(action)
        for index, (x, y, r) in enumerate(zip(action[::3], action[1::3], action[2::3])):
            self.env.factory.update(index, xPosition=x, yPosition=y, rotation=r)
        self.env.tryEvaluate()

        if render:
            self.env.render_mode = "human"
            if generation is None:
                self.env._render_frame()
            else:
                output = os.path.join(self.outputPath, f"{self.starting_time}___{self.problem_id}___{generation}___{self.env.currentMappedReward}")
                self.env._render_frame(output)
            self.env.render_mode = "rgb_array"
        return  self.env.currentMappedReward, self.env.info

def worker_main(task_queue, result_queue, env_name):
    starting_time = datetime.now().strftime("%Y-%m-%d___%H-%M-%S")
    worker = Worker(env_name, starting_time)
    while True:
        try:
            task = task_queue.get(timeout=3)  # Adjust timeout as needed
            if task is None:
                break
            #task[0] is the index of the individual
            #task[1] is the individual
            #task[2] is a boolean to render
            #task[3] is the generation number
            result = worker.process_action(task[1], task[2], task[3])
            result_queue.put((task[0], result))
        except queue.Empty:
            continue#





def mycxBlend(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = random.uniform(-alpha, 1. + alpha)
        
        x = (1. - gamma) * x1 + gamma * x2
        ind1[i] = min(max(x, -1), 1)
        x = gamma * x1 + (1. - gamma) * x2
        ind2[i] = min(max(x, -1), 1)
    
    return ind1, ind2

def main():

    args = parser.parse_args()
    print(f"Using {args.num_workers} workers", flush=True)
    print(f"Started with {args.num_population} individuals for maximum {args.num_generations} generations." , flush=True)

    last_best = None
    last_change_gen = 0


    with open('config.yaml', 'r') as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)
    
    rng = np.random.default_rng(f_config['evaluation_config']["env_config"]["randomSeed"])

    eval_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Evaluation"))
    evalFiles = [x for x in eval_dir.iterdir() if x.is_file() and ".ifc" in x.name]
    evalFiles.sort()
    ifcpath = evalFiles[args.problemID % len(evalFiles)-1]
    f_config['evaluation_config']["env_config"]["inputfile"] = ifcpath
    f_config['evaluation_config']["env_config"]["reward_function"] = 1

    ifc_file = ifcopenshell.open(ifcpath)
    ifc_elements = ifc_file.by_type("IFCBUILDINGELEMENTPROXY")
    NUMMACHINES = len(ifc_elements)
    print(f"Found {NUMMACHINES} machines in ifc file.\n")
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
        toolbox.attr_float, 3*NUMMACHINES)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # register the crossover operator
    #toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mate", mycxBlend, alpha=0.25) # Alpha value is recommended to 0.25

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, indpb=1/NUMMACHINES)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    hall = HallOfFame(10)

    # create an initial population of 300 individuals 
    pop = toolbox.population(n=args.num_population)



    



    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Create worker processes
    workers = []
    for i in range(args.num_workers):
        config = f_config['evaluation_config']["env_config"].copy()
        config["prefix"] = str(i)+"_"
        #config["randomSeed"] = f_config['evaluation_config']["env_config"]["randomSeed"] + i
        p = multiprocessing.Process(target=worker_main, args=(task_queue, result_queue, config))
        p.start()
        workers.append(p)

    initialSolutionPath = str(ifcpath).replace(".ifc", "_pos.json")
    if os.path.exists(initialSolutionPath):
        print(f"Found initial solution {initialSolutionPath}. Loading...", flush=True)
        import json
        with open(initialSolutionPath, 'r') as f:
            initialSolution = json.load(f)
        #append initial solution to population
        individual = []
        for gene in initialSolution["config"].values():
            individual.append(gene["position"][0])
            individual.append(gene["position"][1])
            individual.append(gene["rotation"])
        pop.append(creator.Individual(individual))        
        print(f"Added initial solution to population", flush=True)
        print(individual, flush=True)
        task_queue.put((-1,individual,True,-5))
        result = result_queue.get()
        pp(result[1][1])


    print("Start of evolution", flush=True)
    CUR_ETA = ETA



# --- EVOLUTION ---

    # Evaluate the entire population
    num_tasks = len(pop) 
    print(f"Evaluating {num_tasks} individuals", flush=True)
    for index, individual in enumerate(pop):
        task_queue.put((index,individual,False, None)) 

    # Collect results
    for _ in range(num_tasks):
        output = result_queue.get()
        pop[output[0]].fitness.values = (output[1][0],)


    hall.update(pop)
    
    
    print(f"  Best fitness is {hall[0].fitness.values}\n")

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]


    # Begin the evolution
    for g in tqdm(range(1,args.num_generations+1)):

        print(f"____ Generation {g} ___________________________________________ last change at {last_change_gen}_____________", flush=True)

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
                toolbox.mutate(mutant,eta=CUR_ETA)
                del mutant.fitness.values


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        num_tasks = len(invalid_ind) 
        for index, individual in enumerate(invalid_ind):
            task_queue.put((index,individual,False, None)) 
        
        # Collect results
        for _ in range(num_tasks):
            output = result_queue.get()
            invalid_ind[output[0]].fitness.values = (output[1][0],)


        # The population is entirely replaced by the offspring
        pop[:] = offspring
        #Update hall of fame
        hall.update(pop)
        fits = [ind.fitness.values[0] for ind in pop]

        print("  Evaluated %i individuals" % len(invalid_ind), flush=True)
        if last_best != hall[0]:
            print(f"---> Found new best individual with fitness {hall[0].fitness.values}", flush=True)
            last_best = hall[0]
            last_change_gen = g
            #Render and evaluate the best individuals again
            task_queue.put((-1,hall[0],True,g))
            result = result_queue.get()
            pp(result[1][1])


        else: 
            print(f"  Best fitness is {hall[0].fitness.values}", flush=True)
        print("\n\n")
        #Resetting crowding factor after new improvement
        if g - last_change_gen == 0 and g > 50 and CUR_ETA != ETA:
            print(f"Resetting Crowding Factor to local search: {ETA}", flush=True)
            CUR_ETA = ETA
        #Change crowding factor if no improvement for 50 generations
        if g- last_change_gen > 50 and CUR_ETA > 0.1:  
            CUR_ETA-=0.01
            print(f"No improvement for 50 generations. Decrease crowding factor for bigger search space to {CUR_ETA}", flush=True)

        if max(fits) > 0.9 or g - last_change_gen > 300:
            print(f"No improvement for 300 generations or fitness > 0.9. Stopping...", flush=True)
            break

    print("-- End of (successful) evolution --\n\n", flush=True)

    result = {}
    print("\n------------------------------------------------------------------------", flush=True)
    print("Hall of fame:", flush=True)
    print("------------------------------------------------------------------------\n", flush=True)

    for i ,ind in enumerate(hall):
        print(f"{i+1} - {ind.fitness.values} - {ind}", flush=True)
        task_queue.put((i,ind,True,f"H{i+1}"))

        data = {}

        for index, (x, y, r) in enumerate(zip(ind[::3], ind[1::3], ind[2::3])):
            data[index] = {"position": (x,y), "rotation": r}

        result[i] = {"fitness": ind.fitness.values[0],
                    "individual": ind,
                    "problem_id": os.path.splitext(os.path.basename(ifcpath))[0],
                     "creator": "Hendrik Unger",
                     "config": data
                    }

    
    print("------------------------------------------------------------------------\n\n", flush=True)

    for _ in range(len(hall)):
        output = result_queue.get()
        #pp(output[1][1])

    # Signal workers to exit
    for _ in range(args.num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for p in workers:
        p.join()


# --- Result Processing ---

    #json    
    
    print("Saving to json...")

    import json
    with open('result.json', 'w') as fp:
        json.dump(result, fp, indent=4, sort_keys=True)

    #Upload
    print("Uploading results...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    
    records = []
    for element in result.values():
        if element["fitness"] < 0.6:
            continue

        copy = element.copy()
        element["config"] = copy
        records.append(element)

    
    if len(records) == 0:
        print("No results to upload", flush=True)
    else:
        data, count = supabase.table('highscore').insert(records).execute()






if __name__ == '__main__':
    main()
