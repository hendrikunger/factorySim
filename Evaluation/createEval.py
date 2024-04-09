from factorySim.creation import FactoryCreator
import factorySim.factorySimClass as FactorySim
from tqdm.auto import tqdm
import os


factoryConfig = baseConfigs.SMALLSQUARE



if __name__ == "__main__":

    print("Creating Factory")
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    outputPath = os.path.join(basePath, "Output")
    #Create directory if it does not exist
    ifcPath = os.path.join(basePath, "..","Input", "2")
    print(ifcPath)

    factory = FactorySim(path_to_ifc_file=ifcPath,factoryConfig=factoryConfig, randomSeed=0, createMachines=True, maxMF_Elements = 5)

    for i in tqdm(range(5)):
        print(f"Creating Factory {i}")
        # factory = FactoryCreator(factoryConfig)
        # factory.saveFactory("evalFactory"+str(i))
        # factory.saveFactory("evalFactory"+str(i), "evalFactory"+str(i)+".png")
