#%%
from factorySim.creation import FactoryCreator
import factorySim.baseConfigs as baseConfigs
from factorySim.factorySimClass import FactorySim
from tqdm.auto import tqdm
import os
import ifcopenshell

factoryConfig = baseConfigs.SMALLSQUARE



if __name__ == "__main__":

    print("Creating Factory")
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    outputPath = os.path.join(basePath, "Output")
    #Create directory if it does not exist
    ifcPath = os.path.join(basePath, "..","Input", "2")
    print(ifcPath)

    factory = FactorySim(path_to_ifc_file=ifcPath,factoryConfig=factoryConfig, randSeed=0, createMachines=False)
    print(factory.machine_dict)


    # #save line to json
    # with open("factory.pkl", "wb") as f:
    #     pickle.dump(factory.machine_dict, f)
    

#%%

ifc_file = ifcopenshell.open(os.path.join(ifcPath, "Basic.ifc"))
# %%
