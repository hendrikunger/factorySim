from setuptools import setup, find_packages
 
setup(name='local_factorySim',
      version='0.2.dev0',
      packages=find_packages(),
      install_requires=['gymnasium', 'pycairo', 'pandas', 'shapely', 'networkx', 'scipy', 'ifcopenshell']
      
)








