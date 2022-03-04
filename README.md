# factorySim
Gym Environment to simulate the Layout Problem as a Markov Decision Process to be solved by Reinforcement Learning

## Running instructions
Use Docker host with Nvidia drivers installed.
Clone repository to Docker host.
```sh
git clone https://github.com/hendrikunger/factorySim.git
cd factorySim
```
Build the Docker image using
```sh
docker build -t factorysim
```
Run image with appropriate command e.g.
```sh
docker run --rm -it --gpus all --shm-size=12gb factorysim:latest
```
 - shm-size needs to be greater than 30% of RAM of Docker host

All files from github repository are located in the default location /home/ray/factorySim. Training scripts can be run from this location as well.

## Developing instructions
Clone Repository to your local machine or use Docker container from above
Navigate to the factorySim/env directory
```sh
git clone https://github.com/hendrikunger/factorySim.git
cd factorySim
```
If you are not using docker you need to install dependecies using:
```sh
apt-get update
apt-get install build-essential ibcairo2-dev pkg-config python3-dev
pip install -r requirements_factorySim.txt
```
IFC Open Shell is no in the index and needs to be installed manually.
Download appropriate Version from http://ifcopenshell.org/python and unpack into site packages directory of your Python installation.
e.g.
```sh
wget https://s3.amazonaws.com/ifcopenshell-builds/ifcopenshell-python-37-v0.6.0-517b819-linux64.zip
unzip -q ifcopenshell-python-37-v0.6.0-517b819-linux64.zip -d $HOME/anaconda3/lib/python3.7/site-packages
```
Navigate to the factorySim/env directory
```sh
cd env
```
Build a local package of factorySim using
```sh
python -m pip install -e .
```


