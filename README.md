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
