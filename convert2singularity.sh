#!/bin/bash
#Convert Docker Build File to Singularity
#pip3 install spython # if you do not have spython install it from the command line

spython recipe DockerfileNvidia &> Singularity.def
echo "Singularity.def created"
