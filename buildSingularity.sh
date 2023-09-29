#!/bin/bash
#Build singularity image
sudo apptainer build factorysim.sif Singularity.def
apptainer sign factorysim.sif
#singularity push factorysim.sif library://hendrik_unger/FactorySim/factorysim:latest