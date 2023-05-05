#!/bin/bash
#Build singularity image
sudo singularity build factorysim.sif Singularity.def
singularity sign factorysim.sif
#singularity push factorysim.sif library://hendrik_unger/FactorySim/factorysim:latest