#!/bin/bash
apptainer instance stop ray-head || true
apptainer instance stop ray-worker01 || true
