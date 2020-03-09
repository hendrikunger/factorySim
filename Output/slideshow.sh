#!/bin/sh
ffmpeg -framerate 4/1  -pattern_type glob -i '*.png' -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4
find -type f -iname '*.png' -delete