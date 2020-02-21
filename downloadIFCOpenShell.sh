#!/bin/sh

cd `python -m site --user-site`   && \
curl -sS https://s3.amazonaws.com/ifcopenshell-builds/ifcopenshell-python-37-v0.6.0-e44221c-linux64.zip > file.zip && \
unzip file.zip                                  && \
rm file.zip