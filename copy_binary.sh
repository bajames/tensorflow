#!/bin/bash
eval $(docker-machine env default)
docker cp ~/assignments/notMNIST_large.tar.gz compassionate_ramanujan:/notebooks
docker cp ~/assignments/notMNIST_small.tar.gz compassionate_ramanujan:/notebooks
docker cp ~/assignments/notMNIST.pickle compassionate_ramanujan:/notebooks
