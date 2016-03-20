#!/bin/bash
eval $(docker-machine env default)
docker-machine ip default
#docker cp ~/ml/notMNIST_large.tar.gz compassionate_ramanujan:/notebooks
#docker cp ~/ml/notMNIST_small.tar.gz compassionate_ramanujan:/notebooks
#docker run -p 8888:8888 -v /Users/bjames/ml/tensorflow/tensorflow/examples/udacity -it b.gcr.io/tensorflow/tensorflow
#docker run -p 8888:8888 -v /Users/bjames/ml/tensorflow/tensorflow/examples/udacity -it --rm $USER/assignments
#docker run -p 8888:8888 -v /Users/bjames/tensorflow:/ -it b.gcr.io/tensorflow/tensorflow
docker run -p 8888:8888 -v /Users/bjames/tensorflow:/notebooks -it --rm $USER/assignments
