#!/bin/bash

conda env create -f environment.yml
conda activate code275
conda install -c conda-forge flatbuffers

sudo npm install -g jsonrepair

wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip
unzip v2.9.1

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
which bazel

cd tensorflow-2.9.1/
./configure
cd ..

cp ./files/kernel_files/* ./tensorflow-2.9.1/tensorflow/lite/kernels/
cp ./files/build_files/build.sh ./tensorflow-2.9.1/