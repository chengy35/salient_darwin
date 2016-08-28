#!/bin/bash
make
cd improved_trajectory_release
make
cp ./debug/DenseTrackStab ../debug/
cd ..
./debug/genDescriptor