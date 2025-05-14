#!/bin/bash

LIVE=$(pwd)
ARCH=$(pwd)/../glads-borehole-calibration-zenodo/
REPO="git@github.com:timghill/glads-borehole-calibration.git"

echo $LIVE
echo $ARCH

# Clean the archive directory
rm -rf $ARCH
mkdir $ARCH

# Clone the repo
cd $ARCH
git clone $REPO .
rm -rf .git

# Copy borehole configuration
cp -v $LIVE/GL12-2A.pkl $ARCH/

# Copy ISSM outputs
cp -rv $LIVE/issm/issm/post_borehole $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/post_synthetic $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/S01_creepopen $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/S02_nomoulins $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/S03_turbulent $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/test $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/test_numerics $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/train $ARCH/issm/issm/
cp -rv $LIVE/issm/issm/train_numerics $ARCH/issm/issm/


# Copy GP models
cp -rv $LIVE/analysis/borehole/data $ARCH/analysis/borehole/data
cp -rv $LIVE/analysis/leave_one_out/data $ARCH/analysis/leave_one_out/data
cp -rv $LIVE/analysis/synthetic/data $ARCH/analysis/synthetic/data
