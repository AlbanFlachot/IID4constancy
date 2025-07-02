#!/usr/bin/env bash

# get absolute path of a directory or file
function abspath() {
  echo `readlink -f "$1"`
}

# add a directory ($2) to a list of directories ($1)
function adddir() {
  if [ -d "$2" ] && [[ ":${!1}:" != *":$2:"* ]]; then
      export $1="${!1}:$2"
  fi
}

# $SLURM_ROOTDIR is the absolute path of parent directory of this file
SLURM_ROOTDIR=`dirname "${BASH_SOURCE[0]}"`
export SLURM_ROOTDIR=`abspath "$SLURM_ROOTDIR"`
export SLURM_ROOTDIR=`dirname "$SLURM_ROOTDIR"`

# export SLURM_LOCALDIR=`abspath "$SLURM_ROOTDIR/local-"`

export SLURM_LIBDIR=`abspath "$SLURM_ROOTDIR/lib"`
adddir PYTHONPATH "$SLURM_LIBDIR"

module load StdEnv/2020
module load python/3.8 blender/2.92 scipy-stack


