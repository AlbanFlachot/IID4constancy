#!/usr/bin/env bash

function abspath() {
  echo `readlink -f "$1"`
}

function adddir() {
  if [ -d "$2" ] && [[ ":${!1}:" != *":$2:"* ]]; then
      export $1="${!1}:$2"
  fi
}

# $SLURM_ROOTDIR is the absolute path of the nn-intrinsic project
SLURM_ROOTDIR=`dirname "${BASH_SOURCE[0]}"`
export SLURM_ROOTDIR=`abspath "$SLURM_ROOTDIR"`
export SLURM_ROOTDIR=`dirname "$SLURM_ROOTDIR"`

# export SLURM_LOCALDIR=`abspath "$SLURM_ROOTDIR/local-"`

export SLURM_LIBDIR=`abspath "$SLURM_ROOTDIR/lib"`
adddir PYTHONPATH "$SLURM_LIBDIR"


#export PATH=/home/alban/Downloads/blender-3.6.2-linux-x64:$PATH
export PATH=/home/alban/Downloads/blender-2.92.0-linux64:$PATH
source /home/alban/env4mitsuba/bin/activate
