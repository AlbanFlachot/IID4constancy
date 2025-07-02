#!/bin/bash
#SBATCH --account=def-rfm
#SBATCH --gres=gpu:1
#SBATCH	--cpus-per-task=16
#SBATCH --mem=12G

module load python/3.8.2 blender/3.6.0 scipy-stack
#source $SLURM_LIBDIR/env/bin/activate

# export IMAGEDIR=$SLURM_TMPDIR/images
# mkdir $IMAGEDIR

# export IMAGEDIR=`mktemp -d -p .`

mkdir $SLURM_TMPDIR/images
export IMAGEDIR=`mktemp -p .`
ln -sf $SLURM_TMPDIR/images $IMAGEDIR

eval $1

python `echo "$2" | envsubst`

# rm -fr $IMAGEDIR
rm $IMAGEDIR

