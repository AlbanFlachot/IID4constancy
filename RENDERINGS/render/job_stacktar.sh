#!/bin/bash
#SBATCH --account=def-rfm
#SBATCH	--cpus-per-task=4
#SBATCH --mem=12G

module load python/3.8 blender/3.6 scipy-stack

## initialize variables
IMAGEDIR=$1

rm $IMAGEDIR/train.tar
rm -r $IMAGEDIR/tmp

python stack_taring.py --imgdir $IMAGEDIR
