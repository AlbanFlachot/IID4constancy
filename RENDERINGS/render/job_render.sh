#!/bin/bash
#SBATCH --account=def-rfm
#SBATCH --gres=gpu:1
#SBATCH	--cpus-per-task=16
#SBATCH --mem=12G

module load python/3.8 blender/2.92 scipy-stack

## initialize variables
idstart=$1
idend=$2
IMAGEDIR=$3
scriptfile=$4
renderer=$5
counts=$6
archiv=$7


for imgid in $(seq $idstart 1 $idend)
do
	## First render
	blender --background --python-use-system-env --python $scriptfile -- --filename "$IMAGEDIR/image$imgid" --renderer $renderer --samples $counts
	## then add to corresponding archive
	python taring.py --set $archiv --scene "image$imgid" --imgdir $IMAGEDIR
done
