#!/bin/bash
#SBATCH --account=def-rfm
#SBATCH --gres=gpu:1
#SBATCH	--cpus-per-task=16
#SBATCH --mem=12G

# module load python/3.8 blender scipy-stack
source $SLURM_LIBDIR/init.sh
source $SLURM_LIBDIR/env/bin/activate

# $1 = dstfile
# $2, $3 = i1, i2 for val
# $4, $5 = i1, i2 for test
# $6 = batchfiles

export IMAGEDIR=$SLURM_TMPDIR/images
mkdir $IMAGEDIR

echo "[`date`]  extracting tar files"
mkdir $IMAGEDIR/train
for f in $6; do tar xf "$f" -C $IMAGEDIR/train; done

echo "[`date`]  moving validation files"
mkdir $IMAGEDIR/val
for (( k=$2 ; k <= $3 ; k++ ))
do
	base=$IMAGEDIR/train/`printf "image%06d" $k`
	# mv ${base}*.hdr $IMAGEDIR/val
	# (wildcard caused long delays in directories with 100Ks of files?)
	mv $base.hdr ${base}_id.hdr ${base}_nrm.hdr ${base}_ref.hdr $IMAGEDIR/val
done

echo "[`date`]  moving test files"
mkdir $IMAGEDIR/test
for (( k=$4 ; k <= $5 ; k++ ))
do
	base=$IMAGEDIR/train/`printf "image%06d" $k`
	# mv ${base}*.hdr $IMAGEDIR/test
	mv $base.hdr ${base}_id.hdr ${base}_nrm.hdr ${base}_ref.hdr $IMAGEDIR/test
done

echo "[`date`]  archiving files"
tar cf $1 -C $IMAGEDIR train val test

echo "[`date`]  finished"

