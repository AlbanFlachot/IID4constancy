# IID4constancy
Code from the study "Deep neural networks trained for estimating albedo and illumination solve lightness constancy differently than human observers."

## Forwords:

The directory is organized in two sub-directories: 
 - The directory "RENDERINGS" contains the code we used to generate our training and test datasets; 
 - The directory "MODELING AND ANALYSIS" that includes the code used for the modeling part as well as the analysis and figures incorporated in the paper.

We also provide links to download the:
 - testsets: [LINK](https://www.dropbox.com/scl/fo/yb360ltk2llhyzvhzu7bj/ALmepfhRmdUinTKEZ5rrRKk?rlkey=matwof0s7ew9ei9udk7l8raxb&st=0zz49pim&dl=0)
 - and the pretrained models: [LINK](https://www.dropbox.com/scl/fo/fxl7486bp8b2py5nbv3lw/AKRNDf1pVEOJu45nl0JMt7s?rlkey=ftgr9y2h7hndjw1xifo5jzh8n&st=l7paaw7s&dl=0)

 - Note that we don't provide the training sets for now. We will make it available at once the paper is accepted.

## RENDERING

Requires to have blender 2.9 installed.

Note: The datasets were rendered on a large cluster using SLURM.
For anything else, you will need to adapt the code.

### To render the training set:

First run on a terminal from the RENDERING directory "source lib/init.sh" to have the correct path and load the moduls.

Then, in the "RENDERINGS/render" directory, run: python3 jobs_render.py
Note the arguments. By default, it renders 135K images with EEVEE on 20 GPUs. The output dataset is divided into <nb_GPUS> tar archives.
Then run python3 stack_taring.py to organize the dataset into training, validation and test sets.

### To render the test sets:

Go to the render_testsets directory.
First run bash.py --> renders the normal and contrast conditions
Then run bash_conditions.py --> renders the other conditions, namely shadows, shadings, shadows and shadings, and no-cue


## MODELING & ANALYSIS

First, you will need python 3.11 and to install the dependencies found in "requirement.txt". 
Note that the line for torch is commented, because when using pip3, torch + cu118 won't get installed automatically. You will need to do it manually with: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

To train, test and run the first level of analysis, look at 'MODELING_and_ANALYSIS/bash.py' script.
For the last level of evaluation and most figures, look at the jupyter notebooks.

Note: The LiNet directory essentially consists in code written by Zhengqin Li and adapted (simplified) for our purpose. The original code can be found here: https://github.com/lzqsd/InverseRenderingOfIndoorScene. Our U-net architecture is derived from this code.


