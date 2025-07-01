# IID4constancy
Code from the study "Deep neural networks trained for estimating albedo and illumination solve lightness constancy differently than human observers."

INTRO

The directory is organized in three sub-directories: The directory "renderings" containing the code we used to generate our datasets; The directory "training" with the code we used to train our models; One directory "Analysis" that includes the code used for the analysis and figures incorporated in the paper.

RENDERING OF THE DATASET (requires to have blender 2.9 installed)

MODEL TRAINING AND TESTING

Note: The LiNet directory essentially consists in code written by Zhengqin Li and adapted (simplified) to our purpose. The original code can be found here: https://github.com/lzqsd/InverseRenderingOfIndoorScene. Our U-net architecture is derived from this code.

First, you will need python 3.11 and to install the dependencies found in "requirement.txt". 
Note that the line for torch is commented, because when using pip3, torch + cu118 won't get installed automatically. You will need to do it manually with: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Training

Reference:
@inproceedings{li2020inverse,
title={Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image},
author={Li, Zhengqin and Shafiei, Mohammad and Ramamoorthi, Ravi and Sunkavalli, Kalyan and Chandraker, Manmohan},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2475--2484},
year={2020}
}
