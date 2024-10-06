#!/bin/bash
# create a new conda environment and install dependencies
ENV_NAME="MMAPIS_client"
# 1. Create a new conda environment

conda create -n $ENV_NAME python=3.11 -y

# 2. Activate the conda environment
conda activate $ENV_NAME


# 3. Install dependencies
pip install -r requirements.txt
# pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Display environment activation success information
echo "Virtual environment '$ENV_NAME' created and activated successfully."