#!/bin/bash
# create a new conda environment and install dependencies
ENV_NAME="MMAPIS"
# 1. Create a new conda environment
# if conda env list | grep -q "^$ENV_NAME\s"; then
#     echo "Environment '$ENV_NAME' already exists."
#     read -p "Do you want to recreate it? (y/n) " REPLY
#     if [[ $REPLY =~ ^[Yy]$ ]]; then
#         echo "Recreating environment '$ENV_NAME'..."
#         conda env remove -n $ENV_NAME
#         conda create -n $ENV_NAME python=3.10 -y
#     else
#         echo "Skipping environment creation."
#         exit 0
#     fi
# else
#     conda create -n $ENV_NAME python=3.10 -y
# fi

# # 2. Activate the conda environment
# conda activate $ENV_NAME

# 3. Install dependencies
pip install -r requirements.txt
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Display environment activation success information
echo "Virtual environment '$ENV_NAME' created and activated successfully."