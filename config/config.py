import json
from typing import List,Literal
import yaml
import os
import sys

def get_config():
    with open(os.path.join(DIR_PATH,CONFIG_PATH), 'r') as config_file:
        CONFIG = yaml.safe_load(config_file)
    return CONFIG

def get_prompts():
    prompts_path = CONFIG['openai']['prompts_path']
    with open(os.path.join(DIR_PATH,prompts_path), 'r') as prompts_file:
        PROMPTS = json.load(prompts_file)
    return PROMPTS

PROJ_DIR = "/Users/ke/workshop/code_pytorch/MMAPIS"
sys.path.append(PROJ_DIR)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LOGGER_MODES:Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
CONFIG_PATH = './config.yaml'
CONFIG = get_config()
PROMPTS = get_prompts()
SECTION_PROMPTS = PROMPTS['section_prompts']
INTEGRATE_PROMPTS = PROMPTS['integrate_prompts']
APPLICATION_PROMPTS = PROMPTS['application_prompts']
