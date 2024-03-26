import json
from typing import List,Literal
import yaml
import os

def get_config():
    with open(os.path.join(DIR_PATH,CONFIG_PATH), 'r') as config_file:
        CONFIG = yaml.safe_load(config_file)
    return CONFIG

def get_prompts():
    prompts_path = OPENAI_CONFIG['prompts_path']
    with open(os.path.join(DIR_PATH,prompts_path), 'rb') as prompts_file:
        PROMPTS = json.load(prompts_file)
    return PROMPTS

DIR_PATH = os.path.dirname(os.path.abspath(__file__)) # MMAPIS/config
MMAPIS_PATH = os.path.dirname(DIR_PATH) # MMAPIS
LOGGER_MODES:Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
CONFIG_PATH = './config.yaml'
CONFIG = get_config()
GENERAL_CONFIG = CONFIG['general']
GENERAL_CONFIG["save_dir"] = os.path.join(MMAPIS_PATH, GENERAL_CONFIG["save_dir"])  # Middleware result will be saved in MMAPIS/app_res
GENERAL_CONFIG["app_save_dir"] = os.path.join(MMAPIS_PATH, GENERAL_CONFIG["app_save_dir"])  # Middleware result will be saved in MMAPIS/app_res
TEMPLATE_DIR = os.path.join(DIR_PATH, "templates")
ARXIV_CONFIG = CONFIG['arxiv']
NOUGAT_CONFIG = CONFIG['nougat']
OPENAI_CONFIG = CONFIG['openai']
TTS_CONFIG = CONFIG['tts']
ALIGNMENT_CONFIG = CONFIG['alignment']

PROMPTS = get_prompts()
SECTION_PROMPTS = PROMPTS['section_prompts']
INTEGRATE_PROMPTS = PROMPTS['integrate_prompts']
APPLICATION_PROMPTS = PROMPTS['application_prompts']

TEMPLATE_DIR = os.path.join(DIR_PATH, "templates")

