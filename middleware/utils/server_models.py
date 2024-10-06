from pydantic import BaseModel,Field
from typing import List,Union,Literal,Dict
from MMAPIS.middleware.config import GENERAL_CONFIG,ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES,APPLICATION_PROMPTS, SECTION_PROMPTS,DOCUMENT_PROMPTS,TTS_CONFIG


class gpt_model_config(BaseModel):
    model: str = OPENAI_CONFIG['model_config']['model']
    temperature:float = OPENAI_CONFIG['model_config']['temperature']
    max_tokens: int = OPENAI_CONFIG['model_config']['max_tokens']
    top_p:float = OPENAI_CONFIG['model_config']['top_p']
    frequency_penalty:float = OPENAI_CONFIG['model_config']['frequency_penalty']
    presence_penalty:float = OPENAI_CONFIG['model_config']['presence_penalty']
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for openai model",
                    "model": "gpt3.5-turbo-16k-0613",
                    "temperature": 0.9,
                    "max_tokens": 16385,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            ]
        }
    }

class Summarizer_Config(BaseModel):
    rpm_limit:int = OPENAI_CONFIG['rpm_limit']
    ignore_titles: Union[List, None] = OPENAI_CONFIG['ignore_title']
    prompt_ratio:float = OPENAI_CONFIG['prompt_ratio']
    num_processes:int = OPENAI_CONFIG['num_processes']
    gpt_model_params: Union[gpt_model_config, None] = gpt_model_config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rpm_limit": 3,
                    "ignore_titles": ["references", "appendix"],
                    "prompt_ratio": 0.8,
                    "num_processes": 10,
                    "gpt_config":
                        {
                            "description": "parameters for openai api",
                            "model": "gpt3.5-turbo-16k-0613",
                            "temperature": 0.9,
                            "max_tokens": 16385,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                        }
                }
            ]

        }
    }

summarizer_config = Summarizer_Config()

class ArxivRequest(BaseModel):
    key_word:Union[str,None,List[str]] = Field(None,example="graph neural network",description="search key word")
    searchtype:str = Field('all',example="all",description="search type")
    abstracts:str = Field('show',example="show",description="if show abstracts")
    order:str = Field("-announced_date_first",example="-announced_date_first",description="search order")
    size:int = Field(50,example=50,description="search size")
    max_return:int = Field(10,example=10,description="max number of return items")
    line_length:int = Field(15,example=15,description="display line length of abstracts in the front end")
    return_md: bool = Field(False,example=False,description="if return markdown")
    daily_type: str = Field("cs",example="cs",description="search daily type, only used when key_word is None, i.e. search new submissions")
    model_config = {"json_schema_extra": {
                        "examples": [
                            {
                            "description": "parameters for get arxiv links",
                            "key_word": "graph neural network",
                            "searchtype": "all",
                            "abstracts": "show",
                            "order": "-announced_date_first",
                            "size": 50,
                            "max_return": 10,
                            "line_length": 15,
                            "return_md": False,
                            "daily_type": "cs"
                            }
                        ]
                    }
                }

class RecommendationRequest(BaseModel):
    api_key: str = Field(..., example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(..., example="https://api.ai-gaochao.cn/v1")
    document_level_summary: str = ...
    raw_md_text: str = ...
    score_prompts: dict = APPLICATION_PROMPTS["score_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for recommendation",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "document_level_summary": "xxxxxx",
                    "raw_text": "xxxxxx",
                    "score_prompts":
                        {
                            "score": "xxxx",
                            "score_system": "xxxx",
                        },
                    "summarizer_params":
                        {
                            "rpm_limit": 3,
                            "ignore_titles": ["references", "appendix"],
                            "prompt_ratio": 0.8,
                            "num_processes": 10,
                            "gpt_config":
                                {
                                    "description": "parameters for openai api",
                                    "model": "gpt3.5-turbo-16k-0613",
                                    "temperature": 0.9,
                                    "max_tokens": 16385,
                                    "top_p": 1,
                                    "frequency_penalty": 0,
                                    "presence_penalty": 0,

                                }
                        }
                }
            ]

        }
    }


class BlogRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://api.ai-gaochao.cn/v1")
    section_level_summary:str= Field(...)
    document_level_summary: str= Field(...)
    pdf:str= Field(...)
    blog_prompts: dict = APPLICATION_PROMPTS["blog_prompts"]
    min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level']
    max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level']
    img_width: int = ALIGNMENT_CONFIG['img_width']
    threshold: float = ALIGNMENT_CONFIG['threshold']
    margin:int = ALIGNMENT_CONFIG['margin']
    raw_md_text:str = None
    summarizer_params: Summarizer_Config = Summarizer_Config()
    from_middleware: bool = False




class BroadcastScriptRequest(BaseModel):
    llm_api_key: str
    llm_base_url: str
    document_level_summary: str
    section_level_summary: Union[str, List[str]]
    broadcast_prompts: dict = APPLICATION_PROMPTS["broadcast_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()

class TTSRequest(BaseModel):
    text: str = Field(...,example="text",description="text to be converted to speech")
    model: str = Field(TTS_CONFIG['model'],example="youdao")
    api_key: str = Field(TTS_CONFIG['api_key'],example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(TTS_CONFIG['base_url'],example="https://xxxxxx")
    app_secret: str = Field(TTS_CONFIG['app_secret'],example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")



class Multimodal_QA_Request(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://xxxxxx")
    user_input: str = Field(...,example="user input")
    document_level_summary: str = Field(...,example="document level summary")
    session_message: List = Field(...,example="session message")
    raw_md_text: str = Field(...,example="article")
    prompts: dict = Field(APPLICATION_PROMPTS["multimodal_qa"],example="{'qa': 'xxxx', 'qa_system': 'xxxx'}")
    min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level']
    max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level']
    ignore_titles: Union[List, None] = OPENAI_CONFIG['ignore_title']
    detailed_img: bool = False
    img_width: int = 300
    margin: int = 10
    prompt_ratio: float = 0.8
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for multimodal qa",
                    "multimodal_qa_generator": {
                        "description": "parameters for multimodal qa",
                        "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                        "base_url": "https://xxxxxx",
                        "model_config": {
                            "description": "parameters for openai api",
                            "model": "gpt3.5-turbo-16k-0613",
                            "temperature": 0.9,
                            "max_tokens": 16385,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                        },
                        "prompt_ratio": 0.8,
                        "num_processes": 10,
                    }
                }
            ]

        }
    }