from pydantic import BaseModel,Field
from typing import List,Union,Literal,Dict
from MMAPIS.backend.config.config import GENERAL_CONFIG,ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES,APPLICATION_PROMPTS, SECTION_PROMPTS,DOCUMENT_PROMPTS,TTS_CONFIG

class ArxivRequest(BaseModel):
    """
    Class representing the parameters for an Arxiv search request.
    """
    key_word:Union[str,None,List[str]] = Field(None,example="graph neural network",description="The search keyword")
    searchtype:str = Field('all',example="all",description="The search type")
    abstracts:str = Field('show',example="show",description="Whether to show abstracts")
    order:str = Field("-announced_date_first",example="-announced_date_first",description="The search order")
    size:int = Field(50,example=50,description="The search size")
    max_return:int = Field(10,example=10,description="The maximum number of return items")
    line_length:int = Field(15,example=15,description="The display line length of abstracts in the front end")
    return_md: bool = Field(False,example=False,description="Whether to return Markdown")
    daily_type: str = Field("cs",example="cs",description="The search daily type, only used when key_word is None, i.e. search new submissions")




class gpt_model_config(BaseModel):
    """
    Class representing the configuration for the GPT model.
    """
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
    """
    Class representing the configuration for the summarizer.
    """
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

class SummaryRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",description="openai api key")
    base_url: str = Field(...,example="https://xxxxxx",description="base url")
    raw_md_text:Union[str,None] = Field(...,example="# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n",description="article text need to be summarized")
    file_name:Union[str,None] = Field(None,example="xxxxxx",description="file name")
    min_grained_level: int = Field(3,example=3,description="initial grid of section summarizer, 3 means ### [Title]")
    max_grained_level: int = Field(4,example=4,description="max grid of section summarizer, 4 means #### [Title]")
    section_prompts: Union[Dict, str] = SECTION_PROMPTS
    document_prompts: Union[Dict, str] = DOCUMENT_PROMPTS
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for section summarizer",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "artile_text": "# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n",
                    "file_name": "xxxxxx",
                    "min_grained_level": 2,
                    "max_grained_level": 4,
                    "summarizer_config": {
                        "requests_per_minute": 3,
                        "ignore_titles": ["references", "appendix"],
                        "prompt_factor": 0.8,  # prompt tokens / total tokens
                        "summary_prompts":
                            {
                                "system": "xxxx",
                                "abstract": "xxxx",
                                "introduction": "xxxx",
                                "related_work": "xxxx",
                                "method": "xxxx",
                                "experiment": "xxxx",
                                "conclusion": "xxxx",
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
                }
            ]

        }
    }

class SectionSummaryRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",description="openai api key")
    base_url: str = Field(...,example="https://xxxxxx",description="base url")
    raw_md_text:Union[str,None] = Field(..., example="# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n", description="article text need to be summarized")
    file_name:Union[str,None] = Field(None,example="xxxxxx",description="file name")
    min_grained_level: int = Field(3,example=3,description="initial grid of section summarizer, 3 means ### [Title]")
    max_grained_level: int = Field(4,example=4,description="max grid of section summarizer, 4 means #### [Title]")
    section_prompts: Union[Dict, str] = SECTION_PROMPTS
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for section summarizer",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "artile_text": "# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n",
                    "file_name": "xxxxxx",
                    "min_grained_level": 2,
                    "max_grained_level": 4,
                    "summarizer_config": {
                        "requests_per_minute": 3,
                        "ignore_titles": ["references", "appendix"],
                        "prompt_factor": 0.8,  # prompt tokens / total tokens
                        "summary_prompts":
                            {
                                "system": "xxxx",
                                "abstract": "xxxx",
                                "introduction": "xxxx",
                                "related_work": "xxxx",
                                "method": "xxxx",
                                "experiment": "xxxx",
                                "conclusion": "xxxx",
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
                }
            ]

        }
    }

class DocumentLevelSummaryRequest(BaseModel):
    api_key: str = Field(..., example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(..., example="https://xxxxxx")
    section_level_summary: Union[str, List[str]] = Field(..., example="xxxxxx")
    document_prompts: dict = DOCUMENT_PROMPTS
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for document level summarizer",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "section_level_summary": "xxxxxx",
                    "document_prompts":
                        {
                            "integrate": "xxxx",
                            "integrate_system": "xxxx",
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


class RecommendationRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://api.ai-gaochao.cn/v1")
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




class AlignmentRequest(BaseModel):
    text:Union[str,None] = ...
    img_paths:Union[List,None] = ...
    raw_md_text:Union[str,None] = None
    min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level']
    max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level']
    img_width: int = ALIGNMENT_CONFIG['img_width']
    threshold: float = ALIGNMENT_CONFIG['threshold']
    margin:int = ALIGNMENT_CONFIG['margin']
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for alignment",
                    "text": "xxxxxx",
                    "img_paths": "[SectionIMGPaths,SectionIMGPaths]",
                    "min_grained_level": 2,
                    "max_grained_level": 4,
                    "img_width": 500,
                    "threshold": 0.7
                }
            ]

        }
    }

class RegenerationRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://api.ai-gaochao.cn/v1")
    section_level_summary: Union[str, List[str]] = ...
    regenerate_prompts: dict = APPLICATION_PROMPTS["regenerate_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for regeneration",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "document_level_summary": "xxxxxx",
                    "section_level_summary": "xxxxxx",
                    "regenerate_prompts":
                        {
                            "regenerate": "xxxx",
                            "regenerate_system": "xxxx",
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
    blog_prompts: dict = APPLICATION_PROMPTS["blog_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()




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
    user_id:str = Field("Unknown_user",example="user_id")



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

