from MMAPIS.server.downstream.audio_broadcast.script_conversion import Broadcast_Generator
from MMAPIS.tools.tts import YouDaoTTSConverter, OpenAI_TTSConverter
from typing import List, Union
from MMAPIS.config.config import TTS_CONFIG,OPENAI_CONFIG, APPLICATION_PROMPTS
import reprlib
import os
import logging


class BroadcastTTSGenerator():
    def __init__(self,
                 llm_api_key,
                 llm_base_url,
                 tts_base_url:str=None,
                 tts_api_key:str=None,
                 tts_model:str="youdao",
                 app_secret:str=None,
                 model_config:dict={},
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs
                 ):
        self.tts_model = tts_model
        self.broadcast_generator = Broadcast_Generator(api_key=llm_api_key,
                                                   base_url=llm_base_url,
                                                   model_config=model_config,
                                                   proxy=proxy,
                                                   prompt_ratio=prompt_ratio,
                                                   **kwargs)

        if tts_model == "youdao":
            self.tts_converter = YouDaoTTSConverter(base_url=tts_base_url,
                                                api_key=tts_api_key,
                                                app_secret=app_secret,
                                                proxy=proxy,
                                                )
        elif tts_model == "openai":
            self.tts_converter = OpenAI_TTSConverter(base_url=llm_base_url,api_key=llm_api_key)
        else:
            raise ValueError(f"Unsupported TTS model: {tts_model}, supported models are: 'youdao','openai'")


    def broadcast_script_generation(self,
                                    document_level_summary:str,
                                    section_summaries:Union[str, List[str]],
                                    broadcast_prompts:dict = None,
                                    reset_messages:bool = True,
                                    response_only:bool = True,
                                    raw_marker:str = "Raw Broadcast Content",
                                    final_marker:str = "New Broadcast Content",
                                    ):
        flag,content = self.broadcast_generator.broadcast_generation(document_level_summary=document_level_summary,
                                                                    section_summaries=section_summaries,
                                                                    broadcast_prompts=broadcast_prompts,
                                                                    reset_messages=reset_messages,
                                                                    response_only=response_only,
                                                                    raw_marker=raw_marker,
                                                                    final_marker=final_marker)
        return flag,content

    def text2speech(self, text:str, return_bytes:bool = False):
        if self.tts_model == "youdao" and not all([self.tts_converter.base_url,self.tts_converter.api_key,self.tts_converter.app_secret]):
            logging.warning("Due to missing TTS credentials, the text to speech conversion will not be performed.")
            return True,None
        flag, bytes_content = self.tts_converter.convert_texts_to_speech(text,return_bytes=return_bytes)
        return flag, bytes_content

    def broadcast_tts_generation(self,
                                 document_level_summary:str,
                                 section_summaries:Union[str, List[str]],
                                 broadcast_prompts:dict = None,
                                 reset_messages:bool = True,
                                 response_only:bool = True,
                                 raw_marker:str = "Raw Broadcast Content",
                                 final_marker:str = "New Broadcast Content",
                                 return_bytes:bool = False,
                                 **kwargs):
        script_flag,content = self.broadcast_script_generation(document_level_summary=document_level_summary,
                                                        section_summaries=section_summaries,
                                                        broadcast_prompts=broadcast_prompts,
                                                        reset_messages=reset_messages,
                                                        response_only=response_only,
                                                        raw_marker=raw_marker,
                                                        final_marker=final_marker)
        tts_flag, bytes_content = self.text2speech(content,return_bytes=return_bytes)
        flag = script_flag and tts_flag
        return flag,content, bytes_content


    def play_sound(self, bytes_content:bytes):
        self.tts_converter.play_sound(bytes_content)

    def save(self, bytes_content:bytes, save_dir:str):
        return self.tts_converter.save(bytes_content,save_dir=save_dir)





