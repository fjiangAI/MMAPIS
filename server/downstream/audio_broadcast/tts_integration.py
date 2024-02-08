from MMAPIS.server.downstream.audio_broadcast.script_conversion import Broadcast_Generator
from MMAPIS.tools.tts import YouDaoTTSConverter
from typing import List, Union
from MMAPIS.config.config import TTS_CONFIG,OPENAI_CONFIG, APPLICATION_PROMPTS
import reprlib
import os
class BroadcastTTSGenerator():
    def __init__(self,
                 llm_api_key,
                 llm_base_url,
                 tts_base_url:str=None,
                 tts_api_key:str=None,
                 app_secret:str=None,
                 model_config:dict={},
                 proxy:dict = None,
                 **kwargs
                 ):
        self.broadcast_generator = Broadcast_Generator(api_key=llm_api_key,
                                                        base_url=llm_base_url,
                                                        model_config=model_config,
                                                        proxy=proxy,
                                                        **kwargs)

        self.tts_converter = YouDaoTTSConverter(base_url=tts_base_url,
                                                api_key=tts_api_key,
                                                app_secret=app_secret,
                                                proxy=proxy,
                                                )

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
        flag,content = self.broadcast_generator.broadcast_generation(document_level_summary=document_level_summary,
                                                                    section_summaries=section_summaries,
                                                                    broadcast_prompts=broadcast_prompts,
                                                                    reset_messages=reset_messages,
                                                                    response_only=response_only,
                                                                    raw_marker=raw_marker,
                                                                    final_marker=final_marker,
                                                                    **kwargs)
        if all([self.tts_converter.base_url,self.tts_converter.api_key,self.tts_converter.app_secret]):
            flag, bytes_content = self.tts_converter.convert_texts_to_speech(content,return_bytes=return_bytes)
        else:
            bytes_content = None
            flag = True
        return flag,content, bytes_content

    def play_sound(self, bytes_content:bytes):
        self.tts_converter.playsound(bytes_content)


if __name__ == "__main__":
    user_input_path = "../integrate.md"
    with open(user_input_path, 'r') as f:
        user_input = f.read()
    tts_base_url = TTS_CONFIG['base_url']
    tts_api_key = TTS_CONFIG['api_key']
    app_secret = TTS_CONFIG['app_secret']
    llm_api_key = OPENAI_CONFIG["api_key"]
    llm_base_url = OPENAI_CONFIG["base_url"]
    model_config = OPENAI_CONFIG["model_config"]
    broadcast_tts_generator = BroadcastTTSGenerator(llm_api_key=llm_api_key,
                                                    llm_base_url=llm_base_url,
                                                    tts_base_url=tts_base_url,
                                                    tts_api_key=tts_api_key,
                                                    app_secret=app_secret,
                                                    model_config=model_config,
                                                    )
    broadcast_prompts = APPLICATION_PROMPTS["broadcast_prompts"]
    section_summaries_path = "../summary.md"
    with open(section_summaries_path, 'r') as f:
        section_summaries = f.read()

    user_input_path = "../integrate.md"
    with open(user_input_path, 'r') as f:
        user_input = f.read()

    flag,broadcast_script,bytes_content = broadcast_tts_generator.broadcast_tts_generation(document_level_summary=user_input,
                                                                           section_summaries=section_summaries,
                                                                           broadcast_prompts=broadcast_prompts,
                                                                           return_bytes=True,
                                                                            )
    broadcast_tts_generator.play_sound(bytes_content)
    print(broadcast_script)
    broadcast_tts_generator.play_sound(bytes_content)


