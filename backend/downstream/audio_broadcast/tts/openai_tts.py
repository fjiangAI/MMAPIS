from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.downstream.audio_broadcast.tts.base_tts import TTSConverter
from typing import Union, List
from MMAPIS.backend.config.config import CONFIG,DOCUMENT_PROMPTS
import reprlib
from functools import partial
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from typing import List
import os
from openai import OpenAI
import time
import logging
from MMAPIS.backend.config.config import GENERAL_CONFIG
import multiprocessing
from playsound import playsound
from typing import Tuple
from pathlib import Path



class OpenAITTSConverter:
    def __init__(self,
                api_key,
                base_url,
                model_config:dict={},
                proxy:dict = None,
                prompt_ratio:float = 0.8,
                **kwargs):
        self.tts_generator = GPTHelper(api_key=api_key,
                                       base_url=base_url,
                                       model_config=model_config,
                                       proxy=proxy,
                                       prompt_ratio=prompt_ratio,
                                       **kwargs)
        self.tts_generator.check_model(model_type="tts")

    def convert_text_to_speech(self, text:str,**kwargs)->Tuple[bool,Union[bytes,str]]:
        """
        Convert text to speech using the TTS service.
        :param text: Text to convert to speech.
        :return: A string containing the speech.
        """
        # Implement the logic to convert text to speech
        # Example (pseudocode):
        # rich_text = nougat_helper.convert_pdf_to_rich_text(pdf_path)
        # return rich_text
        client = OpenAI(
            api_key=self.tts_generator.api_key,
            base_url=self.tts_generator.base_url
        )
        try:
            response = client.audio.speech.create(
                model=self.tts_generator.model,
                input=text,
                voice="alloy",
                response_format="mp3"
            )
            return True, response.content
        except Exception as e:
            return False, e

    def convert_texts_to_speech(self,
                                text: str,
                                num_processes: int = 4,
                                return_bytes: bool = False,
                                save_dir: str = None):
        text_list = self.split_text(text)
        results = self.multi_processing(text_list, num_processes=num_processes)
        flag = all([r[0] for r in results])
        if flag:
            bytes_list = [r[1] for r in results]
            bytes = b''.join(bytes_list)
            if return_bytes:
                return flag, bytes
            else:
                file_path = self.save(bytes, save_dir=save_dir)
                return flag, file_path
        else:
            return flag, None
    @staticmethod
    def save(bytes_data, save_dir=None):
        millis = int(round(time.time() * 1000))
        if not save_dir:
            save_dir = os.path.join(GENERAL_CONFIG['save_dir'],"speech_output")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, str(millis) + ".mp3")
        with open(file_path, 'wb') as fo:
            fo.write(bytes_data)
        print(f"File saved at {file_path}")
        return file_path


    @staticmethod
    def split_text(text: str, max_len=1000):
        text_len = len(text.encode('utf-8'))
        if text_len <= max_len:
            return [text]
        else:
            split_pattern = ['\n', '。', '.', '，', ',']
            for pattern in split_pattern:
                text_list = text.split(pattern)
                text_list = [t.strip() for t in text_list if t.strip() != '']
                if len(text_list) == 1:
                    continue
                else:
                    cur_max_len = max([len(t.encode('utf-8')) for t in text_list])
                    if cur_max_len > max_len:
                        continue
                    else:
                        logging.info(f"split text with{repr(pattern)},max_len={cur_max_len}")
                        return text_list
            return text_list

    def multi_processing(self, text_list: List[str], num_processes: int = 4):

        num_processes = min(num_processes, len(text_list), multiprocessing.cpu_count())
        with Pool(processes=num_processes) as pool:
            text2speech_func = partial(self.convert_text_to_speech)
            text_list = tqdm(text_list, position=0, leave=True)
            text_list.set_description(
                f"[TTS Converter] Total {len(text_list)} section | num_processes:{num_processes}")
            results = [
                pool.apply_async(text2speech_func, args=(text,))
                for text in text_list
            ]
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        return results

    def play_sound(self,data:Union[str,bytes,Path],save_dir:str = None):
        if not save_dir:
            save_dir = os.path.join(GENERAL_CONFIG['save_dir'],"speech_output")
        if isinstance(data,bytes):
            file_path = self.save(data,save_dir=save_dir)
            playsound(file_path)
        if isinstance(data,str) or isinstance(data,Path):
            playsound(data)
        else:
            raise TypeError(f"Unsupported data type {type(data)}")

    def __repr__(self):
        return f"OpenAI_TTSConverter(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
