from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
from MMAPIS.config.config import CONFIG,INTEGRATE_PROMPTS
import reprlib
from functools import partial
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from typing import List
import os
from openai import OpenAI
import time
import logging
from MMAPIS.config.config import GENERAL_CONFIG
import multiprocessing



class OpenAI_TTSConverter(GPT_Helper):
    def __init__(self,
                api_key,
                base_url,
                model_config:dict={},
                proxy:dict = None,
                prompt_ratio:float = 0.8,
                **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         prompt_ratio=prompt_ratio,
                         **kwargs)
        if self.model != "tts-1":
            self.model = "tts-1"

    def convert_text_to_speech(self, text:str,**kwargs):
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
            api_key=self.api_key,
            base_url=self.base_url
        )
        try:
            response = client.audio.speech.create(
                model=self.model,
                input=text,
                voice="alloy",
                response_format="mp3"
            )
            return True, response.content
        except Exception as e:
            return False, e

    def convert_texts_to_speech(self, text: str, num_processes: int = 4,
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

    def save(self,bytes_data, save_dir=None):
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

    def __repr__(self):
        return f"OpenAI_TTSConverter(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"



if __name__ == '__main__':
    # Test the OpenAI_TTSConverter class
    api_key = "sk-npSRVdUhy5gszhxT508aE80f6f0d40638a6b8309DbF5E569"
    api_base = "https://api.ai-gaochao.cn/v1"
    tts_converter = OpenAI_TTSConverter(api_key=api_key, base_url=api_base)
    print("tts_converter:", tts_converter)
    text = """
    Welcome to the Academic Morning Brief! Today, we delve into the groundbreaking paper titled "Attention Is All You Need," authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, and a talented team from Google Brain and Google Research. This paper introduces the Transformer model, a paradigm shift in neural network architecture, relying solely on attention mechanisms, ushering in revolutionary advancements in machine translation tasks.

    The Transformer model, eliminating traditional recurrent and convolutional layers,Welcome to the Academic Morning Brief! Today, we delve into the groundbreaking paper titled "Attention Is All You Need," authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, and a talented team from Google Brain and Google Research. This paper introduces the Transformer model, a paradigm shift in neural network architecture, relying solely on attention mechanisms, ushering in revolutionary advancements in machine translation tasks.

    The Transformer model, eliminating traditional recurrent and convolutional layers,Welcome to the Academic Morning Brief! Today, we delve into the groundbreaking paper titled "Attention Is All You Need," authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, and a talented team from Google Brain and Google Research. This paper introduces the Transformer model, a paradigm shift in neural network architecture, relying solely on attention mechanisms, ushering in revolutionary advancements in machine translation tasks.

    The Transformer model, eliminating traditional recurrent and convolutional layers,Welcome to the Academic Morning Brief! Today, we delve into the groundbreaking paper titled "Attention Is All You Need," authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, and a talented team from Google Brain and Google Research. This paper introduces the Transformer model, a paradigm shift in neural network architecture, relying solely on attention mechanisms, ushering in revolutionary advancements in machine translation tasks.
    """
    # flag, speech = tts_converter.convert_texts_to_speech(
    #     text=text,
    #     return_bytes=True
    # )
    # print(f"Speech conversion successful: {flag}")
    # print(f"Speech content: {type(speech)}")
    # # Save the speech content to a file
    # file_path = tts_converter.save(speech)