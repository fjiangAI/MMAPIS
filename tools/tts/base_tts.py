import logging
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from typing import List

class TTSConverter:
    def __init__(self,base_url,api_key):
        """
        Initialize the TTS
        """
        # Initialization, if needed (e.g., setting up configurations for Nougat library)
        self.base_url = base_url
        self.api_key = api_key


    def convert_text_to_speech(self, text,**kwargs):
        """
        Convert text to speech using the TTS service.
        :param text: Text to convert to speech.
        :return: A string containing the speech.
        """
        # Implement the logic to convert text to speech
        # Example (pseudocode):
        # rich_text = nougat_helper.convert_pdf_to_rich_text(pdf_path)
        # return rich_text
        raise NotImplementedError("Please Implement this method")

    def request_api(self,data):
        """send POST request to TTS API and return response"""
        raise NotImplementedError("Please Implement this method")

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
            logging.warning(f"Can't split text into pieces with max_len={max_len}")
            return text_list

    def multi_processing(self,text_list: List[str], num_processes: int = 4):

        num_processes = min(num_processes, len(text_list), multiprocessing.cpu_count())
        with Pool(processes=num_processes) as pool:
            text2speech_func = partial(self.convert_text_to_speech)
            text_list = tqdm(text_list, position=0, leave=True)
            text_list.set_description(f"[TTS Converter] Total {len(text_list)} section | num_processes:{num_processes}")
            results = [
                pool.apply_async(text2speech_func, args=(text,))
                for text in text_list
            ]
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        return results



