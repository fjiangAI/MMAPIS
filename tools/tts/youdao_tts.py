from .base_tts import TTSConverter
import requests
import hashlib
import time
from playsound import playsound
import os
import uuid
from MMAPIS.config.config import CONFIG
from typing import Union
from pathlib import Path

class YouDaoTTSConverter(TTSConverter):
    def __init__(self,base_url,api_key,app_secret):
        """
        Initialize the YouDaoTTS
        :param base_url:
        :param api_key:
        :param app_secret:
        """
        super().__init__(base_url,api_key)
        self.app_secret = app_secret

    @staticmethod
    def encrypt(signStr):
        """encrypt signStr with MD5"""
        hash_algorithm = hashlib.md5()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()

    def request_api(self,data):
        """send POST request to YouDao TTS API and return response"""
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return requests.post(self.base_url, data=data, headers=headers)

    def convert_text_to_speech(
                       self,
                       text:str,
                       **kwargs):
        """convert text to speech"""
        data = {}
        data['langType'] = 'zh-CHS'
        salt = str(uuid.uuid1())
        signStr = self.api_key + text + salt + self.app_secret
        sign = self.encrypt(signStr)

        data['appKey'] = self.api_key
        data['q'] = text
        data['salt'] = salt
        data['sign'] = sign

        response = self.request_api(data)
        contentType = response.headers['Content-Type']
        flag = True
        if contentType == "audio/mp3":
            # if return_bytes:
            return flag, response.content
            # millis = int(round(time.time() * 1000))
            # file_dir = "./res/speech_output/"
            # if not os.path.exists(file_dir):
            #     os.makedirs(file_dir)
            # filePath = file_dir + str(millis) + ".mp3"
            # with open(filePath, 'wb') as fo:
            #     fo.write(response.content)
            # print(f"File saved at {filePath}")
            # playsound(filePath)
            # return flag, filePath
        else:
            flag = False
            return flag, response.text


    def convert_texts_to_speech(self,text: str, num_processes: int = 4,
                                return_bytes: bool = False,save_dir:str = None):
        text_list = self.split_text(text, max_len=1000)
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

    def save(self,bytes_data, save_dir):
        millis = int(round(time.time() * 1000))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = save_dir + str(millis) + ".mp3"
        with open(file_path, 'wb') as fo:
            fo.write(bytes_data)
        print(f"File saved at {file_path}")
        return file_path

    def playsound(self,data:Union[str,bytes,Path]):
        if isinstance(data,bytes):
            file_path = self.save(data,save_dir="./res/speech_output/")
            playsound(file_path)
        if isinstance(data,str) or isinstance(data,Path):
            playsound(data)
        else:
            raise TypeError(f"Unsupported data type {type(data)}")

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)



if __name__ == "__main__":
    youdao_tts = YouDaoTTSConverter(base_url = CONFIG['tts']['base_url'],
                                    api_key= CONFIG['tts']['api_key'],
                                    app_secret= CONFIG['tts']['app_secret'])

    text = """
This works because ThreadPool shares memory with the main thread, rather than creating a new process- this means that pickling is not required.

The downside to this method is that python isn't the greatest language with handling threads- it uses something called the Global Interpreter Lock to stay thread safe, which can slow down some use cases here. However, if you're primarily interacting with other systems (running HTTP commands, talking with a database, writing to filesystems) then your code is likely not bound by CPU and won't take much of a hit. In fact I've found when writing HTTP/HTTPS benchmarks that the threaded model used here has less overhead and delays, as the overhead from creating new processes is much higher than the overhead for creating new threads and the program was otherwise just waiting for HTTP responses.

So if you're processing a ton of stuff in python userspace this might not be the best method.
This works because ThreadPool shares memory with the main thread, rather than creating a new process- this means that pickling is not required.

The downside to this method is that python isn't the greatest language with handling threads- it uses something called the Global Interpreter Lock to stay thread safe, which can slow down some use cases here. However, if you're primarily interacting with other systems (running HTTP commands, talking with a database, writing to filesystems) then your code is likely not bound by CPU and won't take much of a hit. In fact I've found when writing HTTP/HTTPS benchmarks that the threaded model used here has less overhead and delays, as the overhead from creating new processes is much higher than the overhead for creating new threads and the program was otherwise just waiting for HTTP responses.

So if you're processing a ton of stuff in python userspace this might not be the best method.
    """
    flag, bytes_content = youdao_tts.convert_texts_to_speech(text,return_bytes=True)
    youdao_tts.playsound(bytes_content)