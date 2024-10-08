from MMAPIS.backend.downstream.audio_broadcast.tts.base_tts import TTSConverter
from MMAPIS.backend.config.config import GENERAL_CONFIG
import requests
import hashlib
import uuid


class YouDaoTTSConverter(TTSConverter):
    def __init__(self,base_url,api_key,app_secret,proxy:dict = None):
        """
        Initialize the YouDaoTTS
        :param base_url:
        :param api_key:
        :param app_secret:
        """
        super().__init__(base_url,api_key)
        self.app_secret = app_secret
        self.proxy = proxy

    @staticmethod
    def encrypt(signStr):
        """encrypt signStr with MD5"""
        hash_algorithm = hashlib.md5()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()

    def request_api(self,data):
        """send POST request to YouDao TTS API and return response"""
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return requests.post(self.base_url, data=data, headers=headers,proxies=self.proxy)


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
            return flag, response.content
        else:
            flag = False
            return flag, response.text


    def convert_texts_to_speech(self,text: str, num_processes: int = 4,
                                return_bytes: bool = False,
                                save_dir:str = None):
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


    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)



