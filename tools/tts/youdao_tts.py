from MMAPIS.tools.tts.base_tts import TTSConverter
import requests
import hashlib
import time
from playsound import playsound
import os
import uuid
from MMAPIS.config.config import TTS_CONFIG
from typing import Union
from pathlib import Path

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
        lens = [len(i) for i in text_list]
        results = self.multi_processing(text_list, num_processes=num_processes)
        flag = all([r[0] for r in results])
        print("return_bytes",return_bytes)
        for r in results:
            print(r[0],"res:",len(r[1]),"type:",type(r[1]))
        print("flag",flag)
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
        file_path = os.path.join(save_dir, str(millis) + ".mp3")
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
    youdao_tts = YouDaoTTSConverter(base_url = TTS_CONFIG['base_url'],
                                    api_key=TTS_CONFIG['api_key'],
                                    app_secret= TTS_CONFIG['app_secret'])

    text = """
'Welcome to our discussion, today we will introduce a paper titled "Attention Is All You Need," with primary authors including Vaswani et al. The paper proposes a network architecture called the Transformer that relies solely on attention mechanisms, eliminating the need for recurrent or convolutional neural networks.\n\nNext, we will delve into the model architecture of the Transformer. The core research model consists of an encoder and a decoder, both composed of stacked layers. The encoder stack includes multi-head self-attention mechanisms and position-wise fully connected feed-forward networks. The decoder stack performs multi-head attention over the output of the encoder stack and modifies the self-attention sub-layer to ensure autoregressive generation of output symbols. The attention mechanism employed is called Scaled Dot-Product Attention, which allows the model to attend to different positions within the input sequence efficiently. Multi-Head Attention is introduced to enable joint attention to different representation subspaces.\n\nThe empirical analysis on machine translation tasks demonstrates the effectiveness of the Transformer model. It achieves superior translation quality compared to existing models, including ensembles, with a BLEU score of 28.4 on the WMT 2014 English-to-German translation task and a new state-of-the-art BLEU score of 41.8 on the WMT 2014 English-to-French translation task. The model also generalizes well to other tasks, such as English constituency parsing.\n\nThe experimental techniques involve training the Transformer on large-scale datasets using GPUs. The authors provide details on the data collection and processing methods, training hardware, and schedule. The Transformer\'s efficacy is supported by substantial improvements in translation quality, parallelizability, and training time compared to previous models.\n\nHowever, it is important to note that the summary lacks specific quantitative measurements to support the claims made about the model\'s performance. Including these measurements would enhance the clarity and credibility of the summary. Additionally, the limitations or challenges encountered in the experimental analysis are not mentioned, which could provide a more balanced perspective on the findings of the paper.\n\nIn conclusion, the paper "Attention Is All You Need" introduces the Transformer model, a network architecture that relies solely on attention mechanisms. The model achieves superior translation quality, captures long-range dependencies, and enables parallelization while requiring less training time compared to previous models. The empirical analysis validates the effectiveness of the Transformer and its potential for advancing the field of machine translation and natural language processing.'
    """
    flag, bytes_content = youdao_tts.convert_texts_to_speech(text,return_bytes=True)
    youdao_tts.playsound(bytes_content)