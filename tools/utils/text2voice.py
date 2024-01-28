# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from playsound import playsound
import os
from typing import Literal,List
import logging
import multiprocessing
from functools import partial
from tqdm import tqdm




YOUDAO_URL = 'https://openapi.youdao.com/ttsapi'
APP_KEY = '21a52d8cbabd6256'
APP_SECRET = 'MYttHedJuswCG6CNoyhd5YWH5egzQPSa'


def encrypt(signStr):
    """对字符串进行MD5加密"""
    hash_algorithm = hashlib.md5()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def do_request(data):
    """发送POST请求到有道云API并返回响应"""
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def text_to_speech(text,
                   return_type:Literal['path','bytes']='path'):
    assert return_type in ['path','bytes'], f"return_type must be 'path' or 'bytes', but got {return_type}"
    """将指定的文本转换为语音并播放"""
    data = {}
    data['langType'] = 'zh-CHS'
    salt = str(uuid.uuid1())
    signStr = APP_KEY + text + salt + APP_SECRET
    sign = encrypt(signStr)

    data['appKey'] = APP_KEY
    data['q'] = text
    data['salt'] = salt
    data['sign'] = sign

    response = do_request(data)
    contentType = response.headers['Content-Type']
    flag = True
    if contentType == "audio/mp3":
        if return_type=='bytes':
            return flag,response.content
        millis = int(round(time.time() * 1000))
        file_dir = "./res/speech_output/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        filePath = file_dir + str(millis) + ".mp3"
        with open(filePath, 'wb') as fo:
            fo.write(response.content)
        print("文件保存路径：" + filePath)
        # playsound(filePath)
        return flag,filePath
    else:
        flag = False
        return flag, response.text

def split_text(text:str,max_len=1000):
    text_len = len(text.encode('utf-8'))
    if text_len<=max_len:
        return [text]
    else:
        split_pattern = ['\n','。' , '.' , '，' , ',']
        for pattern in split_pattern:
            text_list = text.split(pattern)
            text_list = [t.strip() for t in text_list if t.strip() != '']
            if len(text_list)==1:
                continue
            else:
                cur_max_len = max([len(t.encode('utf-8')) for t in text_list])
                if cur_max_len>max_len:
                    continue
                else:
                    logging.info(f"split text with{repr(pattern)},max_len={cur_max_len}")
                    return text_list
        logging.warning(f"Can't split text into pieces with max_len={max_len}")
        return text_list

def Multi_Processing(text_list:list[str],num_processes:int=4,return_type:Literal['path','bytes']='path'):

    """
    if not limited by openai api, use this function to chat with openai api
    Args:
        article_texts: list of article text
        response_only: boolean, if True, only return response content, else return messages
        resest_messages: boolean, if True, reset messages to system , else will append messages

    Returns:

    """
    num_processes = min(num_processes,len(text_list),multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        text2speech_func = partial(text_to_speech,return_type=return_type)
        text_list = tqdm(text_list,position=0,leave=True)
        text_list.set_description(f"total {len(text_list)} section | num_processes:{num_processes}")
        results = [
            pool.apply_async(text2speech_func, args=(text,))
            for text in text_list
        ]
        pool.close()
        pool.join()
        results = [p.get() for p in results]
    return results

def text_to_speech_multi(text:str,num_processes:int=4,return_type:Literal['path','bytes']='path'):
    text_list = split_text(text,max_len=1000)
    results = Multi_Processing(text_list,num_processes=num_processes,return_type=return_type)
    flag = all([r[0] for r in results])
    if flag:
        bytes_list = [r[1] for r in results]
        bytes = b''.join(bytes_list)
        return flag,bytes
    else:
        return flag,None




if __name__ == '__main__':
    text = "我们都有一个家，名字叫中国"
    y = text_to_speech(text)
    print(type(y[1]))