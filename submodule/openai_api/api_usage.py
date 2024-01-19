import logging
import re

import openai
import os
from .split_text import split2pieces,num_tokens_from_messages
from .split_text import assgin_prompts
import time
from typing import Union, List,Literal
import multiprocessing
from functools import partial
import requests
import json
import  reprlib
# import concurrent.futures
from tqdm import tqdm
from tabulate import tabulate

class OpenAI_Summarizer():
    def __init__(self,
                 api_key:str,
                 proxy:dict = None,
                 # ['url','openai']
                 acquire_mode:Literal['url','openai'] = 'url',
                 prompt_factor: float = 0.8,  # prompt tokens / total tokens
                 summary_prompts: Union[dict,str] = None,
                 resummry_prompts: Union[dict,str] = None,
                 split_mode: str = 'group',
                 ignore_titles:list = None,
                 num_processes: int = 3,
                 requests_per_minute:Union[int,None] = 3, # if api_key is limited, set this to 3, else set to None or 0
                 base_url:Literal["https://openai.huatuogpt.cn/v1",'https://api.openai.com/v1'] = "https://openai.huatuogpt.cn/v1",
                 **kwargs):
        """
        Args:
            :param openai_info: dict, openai api info, e.g. {'api_key':openai_key}
            :param proxy: dict, proxy info, e.g. {'host':'http://localhost:7890'} or None
            :param acquire_mode: str, ['url','openai'], if 'url', use url to connect openai api, else use openai api package
            :param prompt_factor: float, prompt tokens / total tokens ,i.e. the ratio of prompt tokens to total tokens
            :param summary_prompts: dict or str, summary prompts, e.g. {'system':'you are a helpful assistant who are great in making plans','general_summary':'list the plans and their pros and cons'}
            :param resummry_prompts: dict or str, resummry prompts, e.g. {'system':'you are a helpful assistant who are great in making plans','overview':'list the plans and their pros and cons'}
            :param split_mode: str, ['half','group'], if 'half', split text into chunks based on length, else split text into chunks based on title group
            :param ignore_titles: list, list of subtitles to ignore
            :param num_processes: int, number of processes to use
            :param requests_per_minute: int, number of requests per minute, if api_key is limited, set this to 3(means 3 requests per minute), else set to None
            :param base_url: str, base url of openai api
            :param kwargs:  dict, model info, e.g. {'model':'gpt-3.5-turbo-16k-0613','temperature':0.9,'max_tokens':16385,'top_p':1,'frequency_penalty':0,'presence_penalty':0}
        """
        self.api_key = api_key
        self.proxy = proxy
        self.host = self.proxy['host'] if self.proxy and self.proxy.get('host',None) else None
        self.init_openai_connect()
        self.ignore_titles = ignore_titles
        self.summary_prompts = summary_prompts if isinstance(summary_prompts,dict) else {'general_summary':summary_prompts}
        self.resummry_prompts = resummry_prompts if isinstance(resummry_prompts,dict) else {'overview':resummry_prompts}
        self.prompt_factor = prompt_factor
        self.split_mode = split_mode
        self.base_url = base_url
        self.acquire_mode = acquire_mode
        self.requests_per_minute = requests_per_minute
        self.num_processes = min(self.requests_per_minute,multiprocessing.cpu_count(),num_processes) if self.requests_per_minute else min(multiprocessing.cpu_count(),num_processes)
        self._usages = ['regenerate','blog','speech']

        if kwargs.get('model_config',None) is not None:
            self.init_model(**kwargs['model_config'])
        else:
            self.init_model()
    @property
    def usages(self):
        return self._usages


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        msg = f"OpenAI_Summarizer(model={self.model},\napi_key:{reprlib.repr(self.api_key)},\nproxy_info:{reprlib.repr(self.proxy)},\nsplit_mode:{self.split_mode},\nprompt_factor:{self.prompt_factor},\nignore_titles:{self.ignore_titles},\nnum_processes:{self.num_processes},\nrequests_per_minute:{self.requests_per_minute},\nbase_url:{self.base_url},\nmodel_config:\nmodel:{self.model},\n   temperature:{self.temperature},\n   max_tokens:{self.max_tokens},\n   top_p:{self.top_p},\n   frequency_penalty:{self.frequency_penalty},\n   presence_penalty:{self.presence_penalty})"
        return msg

    def init_openai_connect(self):
        openai.api_key = self.api_key
        if self.host:
            os.environ["http_proxy"] = self.host
            os.environ["https_proxy"] = self.host

    def init_model(self,
                   model:str = 'gpt-3.5-turbo-16k-0613',
                   temperature:float = 0.9,
                   max_tokens:int = 16385,
                   top_p:float = 1,
                   frequency_penalty:float = 0.1,
                   presence_penalty:float = 0.2):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty


    def init_messages(self,role:str,content:Union[str,List[str]]):
        # init openai role
        self.messages = []
        if isinstance(content, str):
            content = [content]
        for c in content:
            self.messages.append({'role': role, 'content': c})



    def request_chatgpt_openai(
                         self,
                         user_input:str,
                         api_key:str,
                         system_messages:Union[str,List[str]] = None,
                         reset_messages:bool = True,
                         response_only:bool = True):

        assert self.acquire_mode in ['url','openai'],f"acquire_mode should be in ['url','openai'], but got {self.acquire_mode}"
       # input user prompt
        openai.api_key = api_key
        if system_messages is not None:
            self.init_messages('system', system_messages)
        self.messages.append({'role': 'user', 'content': user_input})
        input_tokens = num_tokens_from_messages(self.messages, model=self.model)
        if input_tokens > self.max_tokens* self.prompt_factor:
           logging.warning(f'input tokens {input_tokens} is larger than max tokens {self.max_tokens* self.prompt_factor}, will cut the input')
           diff = int(input_tokens - self.max_tokens * self.prompt_factor)
           self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]
       # use openai api
        response = openai.ChatCompletion.create(
            model= self.model,
            messages=self.messages,
            temperature= self.temperature,
            max_tokens= self.max_tokens-input_tokens,
            top_p= self.top_p,
            frequency_penalty= self.frequency_penalty,
            presence_penalty= self.presence_penalty
        )

        if reset_messages:
            # pop user input
            self.messages.pop(-1)
        else:
            # add response to messages
            self.messages.append({
                'role': response['choices'][0]['message']['role'],
                'content': response['choices'][0]['message']['content']
            })
        if  response_only:
            return response['choices'][0]['message']['content']
        else:
            return self.messages


    @staticmethod
    def handle_request(url:str,parameters = None,proxy=None, headers = None):
        success = False
        response = None
        try:
            if proxy is None:
                raw_response = requests.post(url, headers=headers, json=parameters)
            else:
                raw_response = requests.post(url, headers=headers, json=parameters, proxies=proxy)

            raw_response.raise_for_status()
            response = json.loads(raw_response.content.decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            success = True
        except requests.exceptions.RequestException as e:
            content = f"Request Error: {str(e)}"
        except json.JSONDecodeError as e:
            content = f"JSON Decode Error: {str(e)}"
        except KeyError as e:
            content = f"KeyError: {str(e)}"
        except Exception as e:
            content = f"Unexpected Error: {str(e)}"

        return response,content, success



    def request_chatgpt_url(self,
                            user_input:str,
                            api_key:str,
                            system_messages:Union[str,List[str]] = None,
                            response_only:bool = True,
                            reset_messages:bool = True):
        """

        Args:
            parameters: model info,e.g.
                        parameters = {
                            "model": self.model_name,
                            "messages": messages
                            }
            response_only:boolean, if True, only return response content, else return messages
            reset_messages: boolean, if True, reset messages to system , else will append messages
        Returns:
            flag: boolean, use to

        """
        openai.api_key = api_key
        # get response content
        url = self.base_url+"/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if system_messages is not None:
            self.init_messages('system', system_messages)

        self.messages.append({'role': 'user', 'content': user_input})

        input_tokens = num_tokens_from_messages(self.messages, model=self.model)

        if input_tokens > self.max_tokens* self.prompt_factor:
           logging.warning(f'input tokens {input_tokens} is larger than max tokens {self.max_tokens* self.prompt_factor}, will cut the input')
           diff = int(input_tokens - self.max_tokens * self.prompt_factor)
           self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]

        parameters = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens-input_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        response,content, flag = self.handle_request(url = url,parameters=parameters,proxy=self.proxy,headers = headers)

        if reset_messages:
            self.messages.pop(-1)
            # self.init_messages('system', self.summary_prompts['system'])
        else:
            # add text of response to messages
            if flag:
                self.messages.append({
                    'role': response['choices'][0]['message']['role'],
                    'content': response['choices'][0]['message']['content']
                })
        if response_only:
            return flag, content
        else:
            return flag, self.messages


    def chat_with_openai(self,user_input:str,
                         api_key:str,
                         system_messages:Union[str,List[str]] = None,
                         response_only:bool = True,
                         reset_messages:bool = True):
        if self.acquire_mode == 'url':
            flag,responese = self.request_chatgpt_url(user_input = user_input,api_key = api_key,system_messages=system_messages,response_only=response_only,reset_messages=reset_messages)
            if not flag:
                responese = f"request openai error, {multiprocessing.current_process().name} error code: {responese}"
                logging.error(responese)
            return responese
        elif self.acquire_mode == 'openai':
            responese = self.request_chatgpt_openai(user_input = user_input,api_key = api_key,system_messages=system_messages,response_only=response_only,reset_messages=reset_messages)
            return responese
        else:
            raise ValueError(f"acquire_mode should be in ['url','openai'], but got {self.acquire_mode}")


    def Multi_Chat_Processing(self,article_texts:Union[List[str],str],
                              system_messages:Union[str,List[str],List[List]] = None,
                              response_only:bool = True,
                              resest_messages:bool = True):
        """
        if not limited by openai api, use this function to chat with openai api
        Args:
            article_texts: list of article text
            response_only: boolean, if True, only return response content, else return messages
            resest_messages: boolean, if True, reset messages to system , else will append messages
            system_messages:
                None: use the pre-defined system messages
                str: use the same system messages for all article texts
                List[str]: use the same system messages for each article text, where each system message is [{'system':'system message[0]'},{'system':'system message[1]'}]
                List[List]: use different system messages for each article text, assign [[{'system':'system message[i][0]'},{'system':'system message[i][1]'},{"user":"user message[i]"}],...]

        Returns:

        """
        flag = False

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            if self.acquire_mode == 'url':
                logging.info(f"connect openai api through url:{self.base_url}, with {self.num_processes} processes without rate limit")
            else:
                logging.info(f"connect openai api directly, with {self.num_processes} processes without rate limit")
            chat_func = partial(self.chat_with_openai,
                                response_only=response_only,
                                api_key = self.api_key,
                                system_messages=system_messages)
            article_texts = [article_texts] if isinstance(article_texts,str) else article_texts
            if system_messages and isinstance(system_messages[0], List):
                assert len(system_messages) == len(article_texts), logging.error(
                    f"system_messages should be list of length {len(article_texts)}, but got {len(system_messages)}")
                flag = True

            article_texts = tqdm(article_texts,position=0,leave=True)
            article_texts.set_description(f"total {len(article_texts)} section | num_processes:{self.num_processes} | requests_per_minute:{self.requests_per_minute}")
            results = [
                pool.apply_async(chat_func,kwds={'system_messages':system_messages[i] if flag else system_messages,
                                                 'user_input':article_text,
                                                })
                for i,article_text in enumerate(article_texts)
            ]
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        return results

    def process_batches(self, article_texts:Union[str,List],
                        system_messages:Union[str,List[str],List[List]] = None,
                        response_only:bool = True,
                        resest_messages:bool = True):
        """
        Due to openai key rate limit, process batches of article texts and sleep for a while after each batch
        Args:
            article_texts:
            response_only:
            resest_messages:
            system_messages:when system_messages is list of list, e.g. message:[{"system":"xxx"},{"system":"xxx"},{"user":"xxx"}],where system is a list

        Returns: list of results

        """


        if self.acquire_mode == 'url':
            logging.info(f"connect openai api through url:{self.base_url} with {self.num_processes} processing batch, limit: {self.requests_per_minute} requests per minute")
        else:
            logging.info(f"connect openai api directly, with {self.num_processes} processing batch")
        article_texts = [article_texts] if isinstance(article_texts,str) else article_texts
        flag = False
        if system_messages and isinstance(system_messages[0],List):
            assert len(system_messages) == len(article_texts),logging.error(f"system_messages should be list of length {len(article_texts)}, but got {len(system_messages)}")
            flag = True

        results = []
        chat_func = partial(self.chat_with_openai,
                            reset_messages=resest_messages,
                            response_only=response_only,
                            api_key=self.api_key)
        # ------------------------------multi threads---------------------------------
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_processes) as executor:
        #     processing_bar = tqdm(range(0, len(article_texts), self.num_processes))
        #     for i in processing_bar:
        #         processing_bar.set_description(f"total {len(article_texts)} section, processing {i}~{i+self.num_processes}")
        #         start_time = time.time()
        #         batch = article_texts[i:i + self.num_processes]
        #         futures = [executor.submit(chat_func, task_obj) for task_obj in batch]
        #         concurrent.futures.wait(futures)
        #         results.extend([future.result() for future in futures])
        #         elapsed_time = time.time() - start_time
        #         if elapsed_time < 60:
        #             slp_t = int(60 - elapsed_time)+1
        #             logging.info(f'due to rate limit,sleep for {slp_t}s')
        #             time.sleep(slp_t)  # wait util 60s
        # ------------------------------multi processes---------------------------------
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            processing_bar = tqdm(range(0, len(article_texts), self.num_processes))
            for i in processing_bar:
                processing_bar.set_description(
                    f"Processing {i}~{min(i + self.num_processes,len(article_texts))} | total {len(article_texts)} sections | num_processes:{self.num_processes} | requests_per_minute:{self.requests_per_minute}")
                start_time = time.time()
                batch = article_texts[i:i + self.num_processes]
                batch_system_messages = system_messages[i:i + self.num_processes] if flag else system_messages
                futures = pool.apply(chat_func,kwds={'system_messages':batch_system_messages,
                                                     'user_input':batch,
                                                     }
                                        )
                # futures = pool.map(chat_func, batch,batch_system_messages)
                results.extend(futures)
                elapsed_time = time.time() - start_time
                if i + self.num_processes >= len(article_texts):
                    break
                elif elapsed_time < 60:
                    slp_t = int(60 - elapsed_time) + 3
                    logging.info(f'due to rate limit, sleep for {slp_t}s')
                    time.sleep(slp_t)  # wait until 60s
        return results
    @staticmethod
    def clean_resp(raw_text_list: Union[List[str], str]):
        """
        remove the special marker in head of response or the tail of response
        Args:
            raw_text_list: list of raw response

        Returns: list of cleaned response

        """
        is_list = True
        if isinstance(raw_text_list, str):
            raw_text_list = [raw_text_list]
            is_list = False
        if not isinstance(raw_text_list, list):
            raise ValueError(f'raw_text_list should be list or str, but got {type(raw_text_list)}')

        cleaned_str_list = []
        for raw_text in raw_text_list:
            cleaned_str = re.sub(r'^[^a-zA-Z]*', '', raw_text)
            cleaned_str = re.sub(r'[^a-zA-Z.]*$', '', cleaned_str)
            cleaned_str_list.append(cleaned_str)
        if not is_list:
            return cleaned_str_list[0]
        return cleaned_str_list

    def filter_final_response(self,resps:List[str],raw_marker:str,final_marker:str ):
        """
        filter out the final response
        Args:
            resps: list of response
            raw_marker: marker of raw response
            final_marker: marker of final response

        Returns: final response

        """
        is_list = True
        if isinstance(resps,str):
            resps = [resps]
            is_list = False
        if not isinstance(resps,list):
            raise ValueError(f'resps should be list or str, but got {type(resps)}')

        craft_summary_pattern = re.compile(rf"\[{raw_marker}\]|`{raw_marker}`|{raw_marker}", re.IGNORECASE)
        final_summary_pattern = re.compile(rf"\[{final_marker}\]|`{final_marker}`|{final_marker}", re.IGNORECASE)
        res = []
        for resp in resps:
            final_summary_group = final_summary_pattern.search(resp)
            craft_summary_group = craft_summary_pattern.search(resp)

            if final_summary_group:
                res.append(self.clean_resp(resp[final_summary_group.end():].strip()))
            elif craft_summary_group:
                res.append(self.clean_resp(resp[craft_summary_group.end():].strip()))
            else:
                res.append(self.clean_resp(resp.strip()))

        if not is_list:
            return res[0]

        return res

    def summary_with_openai(self,
                            artile_text: str = ...,
                            file_name: str = ...,
                            init_grid: int = 2
                            ):

        assert self.split_mode in ['half', 'group'], logging.error(
            f'split_mode should be in ["half","group"],but got {self.split_mode}')
        self.init_messages('system', self.summary_prompts.get('system',''))

        # half: chunks [text], input prompt should not be too long,~= answer length,so max_tokens//2
        # group: split text into chunks based on title group
        # chunks:[(tag,title,subtext)]
        logging.info(f'Summary with OpenAI: Step 1 - Splitting article. Split mode: {self.split_mode}')
        # input max_tokens should be less than prompt_max_tokens - prompt_tokens
        titles, authors, affiliations, chunks, tables = split2pieces(artile_text.strip(), file_name=file_name,
                                                             max_tokens=int(self.max_tokens * self.prompt_factor),
                                                             mode=self.split_mode, ignore_title=self.ignore_titles,
                                                             init_grid=init_grid)

        if self.split_mode == 'group':
            ## for each chunk, messages remain the same
            # chunks:[(subtitle,subtext,length)]
            logging.info('Summary with OpenAI: Step 2 - Summarizing article.')
            subtitles,summary_prompts = zip(*[assgin_prompts(self.summary_prompts,chunk[0]) for chunk in chunks])
            summary_prompts = [prompt+"```"+chunks[i][0]+"\n"+chunks[i][1]+"```" for i,prompt in enumerate(summary_prompts)]
            self.num_processes = min(self.num_processes,len(summary_prompts))
            if self.requests_per_minute:
                multi_resp = self.process_batches(summary_prompts, response_only=True, resest_messages=True)
            else:
                multi_resp = self.Multi_Chat_Processing(summary_prompts,response_only=True,resest_messages=True)
            titles = '' if titles is None else "title: " + titles + "\n"
            authors = '' if authors is None else "authors: " + authors + "\n"
            affiliations = '' if affiliations is None else "affiliations: " + affiliations + "\n"

            extra_info = ''
            for info in [titles, authors, affiliations]:
                if info:
                    extra_info += info
            print('='*100)
            print("multi_resp before filter:\n")
            for i,r in enumerate(multi_resp):
                print(f"multi_resp {i}:\n",r)
                print('-'*100)
            multi_resp = self.filter_final_response(multi_resp,raw_marker="Raw Summary Content",final_marker="Final Summary Content")
            multi_resp = ['\n## ' + subtitle + ':\n' + resp for (subtitle,resp) in zip(subtitles,multi_resp)]

            total_resp = extra_info + ''.join(multi_resp)
            logging.info('Summary with OpenAI: Step 3 - Resummarizing article.')
            # reset messages
            integrate_system = [self.resummry_prompts.get('integrate_system', ''),self.resummry_prompts.get('integrate', '')]
            score_system = [self.resummry_prompts.get('score_system', ''),self.resummry_prompts.get('score', '')]

            resummary_prompt = self.resummry_prompts.get('integrate_input', '').replace('{summary chunk}', total_resp, 1)
            score_prompt = self.resummry_prompts.get('score_input', '').replace('{article}', total_resp, 1).replace('{paper excerpt}', chunks[0][0]+ chunks[0][1] + '\n' + chunks[-1][0]+ chunks[-1][1], 1)
            prompts_l = [resummary_prompt, score_prompt]
            system_l = [integrate_system,score_system]

            filter_func = partial(self.filter_final_response,raw_marker="Raw Integrate Content",final_marker="Final Integrate Content")
            if self.requests_per_minute:
                re_respnse = self.process_batches(article_texts=prompts_l,
                                                  system_messages=system_l,
                                                  response_only=True, resest_messages=True)
            else:
                re_respnse = self.Multi_Chat_Processing(prompts_l,
                                                        system_messages=system_l,
                                                        response_only=True,resest_messages=True)
            table_resp = self.latex2md_table([table.table for table in tables])
            tables = [table+"\n\n\n"+tables[i].caption+"\n" for i,table in enumerate(table_resp)]
            print('='*100)
            print("before_filter,integrated res:\n",re_respnse[0])
            print('='*100)
            re_respnse = [filter_func(re_respnse[0]), self.clean_resp(re_respnse[1])]
            logging.info(f'Finished summarizing article, with the titles:{titles}')
            re_respnse.append(tables)
            return titles, authors, affiliations, total_resp,re_respnse

            # re_choice_l = []
            # for i in range(num_iterations):
            #     prompts_l = [resummary_prompt, score_prompt]
            #     if i == 0:
            #         filter_func = partial(self.filter_final_response,raw_marker="preliminary blog content",final_marker="final blog content")
            #     else:
            #         filter_func = partial(self.filter_final_response,raw_marker="Feedback content",final_marker="New Summary Content")
            #     if self.requests_per_minute:
            #         re_respnse = self.process_batches(prompts_l, response_only=True, resest_messages=True)
            #     else:
            #         re_respnse = self.Multi_Chat_Processing(prompts_l,response_only=True,resest_messages=True)
            #
            #     # re_respnse:[resummary,score]
            #     re_respnse = [filter_func(re_respnse[0]), self.clean_resp(re_respnse[1])]
            #     re_choice_l.append(re_respnse)
            #     resummary_prompt = self.resummry_prompts.get('iter', '').replace('{article}',total_resp,1).replace('{previous summary}',re_respnse[0],1)
            # logging.info(f'Finished summarizing article, with the titles:{titles}')
            # return titles, authors, affiliations, total_resp, re_choice_l

        # when sentence is too long, cut in half
        elif self.split_mode == 'half':
            if len(chunks) == 1:
                ## out range situation TODO
                summary_prompt = chunks[0] + self.summary_prompts['general_summary']
                response = self.chat_with_openai(summary_prompt,
                                                 reset_messages=True,
                                                 response_only=True)
                return response
            else:
                respons = []
                for i, chunk in enumerate(chunks):
                    input_prompt = chunk + self.summary_prompts['general_summary']
                    ## for each chunk, messages remain the same
                    resp = self.chat_with_openai(input_prompt,
                                                 reset_messages=True,
                                                 response_only=True)
                    respons.append(resp)
                self.init_messages('system', self.resummry_prompts['system'])
                total_respon = ''.join(respons)
                prompt = self.resummry_prompts['overview'] + total_respon
                response = self.chat_with_openai(prompt,
                                                 api_key=self.api_key,
                                                 reset_messages=True,
                                                 response_only=True)
                return response

    def Enhance_Answer(self,
                       original_answer:str,
                       summarized_answer:str,
                       usage:Literal['regenerate','blog','speech',None] = None
                       ):
        """
        Enhance the answer with openai api
        Args:
            original_answer: original answer
            summarized_answer: summarized answer
            usage: str, ['regenerate','blog','speech']
            num_iterations: int, number of iterations
        """
        system_messages = [self.resummry_prompts.get(f"{usage}_system", ''),self.resummry_prompts.get(f"{usage}", '')]
        enhance_prompts = self.resummry_prompts.get("enhance_input", '').replace('{article}', original_answer, 1).replace('{generated summary}', summarized_answer, 1)

        enhance_response = self.chat_with_openai(user_input=enhance_prompts,
                                                 system_messages=system_messages,
                                                 api_key=self.api_key,
                                                 reset_messages=True,
                                                 response_only=True)
        print('='*100)
        print("before_filter,enhance res:\n",enhance_response)
        print('='*100)
        enhance_response = self.filter_final_response(enhance_response,
                                                      raw_marker=f"Raw {usage.capitalize()} Content",
                                                      final_marker=f"New {usage.capitalize()} Content")
        return enhance_response

    def latex2md_table(self,tables:List[str]):

        self.init_messages('system', "you are a helpful assistant who are expert in translating latex table to markdown format")
        prompts = ["translate the following latex table to markdown format:\n" + table for table in tables]
        logging.info(f"Step 4 :Start to translate latex table to markdown format with {len(prompts)} prompts")
        if self.requests_per_minute:
            resp = self.process_batches(prompts, response_only=True, resest_messages=True)
        else:
            resp = self.Multi_Chat_Processing(prompts, response_only=True, resest_messages=True)
        markdown_pattern = re.compile(r"```markdown(.*)```|(\|.+\|)", re.DOTALL)

        for i,r in enumerate(resp):
            print("markdown_to_latex resp:\n",r)

        error_idx = []
        resps = []
        for i,r in enumerate(resp):
            match = re.search(markdown_pattern, r)
            if match:
                if match.group(1):
                    resps.append(match.group(1))
                else:
                    resps.append(match.group(2))
            else:
                resps.append(self.latex_table_to_markdown(tables[i]))
                error_idx.append(i)
        logging.info(f"latex2md_table error_idx:{error_idx}")
        return resps
    @staticmethod
    def clean_content(content):
        # Remove all \hline
        clean = re.sub(r'\\hline', '', content)

        # Remove leading and trailing non-alphabetic characters
        clean = re.sub(r'^\W+|\W+$', '', clean)

        return clean

    def latex_table_to_markdown(self,latex_table):
        # 匹配LaTeX表格内容
        match = re.search(r'\\begin{tabular}(.*?)\\hline(.*?)\\\\(.*?)\\end{tabular}', latex_table, re.DOTALL)

        if not match:
            return "Invalid LaTeX table"

        # 获取表格列数和内容
        column_definition = match.group(2)
        table_content = match.group(3)
        column_definition = self.clean_content(column_definition)
        # 解析表格列定义
        columns = [col.strip() for col in column_definition.split('&') if col.strip()]
        table_content = self.clean_content(table_content)

        # 解析表格内容
        table_data = [line.split('&') for line in table_content.strip().split('\\\\')]

        # 替换\multirow
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                cell_content = table_data[i][j].strip()
                match_multirow = re.match(r'\\multirow{(\d+)}{(.*?)}', cell_content)
                if match_multirow:
                    # 获取\multirow的行数和内容
                    num_rows = int(match_multirow.group(1))
                    multirow_content = match_multirow.group(2)
                    # 在后续行中插入空白单元格
                    for k in range(1, num_rows):
                        if i + k < len(table_data):
                            table_data[i + k].insert(j, '')
                        else:
                            # 如果超出了表格行数，添加新行
                            table_data.append([''] * len(columns))
                            table_data[i + k][j] = ''
                    table_data[i][j] = multirow_content

        # 生成Markdown表格
        markdown_table = tabulate(table_data, headers=columns, tablefmt="pipe")

        return markdown_table














if __name__ == "__main__":
    openai_key = "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"
    # init_openai(host="http://localhost:7890",api_key=openai_key)
    # resps = []
    # openai_info = {'api_key': openai_key}
    # proxy_info = {'host': 'http://localhost:7890'}
    # system_messages = 'you are a helpful assistant who are great in making plans'
    # article_text = "Today is a nice day,can you help me write some plans to kill time?"
    # summarizer = OpenAI_Summarizer(openai_info,proxy_info,split_mode='group',summary_system=system_messages,resummry_system=system_messages,
    #                                summary_prompts=
    # summarizer.init_openai(host=proxy_info['host'],api_key=openai_info['api_key'])
    # messages = init_messages('system',system_messages)
    #
    # resp = chat_with_openai(article_text,messages,response_only=True,reset_messages= False)
    # resps.append(resp)
    # print('1: messages:',messages)
    # article_text2 = """could you please list why you list these plans and their pros and cons?"""
    # resp = chat_with_openai(article_text2,messages,response_only=True,reset_messages = False)
    # print('2: messages:',messages)
    # resps.append(resp)
    # for res in resps:
    #     print(f'res:{res}')




