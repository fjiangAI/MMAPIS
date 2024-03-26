from MMAPIS.tools.chatgpt import GPT_Helper
from MMAPIS.server.summarization.section_split import sectional_split,assgin_prompts
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import Union, List
import logging
from multiprocessing.pool import ThreadPool as Pool
import time
import re
from MMAPIS.config.config import OPENAI_CONFIG,GENERAL_CONFIG,LOGGER_MODES,SECTION_PROMPTS
from MMAPIS.tools.utils import init_logging,strip_title



class Section_Summarizer(GPT_Helper):
    def __init__(self,
                 api_key: str ,
                 base_url: str,
                 model_config: dict = {},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 # the ratio of prompt tokens to max tokens,i.e. length of prompt tokens/length of max tokens
                 rpm_limit: int = 3,  # if api_key is limited by 3 times/min, set this to 3, if no limit, set this to 0
                 num_processes: int = 6,
                 ignore_titles=[],
                 ):
        super().__init__(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio)
        self.ignore_titles = ignore_titles
        self.rpm_limit = rpm_limit if rpm_limit > 0 else float("inf")
        self.num_processes = min(self.rpm_limit,multiprocessing.cpu_count(),num_processes)


    def section_split(self,article_text:str,
                      file_name:str = "",
                      init_grid=2,max_grid=4):
        return sectional_split(article_text, file_name, ignore_title=self.ignore_titles,max_tokens=self.max_tokens,init_grid=init_grid,max_grid=max_grid)

    def section_summarize(self,
                          article_text:str,
                          file_name:str = "",
                          summary_prompts: Union[dict, str] = None,
                          init_grid=2,
                          max_grid=4):
        self.init_messages('system', summary_prompts.get('system',''))
        article = self.section_split(article_text,file_name,init_grid,max_grid)
        sections = article.sections
        subtitles, summary_prompts = zip(*[assgin_prompts(summary_prompts, section.parent) for section in sections])
        ignore_idx = []
        temp = []
        for i,prompt in enumerate(summary_prompts):
            if sections[i].text:
                temp.append(prompt + "\n\n```" + sections[i].title + "\n" + sections[i].text + "```")
                ignore_idx.append(False)
            else:
                temp.append('')
                ignore_idx.append(True)
        summary_prompts = temp
        # summary_prompts = [(prompt + "\n\n```" + sections[i].title + "\n" + sections[i].text + "```",False) for i, prompt in
        #                    enumerate(summary_prompts) if sections[i].text else (sections[i].title,True)]
        self.num_processes = min(self.num_processes, len(summary_prompts))
        if float("inf") >self.rpm_limit > 0:
            logging.info("Due to openai api rate limit, use batch processing")
            flag, multi_resp = self.batch_processing(summary_prompts, response_only=True, resest_messages=True, ignore_idx=ignore_idx)
        else:
            logging.info("No openai api rate limit, use multi processing")
            flag, multi_resp = self.multi_chat_processing(summary_prompts, response_only=True, resest_messages=True, ignore_idx=ignore_idx)
        if flag:
            multi_resp = self.filter_final_response(multi_resp, raw_marker="Raw Summary Content",
                                                    final_marker="Final Summary Content")

            subtitles = [strip_title(section.title) for section in article.sections]
            text_summary = ''
            for i,(subtitle, resp) in enumerate(zip(subtitles, multi_resp)):
                text_summary += '#' * sections[i].rank + ' ' + strip_title(subtitle) + '\n' + resp + '\n'
            res = article.extra_info + '\n' + text_summary
            return flag, res
        else:
            return flag, multi_resp



    def multi_chat_processing(self,article_texts:Union[List[str],str],
                              system_messages:Union[str,List[str],List[List]] = None,
                              response_only:bool = True,
                              resest_messages:bool = True,
                              ignore_idx:List[bool] = None):
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
        with Pool(processes=self.num_processes) as pool:
            chat_func = partial(self.summarize_text,
                                response_only=response_only,
                                reset_messages=resest_messages)
            article_texts = [article_texts] if isinstance(article_texts,str) else article_texts
            if system_messages and isinstance(system_messages[0], List):
                if len(system_messages) != len(article_texts):
                    return False, f"system_messages should be list of length {len(article_texts)}, but got {len(system_messages)}"
                flag = True
            if not ignore_idx:
                ignore_idx = [False] * len(article_texts)

            article_texts = tqdm(article_texts,position=0,leave=True)
            article_texts.set_description(f"[Section Summary] Total {len(article_texts)} section | num_processes:{self.num_processes} | requests_per_minute:{self.rpm_limit}")
            try:
                results = [
                    pool.apply_async(chat_func,kwds={'system_messages':system_messages[i] if flag else system_messages,
                                                     'text':article_text,
                                                     'return_the_same':ignore_idx[i],
                                                    })
                    for i,article_text in enumerate(article_texts)
                ]
            except Exception as e:
                error_msg = f"Multi chat processing failed with error {e}"
                logging.error(error_msg)
                return False, error_msg
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        success = all([r[0] for r in results])
        if success:
            results = [r[1] for r in results]
            return success,results
        else:
            for i,result in enumerate(results):
                if not result[0]:
                    logging.error(f"Failed to summarize section {i} with error {result[1]}")
                    return success,result[1]



    def batch_processing(self, article_texts:Union[str,List],
                        system_messages:Union[str,List[str],List[List]] = None,
                        response_only:bool = True,
                        resest_messages:bool = True,
                        ignore_idx:List[bool] = None):
        """
        Due to openai key rate limit, process batches of article texts and sleep for a while after each batch
        Args:
            article_texts:
            response_only:
            resest_messages:
            system_messages:when system_messages is list of list, e.g. message:[{"system":"xxx"},{"system":"xxx"},{"user":"xxx"}],where system is a list

        Returns: list of results

        """


        article_texts = [article_texts] if isinstance(article_texts,str) else article_texts
        flag = False
        if system_messages and isinstance(system_messages[0],List):
            assert len(system_messages) == len(article_texts),logging.error(f"system_messages should be list of length {len(article_texts)}, but got {len(system_messages)}")
            flag = True

        results = []
        chat_func = partial(self.summarize_text,
                            reset_messages=resest_messages,
                            response_only=response_only)

        with Pool(processes=self.num_processes) as pool:
            processing_bar = tqdm(range(0, len(article_texts), self.num_processes))
            if not ignore_idx:
                ignore_idx = [False] * len(article_texts)
            for i in processing_bar:
                processing_bar.set_description(
                    f"Processing {i}~{min(i + self.num_processes,len(article_texts))} | total {len(article_texts)} sections | num_processes:{self.num_processes} | requests_per_minute:{self.rpm_limit}")
                start_time = time.time()
                batches = article_texts[i:i + self.num_processes]
                batch_system_messages = system_messages[i:i + self.num_processes] if flag else system_messages
                try:
                    futures = [pool.apply(chat_func,kwds={'system_messages':batch_system_messages[i] if flag else batch_system_messages,
                                                         'text':batch,
                                                         'return_the_same':ignore_idx[i],
                                                         }
                                            ) for i,batch in enumerate(batches)]
                except Exception as e:
                    error_msg = f"Batch processing failed with error {e}"
                    logging.error(error_msg)
                    return False,error_msg
                results.extend(futures)
                elapsed_time = time.time() - start_time
                if i + self.num_processes >= len(article_texts):
                    break
                elif elapsed_time < 60:
                    slp_t = int(60 - elapsed_time) + 3
                    logging.info(f'due to rate limit, sleep for {slp_t}s')
                    time.sleep(slp_t)  # wait until 60s
        success = all([r[0] for r in results])
        if success:
            results = [r[1] for r in results]
            return success,results
        else:
            for i,result in enumerate(results):
                if not result[0]:
                    logging.error(f"Failed to summarize section {i} with error {result[1]}")
                    return success,result[1]







if __name__ == "__main__":
    logger = init_logging(logger_mode=LOGGER_MODES)
    section_summarizer = Section_Summarizer(api_key=OPENAI_CONFIG["api_key"],
                                            base_url=OPENAI_CONFIG["base_url"],
                                            model_config=OPENAI_CONFIG["model_config"],
                                            proxy=GENERAL_CONFIG["proxy"],
                                            prompt_ratio=OPENAI_CONFIG["prompt_ratio"],
                                            rpm_limit=OPENAI_CONFIG["rpm_limit"],
                                            num_processes=OPENAI_CONFIG["num_processes"],
                                            ignore_titles=OPENAI_CONFIG["ignore_title"],
                                            )
    article_path = "2403_08777.md"
    with open(article_path,"r",encoding="utf-8") as f:
        article_text = f.read()

    flag , article = section_summarizer.section_summarize(article_text,
                                                   file_name="Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper",
                                                   summary_prompts=SECTION_PROMPTS,
                                                   init_grid=2,
                                                   max_grid=4)


    with open("summary.md","w",encoding="utf-8") as f:
        f.write(article)

