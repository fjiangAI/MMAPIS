from MMAPIS.backend.tools.chatgpt import GPTHelper
import multiprocessing
from functools import partial
from tqdm import tqdm
from typing import Union, List
import logging
from multiprocessing.pool import ThreadPool as Pool
import time
import re
from MMAPIS.backend.data_structure.Article import Article
from MMAPIS.backend.config.config import LOGGER_MODES, OPENAI_CONFIG,SECTION_PROMPTS
from MMAPIS.backend.tools.utils import init_logging



class SectionSummarizer:
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
        self.summarizer = GPTHelper(api_key=api_key,
                                    base_url=base_url,
                                    model_config=model_config,
                                    proxy=proxy,
                                    prompt_ratio=prompt_ratio,
        )
        self.ignore_titles = ignore_titles
        self.rpm_limit = rpm_limit if rpm_limit > 0 else float("inf")
        self.num_processes = min(self.rpm_limit,multiprocessing.cpu_count(),num_processes)


    def section_split(self,
                      raw_md_text:str,
                      file_name:str = "",
                      min_grained_level=2,
                      max_grained_level=4):
        return self.sectional_split(raw_md_text,
                                    file_name,
                                    ignore_title=self.ignore_titles,
                                    max_tokens=self.summarizer.max_tokens,
                                    min_grained_level=min_grained_level,
                                    max_grained_level=max_grained_level)

    def section_summarize(self,
                          raw_md_text: str,
                          file_name: str = "",
                          summary_prompts: Union[dict, str] = None,
                          min_grained_level=2,
                          max_grained_level=4):
        """
        Summarizes sections of an article using the OpenAI API.

        Args:
            raw_md_text (str): The full text of the article to be summarized.
            file_name (str, optional): The name of the file containing the article. Defaults to "".
            summary_prompts (Union[dict, str], optional): Prompts used to guide the summary generation.
                                                          If a dictionary, the keys should correspond to
                                                          sections; if a string, it applies to all sections.
                                                          Defaults to None.
            min_grained_level (int, optional): Minimum granularity level for section splitting. Defaults to 2.
            max_grained_level (int, optional): Maximum granularity level for section splitting. Defaults to 4.

        Returns:
            tuple: A tuple where the first element is a boolean indicating success, and the second element
                   is the summary text or an error message.
        """
        # Split the article into sections based on the specified granularity levels
        article = self.section_split(raw_md_text, file_name,
                                     min_grained_level=min_grained_level,
                                     max_grained_level=max_grained_level)
        sections = article.sections
        self.summarizer.init_messages('system', summary_prompts.get('system', ''))
        # Initialize system messages for the summarizer
        self.num_processes = min(self.num_processes, len(summary_prompts))


        # Assign prompts to each section based on the section's parent
        summary_prompts = [self.assign_prompts(summary_prompts, section.parent) for section in sections]

        # Prepare the summary prompts for sections that contain text
        ignore_idx = []
        formatted_prompts = []
        for i, prompt in enumerate(summary_prompts):
            if sections[i].text:
                formatted_prompts.append(f"{prompt}\n\n```{sections[i].title_text}```")
                ignore_idx.append(False)
            else:
                formatted_prompts.append('')
                ignore_idx.append(True)
        summary_prompts = formatted_prompts



        # Determine the appropriate processing method based on API rate limits
        if float("inf") > self.rpm_limit > 0:
            logging.info("API rate limit detected, utilizing batch processing.")
            flag, multi_resp = self.batch_processing(summary_prompts, response_only=True,
                                                     resest_messages=True, ignore_idx=ignore_idx)
        else:
            logging.info("No API rate limit detected, utilizing multiprocessing.")
            flag, multi_resp = self.multi_chat_processing(summary_prompts, response_only=True,
                                                          resest_messages=True, ignore_idx=ignore_idx)

        if flag:
            # Filter and format the final summary response
            multi_resp = self.summarizer.filter_final_response(multi_resp,
                                                               raw_marker="Raw Summary Content",
                                                               final_marker="Final Summary Content")
            text_summary = ''
            for i, resp in enumerate(multi_resp):
                text_summary += sections[i].title + '\n' + resp + '\n' if not resp.startswith("#") else resp
            res = article.extra_info + '\n' + text_summary
            return flag, self.summarizer.format_headers(self.summarizer.clean_math_text(res))
        else:
            return flag, multi_resp

    def multi_chat_processing(self,
                              article_texts: Union[List[str], str],
                              system_messages: Union[str, List[str], List[List]] = None,
                              response_only: bool = True,
                              resest_messages: bool = True,
                              ignore_idx: List[bool] = None):
        """
        Engages the OpenAI API in parallel processing when API rate limits are not a concern.

        Args:
            article_texts (Union[List[str], str]): List of article texts or a single article text.
            response_only (bool, optional): If True, only return the response content; otherwise, return messages.
                                            Defaults to True.
            resest_messages (bool, optional): If True, reset messages to the system; otherwise, append messages.
                                              Defaults to True.
            system_messages (Union[str, List[str], List[List]], optional): System messages to guide the summary generation.
                                                                           Can be None, a single string, a list of strings,
                                                                           or a list of lists of messages. Defaults to None.

        Returns:
            tuple: A tuple where the first element is a boolean indicating success, and the second element
                   is the response content or an error message.
        """

        flag = False
        with Pool(processes=self.num_processes) as pool:
            chat_func = partial(self.summarizer.request_text_api,
                                response_only=response_only,
                                reset_messages=resest_messages)
            article_texts = [article_texts] if isinstance(article_texts, str) else article_texts

            # Check if system_messages is a list of lists and validate its length
            if system_messages and isinstance(system_messages[0], List):
                if len(system_messages) != len(article_texts):
                    return False, f"system_messages should be a list of length {len(article_texts)}, but got {len(system_messages)}"
                flag = True

            # Initialize ignore_idx if not provided
            if not ignore_idx:
                ignore_idx = [False] * len(article_texts)

            # Execute chat processing with a progress bar
            article_texts = tqdm(article_texts, position=0, leave=True)
            article_texts.set_description(
                f"[Section Summary] Total {len(article_texts)} sections | num_processes: {self.num_processes} | model: {self.summarizer.model} | max_tokens: {self.summarizer.max_tokens} | requests_per_minute: {self.rpm_limit}")

            try:
                results = [
                    pool.apply_async(chat_func,
                                     kwds={'system_messages': system_messages[i] if flag else system_messages,
                                           'user_input': article_text,
                                           'return_the_same': ignore_idx[i]})
                    for i, article_text in enumerate(article_texts)
                ]
            except Exception as e:
                error_msg = f"Multi chat processing failed with error: {e}"
                logging.error(error_msg)
                return False, error_msg

            pool.close()
            pool.join()

            # Gather results and check for overall success
            results = [p.get() for p in results]
        success = all([r[0] for r in results])
        if success:
            results = [r[1] for r in results]
            return success, results
        else:
            for i, result in enumerate(results):
                if not result[0]:
                    logging.error(f"Failed to summarize section {i} with error: {result[1]}")
                    return success, result[1]

    def batch_processing(self, article_texts: Union[str, List],
                         system_messages: Union[str, List[str], List[List]] = None,
                         response_only: bool = True,
                         resest_messages: bool = True,
                         ignore_idx: List[bool] = None):
        """
        Processes batches of article texts due to OpenAI API rate limits, with periodic sleep intervals.

        Args:
            article_texts (Union[str, List]): Article texts to be processed.
            response_only (bool, optional): If True, only return the response content; otherwise, return messages.
                                            Defaults to True.
            resest_messages (bool, optional): If True, reset messages to the system; otherwise, append messages.
                                              Defaults to True.
            system_messages (Union[str, List[str], List[List]], optional): System messages for summary generation.
                                                                           Can be None, a single string, a list of strings,
                                                                           or a list of lists of messages. Defaults to None.

        Returns:
            tuple: A tuple where the first element is a boolean indicating success, and the second element
                   is the response content or an error message.
        """
        article_texts = [article_texts] if isinstance(article_texts, str) else article_texts
        flag = False

        # Validate the length of system_messages if it is a list of lists
        if system_messages and isinstance(system_messages[0], List):
            assert len(system_messages) == len(article_texts), logging.error(
                f"system_messages should be a list of length {len(article_texts)}, but got {len(system_messages)}")
            flag = True

        results = []
        chat_func = partial(self.summarizer.request_text_api,
                            reset_messages=resest_messages,
                            response_only=response_only)

        # Execute batch processing with sleep intervals to respect API rate limits
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
                                                         'user_input':batch,
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


    @staticmethod
    def assign_prompts(
            summary_prompts: dict,
            query_title: str,
    )->str:
        """
        This method assigns a prompt based on the query title and a dictionary of summary prompts.

        :param summary_prompts: A dictionary where keys are prompt identifiers and values are prompt templates.
        :param query_title: The title of the query that needs a corresponding prompt.
        :return: A tuple containing the modified query title and the corresponding prompt.
        """

        # Iterates through the keys in summary_prompts to find a matching key in the query_title.
        for key in summary_prompts.keys():
            if key in query_title.lower():
                break

        # Checks if no matching key was found. If so, assigns the default prompt, i.e., the 'general' prompt.
        if 'general' in key:
            # Extracts the actual title using a regular expression pattern.
            title_pattern = re.compile(r"#+\s*(.*)")
            match = re.match(title_pattern, query_title)
            query_title = match.group(1) if match else ""

            # Replaces the placeholder in the prompt template with the query title if replace is True.
            prompt = summary_prompts[key].replace('[title_to_replace]', query_title)
            return prompt
        else:
            # Strips the title if replace is True; otherwise, returns the key and the corresponding prompt.
            return summary_prompts.get(key, '')

    @staticmethod
    def sectional_split(text,
                        file_name,
                        min_grained_level=3,
                        ignore_title=None,
                        max_tokens=16385,
                        max_grained_level=4):
        """
        Splits text into chunks of at most max_tokens tokens.
        :param text: str
        :param max_tokens: int
        :return: list of str
        """

        # groups_info : [title,text,length]
        article = Article(text, file_name=file_name, min_grained_level=min_grained_level, ignore_title=ignore_title, max_grained_level=max_grained_level)
        while any([section.length > max_tokens for section in article.sections]):
            min_grained_level += 1
            article = Article(text, file_name=file_name, min_grained_level=min_grained_level, ignore_title=ignore_title)
            # set max grid(The most granular heading level) to 4
            if min_grained_level >= max_grained_level:
                logging.info(f'grid is up to {max_grained_level},chunking into {max_tokens} tokens')
                break
        logging.debug(
            f'finish split in split grained:{min_grained_level} (i.e. {min_grained_level * "#"}), with {len(article.sections)} sections, max section length:{max([section.length for section in article.sections])}')
        # in case grid > 2, return subsection_titles
        return article








