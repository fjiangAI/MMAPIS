import os.path
from MMAPIS.server.summarization import Article
from MMAPIS.server.downstream.blog_generation.blog_script import  Blog_Script_Generator
from MMAPIS.server.preprocessing import img_txt_alignment
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG
from typing import Union, List
from pathlib import Path
import reprlib

class Blog_Generator():
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        self.blog_script_generator = Blog_Script_Generator(api_key=api_key,
                                                           base_url=base_url,
                                                           model_config=model_config,
                                                           proxy=proxy,
                                                           prompt_ratio=prompt_ratio,
                                                           **kwargs)

    def blog_generation(self,
                        pdf:Union[str, Path],
                        document_level_summary: str,
                        section_summaries: Union[str, List[str]],
                        raw_md_text: str=None,
                        blog_prompts: dict = APPLICATION_PROMPTS['blog_prompts'],
                        reset_messages: bool = True,
                        response_only: bool = True,
                        raw_marker: str = "Raw Blog Content",
                        final_marker: str = "New Blog Content",
                        file_name: str = None,
                        save_dir: str = GENERAL_CONFIG['save_dir'],
                        threshold: float = ALIGNMENT_CONFIG['threshold'],
                        init_grid: int = ALIGNMENT_CONFIG['init_grid'],
                        max_grid: int = ALIGNMENT_CONFIG['max_grid'],
                        img_width: int = ALIGNMENT_CONFIG['img_width'],
                        temp_file: bool = False,
                        **kwargs):
        flag, content = self.blog_script_generator.blog_script_generation(document_level_summary=document_level_summary,
                                                                            section_summaries=section_summaries,
                                                                            blog_prompts=blog_prompts,
                                                                            reset_messages=reset_messages,
                                                                            response_only=response_only,
                                                                            raw_marker=raw_marker,
                                                                            final_marker=final_marker,
                                                                            **kwargs)
        content = "# Blog Content\n" + content
        path = img_txt_alignment(text=content,
                                pdf=pdf,
                                raw_md_text=raw_md_text,
                                file_name=file_name,
                                save_dir=save_dir,
                                threshold=threshold,
                                init_grid=init_grid,
                                max_grid=max_grid,
                                img_width=img_width,
                                 temp_file= temp_file)
        return flag, path

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Blog_GPT(api_key:{reprlib.repr(self.blog_script_generator.api_key)},base_url:{self.blog_script_generator.base_url},model:{self.blog_script_generator.model}, temperature:{self.blog_script_generator.temperature}, max_tokens:{self.blog_script_generator.max_tokens}, top_p:{self.blog_script_generator.top_p}, frequency_penalty:{self.blog_script_generator.frequency_penalty}, presence_penalty:{self.blog_script_generator.presence_penalty})"
        return msg



