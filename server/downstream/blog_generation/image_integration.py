import os.path
from MMAPIS.server.summarization import Article
from MMAPIS.server.downstream.blog_generation.blog_script import  Blog_Script_Generator
from MMAPIS.server.preprocessing import img_txt_alignment
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG
from typing import Union, List
from pathlib import Path

class Blog_Generator():
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                    **kwargs):
        self.blog_script_generator = Blog_Script_Generator(api_key=api_key,
                                                           base_url=base_url,
                                                           model_config=model_config,
                                                           proxy=proxy,
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

if __name__ == "__main__":
    api_key = OPENAI_CONFIG['api_key']
    base_url = OPENAI_CONFIG['base_url']
    model_config = OPENAI_CONFIG['model_config']
    blog_prompts = APPLICATION_PROMPTS['blog_prompts']
    integration_path = "2402_03753/2402_03753_integrated.md"
    with open(integration_path, 'r') as f:
        integration = f.read()
    section_summaries_path = "2402_03753/2402_03753.md"
    with open(section_summaries_path, 'r') as f:
        section_summaries = f.read()
    print(Article(section_summaries).extra_info)
    pdf = "./2402_03753/2402_03753.pdf"
    # blog_generator = Blog_Generator(api_key=api_key,
    #                                base_url=base_url,
    #                                model_config=model_config,
    #                                 proxy=GENERAL_CONFIG['proxy'])
    # flag, response = blog_generator.blog_generation(pdf=pdf,
    #                                                document_level_summary=integration,
    #                                                section_summaries=section_summaries,
    #                                                blog_prompts=blog_prompts,
    #                                                 save_dir="./res/")
    # print(response)
    # print(flag)

