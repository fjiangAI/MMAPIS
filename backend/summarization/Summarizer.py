from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.summarization.SectionSummarizer import SectionSummarizer
from MMAPIS.backend.summarization.DocumentSummarizer import DocumentSummarizer
from MMAPIS.backend.data_structure.Article import Article


class Summarizer():
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_config: dict = None,
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 # the ratio of prompt tokens to max tokens,i.e. length of prompt tokens/length of max tokens
                 rpm_limit: int = 3,  # if api_key is limited by 3 times/min, set this to 3, if no limit, set this to 0
                 num_processes: int = 6,
                 ignore_titles=[],
                 ):
        self.section_summarizer = SectionSummarizer(api_key=api_key,
                                                    base_url=base_url,
                                                    model_config=model_config,
                                                    proxy=proxy,
                                                    prompt_ratio=prompt_ratio,
                                                    rpm_limit=rpm_limit,
                                                    num_processes=num_processes,
                                                    ignore_titles=ignore_titles,
                                                    )
        self.document_summarizer = DocumentSummarizer(
            api_key=api_key,
            base_url=base_url,
            model_config=model_config,
            proxy=proxy,
            prompt_ratio=prompt_ratio,
        )

    def generate_section_level_summary(self,
                                 text: str,
                                 prompt: dict,
                                 file_name: str = "",
                                 min_grained_level: int = 2,
                                 max_grained_level: int = 4,
                            ):
        return  self.section_summarizer.section_summarize(raw_md_text=text,
                                                          file_name=file_name,
                                                          summary_prompts=prompt,
                                                          min_grained_level=min_grained_level,
                                                          max_grained_level=max_grained_level)


    def generate_document_level_summary(self,
                                  section_level_summary: str,
                                  system_prompts: dict,
                                  ):
        return self.document_summarizer.integrate_summary(
            section_level_summary= section_level_summary,
            document_prompts=system_prompts,
        )

    def generate_summary(self,
                         text: str,
                         section_prompts: dict,
                         document_prompts: dict,
                         file_name:str = "",
                         min_grained_level:int = 2,
                         max_grained_level:int = 4
                         ):
        section_flag, section_level_summary = self.generate_section_level_summary(text,
                                                                        section_prompts,
                                                                        file_name=file_name,
                                                                        min_grained_level=min_grained_level,
                                                                        max_grained_level=max_grained_level)
        if section_flag:
            document_flag, document_level_summary = self.generate_document_level_summary(section_level_summary, document_prompts)
            if document_flag:
                return True, {
                    "section_level_summary": section_level_summary,
                    "document_level_summary":document_level_summary
                }
            else:
                return False, f"Document-level summary generation error:{document_level_summary}"
        else:
            return False, f"Section-level summary generation error:{section_level_summary}"
