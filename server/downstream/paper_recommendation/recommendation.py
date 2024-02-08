from MMAPIS.tools import  GPT_Helper,init_logging
from MMAPIS.config.config import CONFIG,APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.server.summarization import Article
from typing import Union,List

class Paper_Recommender(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                 **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         **kwargs)

    def recommendation_generation(self,
                                  document_level_summary: str,
                                  article: Union[str, List[str], Article],
                                  score_prompts: dict,
                                  reset_messages: bool = True,
                                  response_only: bool = True,
                                  **kwargs):

        if isinstance(article, List):
            article_segment = "\n".join([article[0],article[-1]])

        else:
            if isinstance(article, Article):
                article_segment = [article.sections[0].title_text, article.sections[-1].title_text]
            else:
                article = Article(article)
                article_segment = [article.sections[0].title_text, article.sections[-1].title_text]
            article_segment = "\n".join(article_segment)
        score_system = [score_prompts.get('score_system', ''), score_prompts.get('score', '')]
        score_prompt = score_prompts.get('score_input', '').replace('{article}', document_level_summary, 1).replace(
            '{paper excerpt}', article_segment, 1)

        self.init_messages("system",score_system)
        return self.summarize_text(score_prompt, reset_messages=reset_messages, response_only=response_only, **kwargs)

    def __repr__(self):
        return f"Paper_Recommender(api_key={self.api_key},base_url={self.base_url},proxy={self.proxy}),model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"


if __name__ == "__main__":
    logger = init_logging(LOGGER_MODES)
    logger.info("Initializing Paper Recommender")
    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    recommendation_prompts = APPLICATION_PROMPTS["score_prompts"]
    paper_recommender = Paper_Recommender(api_key=api_key, base_url=base_url, model_config=model_config)
    print("paper_recommender:",paper_recommender)
    article_path = "../raw.mmd"
    with open(article_path, 'r') as f:
        article = f.read()
    document_level_summary_path = "../integrate.md"
    with open(document_level_summary_path, 'r') as f:
        document_level_summary = f.read()

    flag, content = paper_recommender.recommendation_generation(document_level_summary=document_level_summary,
                                                                article=article,
                                                                score_prompts=recommendation_prompts,
                                                                reset_messages=True,
                                                                response_only=True)
    print(flag)
    print("-"*20)
    print(content)







