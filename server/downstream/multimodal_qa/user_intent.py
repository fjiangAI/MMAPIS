from MMAPIS.tools import init_logging,GPT_Helper
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS

class UserIntent(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, **kwargs)

    def get_intend(self,
                   prompts: dict,
                   user_input:str,
                   title_list:list,
                   response_only:bool = True,
                   reset_messages:bool = True):
        self.init_messages("system", [prompts.get('qa_system',''),prompts.get('qa', '')])
        user_input = prompts.get('qa_input','').replace('{user query}',user_input,1).replace('{title list}',str(title_list),1)
        flag,content = self.request_api(user_input=user_input,
                                        reset_messages=reset_messages,
                                        response_only=response_only)
        return flag,content



if __name__ == "__main__":

    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    qa_prompts = APPLICATION_PROMPTS["multimodal_qa"]
    user_intent = UserIntent(api_key, base_url, model_config)
    print(user_intent)
    user_input = "What is the 0th picture about in the article?"
    title_list = ["Introduction", "Background of QA", "Proposed Method", "Eval & Results", "Conclusion"]
    flag,content = user_intent.get_intend(qa_prompts,user_input,title_list)
    print(content)
    print(type(eval(content)))



