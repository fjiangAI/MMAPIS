from MMAPIS.tools import init_logging,GPT_Helper

class User_Intent(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, **kwargs)

    def get_intend(self,
                   user_input:str,
                   system_messages:Union[str,List[str]] = "",
                   response_only:bool = True,
                   reset_messages:bool = True):
        self.init_messages("system", system_messages)
        flag,content = self.request_api(user_input=user_input,
                                        reset_messages=reset_messages,
                                        response_only=response_only)
        return flag,content




