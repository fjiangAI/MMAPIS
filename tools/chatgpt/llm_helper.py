import requests
import json

class LLMSummarizer:
    def __init__(self, api_key, base_url, model_config={},proxy=None,prompt_ratio:float = 0.8, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.settings = kwargs
        self.init_model(model_config)
        self.proxy = proxy
        self.prompt_ratio = prompt_ratio
        self.messages = []

    def init_model(self, model_config):
        default_config = {
           'model': 'gpt-3.5-turbo-16k-0613',
           'temperature': 0.9,
           'max_tokens': 16385,
           'top_p': 1,
           'frequency_penalty': 0.1,
           'presence_penalty': 0.2
        }
        default_config.update(model_config)
        self.model = default_config['model']
        self.temperature = default_config['temperature']
        self.max_tokens = default_config['max_tokens']
        self.top_p = default_config['top_p']
        self.frequency_penalty = default_config['frequency_penalty']
        self.presence_penalty = default_config['presence_penalty']
        # for key,value in default_config.items():
        #     setattr(self, key, value)

    def request_api(self,**kwargs):
        pass

    def summarize_text(self,text:str,
                       system_messages,
                       **kwargs):
        raise NotImplementedError("Method not implemented")

    def handle_request(self,url:str,parameters = None,headers = None):
        success = False
        response = None
        try:
            raw_response = requests.post(url, json=parameters, proxies=self.proxy, headers=headers)
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