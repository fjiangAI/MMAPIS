from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.config.config import CONFIG, APPLICATION_PROMPTS,OPENAI_CONFIG
import reprlib


class ImgQAGenerator(GPTHelper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = None,
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)
        self.check_model(model_type="img")

    def __repr__(self):
        return_str = []
        for key, value in self.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"ImgQAGenerator({', '.join(return_str)})"




