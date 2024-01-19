import spacy
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams
from nltk import word_tokenize
from typing import Literal
from .utils import  num_tokens_from_messages,handle_request
import pandas as pd
import logging

class Evaluator():
    def __init__(self, pred,gold = None):
        self.gold = gold
        self.pred = pred

    def __str__(self):
        return f"gold: {self.gold}\npred: {self.pred}"

    def __repr__(self):
        return f"Evaluator(gold={self.gold}, pred={self.pred})"


    def entity_density_eval(self,lang=Literal["en","zh"],show_result=False):
        """
        Compute the entity density of the prediction
        """
        if lang == "en":
            nlp = spacy.load("en_core_web_sm")
        elif lang == "zh":
            nlp = spacy.load("zh_core_web_sm")
        else:
            raise ValueError(f"lang must be one of ['en','zh'],but got {self.lang}")
        doc = nlp(self.pred)
        unique_ents = set([ent.text for ent in doc.ents])
        for ent in doc.ents:
            if show_result:
                print(f'{ent.text:<15}  {ent.start_char:<10}  {ent.end_char:<10}  {ent.label_}')
            unique_ents.add(ent.text)
        # the same entity will be counted only once,e.g. "Steve Jobs" will be counted as 1
        return len(unique_ents) / len(doc),len(unique_ents),len(doc)

    def GPT_4_eval(self,
                   prompts:dict,
                   api_key:str,
                   base_url="https://api.ai-gaochao.cn/v1",
                   model_dict:dict = {
                        "model": "gpt-4",
                        "temperature": 1,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "max_tokens": 8192,
                    }
                   ):
        """
        Compute the GPT-4 score of the prediction, where gold is GPT-4 summary and pred is the generated summary
        """
        system = prompts['system']
        message = [{'role': "system", 'content': system}]
        evaluation = prompts['evaluation'].replace('gold', self.gold).replace('pred', self.pred)
        message.append({'role': 'user', 'content': evaluation})
        input_tokens = num_tokens_from_messages(message)

        # get response content
        url = base_url + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        parameters = {
            "model" : model_dict['model'],
            "messages": message,
            "max_tokens": model_dict['max_tokens'] - input_tokens,
            "temperature": model_dict['temperature'],
            "top_p": model_dict['top_p'],
            "frequency_penalty": model_dict['frequency_penalty'],
            "presence_penalty": model_dict['presence_penalty'],
        }
        response,_, flag = handle_request(url = url,parameters=parameters,headers = headers)

        return flag, response['choices'][0]['message']['content']





    def BLEU_eval(self):
        """
        Compute the BLEU score of the prediction
        """
        return sentence_bleu(self.gold, self.pred)


    def rouge_n_eval(self,n):
        reference_ngrams = list(ngrams(word_tokenize(self.gold), n))
        candidate_ngrams = list(ngrams(word_tokenize(self.pred), n))

        overlapping_ngrams = set(reference_ngrams) & set(candidate_ngrams)
        recall = len(overlapping_ngrams) / len(reference_ngrams)
        precision = len(overlapping_ngrams) / len(candidate_ngrams)

        rouge_n_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return rouge_n_score

    def rouge_l_eval(self):
        reference_tokens = word_tokenize(self.gold)
        candidate_tokens = word_tokenize(self.pred)

        reference_lcs = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=None)

        recall = reference_lcs / len(reference_tokens)
        precision = reference_lcs / len(candidate_tokens)

        rouge_l_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return rouge_l_score



def record_eval(eval_results:list,
                evaluators_names:list,
                eval_type:str):
    """
    Record the evaluation results into a dataframe
    """
    assert len(eval_results) == len(evaluators_names),f"len(eval_results) must be equal to len(evaluators_names),but got {len(eval_results)} and {len(evaluators_names)}"
    df = pd.DataFrame(columns=evaluators_names)
    for i,eval_name in enumerate(evaluators_names):
        df.loc[eval_type, eval_name] = eval_results[i]
    return df




