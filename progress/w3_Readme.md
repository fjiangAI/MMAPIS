# overview

> this is code complement for [arxiv_summarizer](https://github.com/KKKe2922/arxiv_summarizer)

## work flow

ideal work flow is blow:

![work_flow](./img/33400513c75a12b59a58fa4b5df49fa.png)

now have done is below

```mermaid
stateDiagram-v2
[*]-->user
user --> yaml.config
yaml.config -->arxiv_links :keyword,proxy  
yaml.config  --> nougat:save_dir,check_point
yaml.config --> openai : api_key,prompt
arxiv_links --> nougat_predict
nougat--> nougat_predict
nougat_predict:raw_pdf_content
openai --> summary
nougat_predict --> summary
nougat_predict -->piece_1: half_split
nougat_predict -->piece_2 : half_split
nougat_predict -->piece_n : half_split
piece_1 -->sum_piece_1:summary
piece_2 -->sum_piece_2:summary
piece_n -->sum_piece_n:summary
sum_piece_1 --> summary : resummary
sum_piece_2 --> summary : resummary
sum_piece_n --> summary : resummary
summary-->[*]
```

## progress

directory structure:

```
── code_arxiv_summarizer/
   ├── data/
   ├── utils/
   │   ├── arxiv_links/
   │   │   ├── get_links.py
   │   ├── nougat_main/
   │   │   ├── nougat/
   │   │   ├── predict.py
   │   ├── openai_api/
   │   │   ├── api_usage.py
   ├── res/
   │   ├── raw_mmd/
   │   ├── summary_mmd/
   ├── README.md
   ├── config.yaml
   ├── main.py
   ├── img/
   ├── pretrained_w/
```

1. use [main.py](./main.py) to run the whole project

   ```python
   python main.py --pdf path/to/pdf --out path/to/save --markdown
   ```
2. some config that only need to modified once is placed in [yaml](./config.yaml),include proxy(to extract url),openai key and default prompts
3. some input that will change every time user input is got from argparser(like pdf_path,output path)
4. in the function `get_arxiv_links`,you can choose to

   - input the keyword and set params(like max_num,sorted order) to get the papers and summaries under the keywords
   - get the daily new submissions(keyword = None)
5. if proxy is None, [ProxyServer](submodule/arxiv_links/get_links.py) will read default information in Windows
6. if input is too long,it will be splited into piece to ensure each piece will be lower that max_tokens * 0.5(make answer and prompt 1:1)

and summary each pieces and then resummary the summaries

7. the current work flow is show in above,which means we will

   - first get the pdf content by the nougat and save them  as out/raw_mmd/xxxmmd, the raw text will also return to main function
   - then use openai api to summary and save them as out/summary_mmd/xxx.mmd (so we cannot get the raw content and summary one by one but first get all pdf content then get all summaries)
8. the output examples can be seen in [pdf_raw_content(nougat_output)](./res/raw_mmd) and [summary](./res/summary_mmd),they are results of top 10 links when searching LLM in 2023/09/15

# exist problems

* [ ]  how to deal with long article which exceed the max length of the openai api

1. **solution**:  cut in half until each piece length is lower than max_tokens and resummary every section subsequently,code is in [api_usage/split_text](submodule/openai_api/api_usage.py)
2. due to the max length contains prompt and answer,now I define prompt length should not be longer than half of max length(make answer and prompt 1:1)

* [ ]  the problem of too long input when resummary ?
* [X]  ~~how to deal with url input to extract pdf content? the origional input is all from local pdf~~

* **solution**: use requests and get the content(which is binary data),then use fitz to load

* [X]  ~~how to get target url with user input keywords ?~~

1. solution1: Crawl all data for the day and count the frequency of occurrence and filter with the keyword,there's a template in github [Arxiv_daily: A little spider](https://github.com/ZihaoZhao/Arxiv_daily/tree/master),which get all the info from arxiv and cleaned and ended up with a pandas,finally followed by sending the information to email

   - problem 1: maybe in a day,there may not some papers we want and I have tried once, the total num of each day's paper is ~1000(from the [arxiv_daily](https://arxiv.org/list/cs/pastweek?show=1000)),and finally just crawl 307 urls
   - problem 2: we can not search some keywords that may not appear in the subject defined by the arxiv.org(like LLMs,which may just be under the DL subject )
2. search keyword:

   1. use selenium

      - problem 1 : too slow,need cookies and simulate a real person clicking on the browser
      - problem 2 : now troubled with the version,the latest chrome Driver only support 114 but my browser version is 116,I don't know how to download the previous version
   2. find that url is structural data and can divide into pieces:

      ```
      https://arxiv.org/search/?query=CV&searchtype=all&abstracts=show&order=-announced_date_first&size=50
      ```

      can be divided into `key_word`+`searchtype`+`abstracts`+`order`+`size`

      ```
      'https://arxiv.org/search/?query={key_word}&searchtype={searchtype}' \
            f'&abstracts={abstracts}&order={order}&size={size}'
      ```

* [X]  ~~how to import python file from different packages~~? (done,use `__file__` to locate current file and add to sys.path)

want to import `rasterize.py` in `1.py`

directory structure:
![img1](./img/cd34ffda51bcf3190f70d81ab469a44.png)
have tried methods:

1. add sys.path（×）

   ```python
   sys.path.append(os.path.abspath(os.path.join("__file__","..")))
   ```
2. relative import（×）

   ```python
   from ..dataset.rasterize import rasterize_paper
   ```
3. add sys.path（√）

   ```python
   sys.path.append(os.path.abspath(os.path.join(__file__,"../dataset")))
   ```
4. absolute import（×） as there's a same name package in anaconda/lib, when search in sys.path, it will be priority

   ```python
   import nougat.dataset.rasterize
   ```

* [ ]  low speed:`20 pages --> ~3 mins`

  it needs long time to extract image from pdf_url,due to the dataset the code defined is lack of parallelism (only after one pdf is done for all pages,then it will continue due to for loop),especially when it contains many figures,perhaps due to the complex binary file of images
* [X]  ~~input 3 file -> output 1 file?~~

- **solution**: fine that due to the name of each file is arxiv links(like https://arxiv.org/pdf/2309.06477) when being suffix with .mmd , all the `\.[.*]` will be replaced by .mmd,so the latter file will cover the former)

* [ ]  [Robots Beware - arXiv info](https://info.arxiv.org/help/robots.html)

  seems automated downloads(spiders) are not permitted,but if just 10 papers per trial may be not detected
* [ ]  How to realize the `interpretation prompts `and `writing prompts` mentioned in the workflow

  - **solution**: in [config.yaml](./config.yaml), the prompts can be divided into different parts,so we can use a for loop to string together questions from different parts of the interpretation chain

    ```
    openai:
      api_key: "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"
      prompts:
        system_prompt: "you are a wise assistant who is very helpful in summarizing the text."
        summary: >
          Please summarize the paper by author(s) in one concise sentence.
          Then, list key insights and lessons learned from the paper.
          Next, generate 3-5 questions that you would like to ask the authors about their work.
          Finally, provide 3-5 suggestions for related topics or future research directions  based on the content of the paper.
          If applicable, list at least 5 relevant references from the field of study of the paper.
        resummary: >
          Small sections of the paper have already been summarised for you,
          given between triple backticks in an array.
          Unite these summaries into a larger complete summary.\n\n
        intro: "this paper is about:"
        model: "which model do you want to use?"
        conclusion: "conclusion:"
        overview: "this paper is about"
    ```

# appendix

## different models:

### [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)

GPT-3.5 models can understand and generate natural language or code. Our most capable and cost effective model in the GPT-3.5 family is `gpt-3.5-turbo` which has been optimized for chat using the [Chat completions API](https://platform.openai.com/docs/api-reference/chat) but works well for traditional completions tasks as well.


| LATEST MODEL                | DESCRIPTION                                                                                                                                                                                                                                                           | MAX TOKENS    | TRAINING DATA  |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :--------------- |
| gpt-3.5-turbo               | Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of`text-davinci-003`. Will be updated with our latest model iteration 2 weeks after it is released.                                                                                              | 4,097 tokens  | Up to Sep 2021 |
| gpt-3.5-turbo-16k           | Same capabilities as the standard`gpt-3.5-turbo` model but with 4 times the context.                                                                                                                                                                                  | 16,385 tokens | Up to Sep 2021 |
| gpt-3.5-turbo-0613          | Snapshot of`gpt-3.5-turbo` from June 13th 2023 with function calling data. Unlike `gpt-3.5-turbo`, this model will not receive updates, and will be deprecated 3 months after a new version is released.                                                              | 4,097 tokens  | Up to Sep 2021 |
| gpt-3.5-turbo-16k-0613      | Snapshot of`gpt-3.5-turbo-16k` from June 13th 2023. Unlike `gpt-3.5-turbo-16k`, this model will not receive updates, and will be deprecated 3 months after a new version is released.                                                                                 | 16,385 tokens | Up to Sep 2021 |
| gpt-3.5-turbo-0301 (Legacy) | Snapshot of`gpt-3.5-turbo` from March 1st 2023. Unlike `gpt-3.5-turbo`, this model will not receive updates, and will be deprecated on June 13th 2024 at the earliest.                                                                                                | 4,097 tokens  | Up to Sep 2021 |
| text-davinci-003 (Legacy)   | Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models. Also supports some additional features such as[inserting text](https://platform.openai.com/docs/guides/gpt/inserting-text). | 4,097 tokens  | Up to Jun 2021 |
| text-davinci-002 (Legacy)   | Similar capabilities to`text-davinci-003` but trained with supervised fine-tuning instead of reinforcement learning                                                                                                                                                   | 4,097 tokens  | Up to Jun 2021 |
| code-davinci-002 (Legacy)   | Optimized for code-completion tasks                                                                                                                                                                                                                                   | 8,001 tokens  | Up to Jun 2021 |

### [GPT-4](https://platform.openai.com/docs/models/gpt-4)

GPT-4 is currently accessible to those who have made at least [one successful payment](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4) through our developer platform.

GPT-4 is a large multimodal model (accepting text inputs and emitting text outputs today, with image inputs coming in the future) that can solve difficult problems with greater accuracy than any of our previous models, thanks to its broader general knowledge and advanced reasoning capabilities. Like `gpt-3.5-turbo`, GPT-4 is optimized for chat but works well for traditional completions tasks using the [Chat completions API](https://platform.openai.com/docs/api-reference/chat). Learn how to use GPT-4 in our [GPT guide](https://platform.openai.com/docs/guides/gpt).


| LATEST MODEL            | DESCRIPTION                                                                                                                                                                              | MAX TOKENS    | TRAINING DATA  |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :--------------- |
| gpt-4                   | More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration 2 weeks after it is released.                | 8,192 tokens  | Up to Sep 2021 |
| gpt-4-0613              | Snapshot of`gpt-4` from June 13th 2023 with function calling data. Unlike `gpt-4`, this model will not receive updates, and will be deprecated 3 months after a new version is released. | 8,192 tokens  | Up to Sep 2021 |
| gpt-4-32k               | Same capabilities as the standard`gpt-4` mode but with 4x the context length. Will be updated with our latest model iteration.                                                           | 32,768 tokens | Up to Sep 2021 |
| gpt-4-32k-0613          | Snapshot of`gpt-4-32` from June 13th 2023. Unlike `gpt-4-32k`, this model will not receive updates, and will be deprecated 3 months after a new version is released.                     | 32,768 tokens | Up to Sep 2021 |
| gpt-4-0314 (Legacy)     | Snapshot of`gpt-4` from March 14th 2023 with function calling data. Unlike `gpt-4`, this model will not receive updates, and will be deprecated on June 13th 2024 at the earliest.       | 8,192 tokens  | Up to Sep 2021 |
| gpt-4-32k-0314 (Legacy) | Snapshot of`gpt-4-32` from March 14th 2023. Unlike `gpt-4-32k`, this model will not receive updates, and will be deprecated on June 13th 2024 at the earliest.                           | 32,768 tokens | Up to Sep 2021 |
