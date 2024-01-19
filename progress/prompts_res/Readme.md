# prompts

## summary

- summary阶段（对每个sectionsummary）：对于每个section，prompt为 `section summary['system']`+`section summary[section]` +` subtitle `+ 对应的part的源文本内容`subtext`
  - 具体prompt对应summary的逻辑为：得到一个子标题subtitle，然后判断section summary中的key 是否 in subtitle
    - 如果in 则使用该key对应的value作为`user prompt + key + subtext`
    - 如果不在，则使用general_summary对应的value作为`user prompt + subtitle`（也即论文中原子标题，而非section summary中的key） + `subtext`

```json
 {
  "section summary": {
    "system": "you are a wise assistant who is very helpful in summarizing the text.\\n\\n",
    "abstract": "Regarding the abstract, could you explain the main objectives and research findings of the paper?\\n\\n",
    "intro": "In relation to the introduction, what is the main motivation or background study of the authors?\\n\\n",
    "related work": "What previous research has been done that ties into this paper? How does this paper build upon the work of previous researchers?\\n\\n",
    "model": "Could you summarize the theoretical model or the approach of the research, and explain why the authors chose this method?\\n\\n",
    "method": "What research methods and techniques are applied in the methods section of this paper? Could you outline the main steps of data collection and analysis described in the methods section?\\n\\n",
    "conclusion": "What are the main conclusions and findings mentioned in the conclusion section of this paper? Based on the conclusions, what further directions for research do the authors suggest?\\n\\n",
    "result": "What were the main results or findings of the research?\"\nDiscussion: \"How do the authors interpret their results? How do these results compare to previous research, and what implications do they have?\\n\\n",
    "discussion": "How do the authors interpret their results? How do these results compare to previous research, and what implications do they have?\\n\\n",
    "future": "According to the authors, what additional research needs to be conducted or what new questions have emerged from this study?\\n\\n",
    "experiment": "What were the main experimental techniques or procedures employed in the research?\\n\\n",
    "dataset": "What data has been used in this study? Could you describe its features and how it was collected or generated?\\n\\n",
    "limit": "What limitations in the research or methodology are indicated in the paper? With regard to the limitations section of the paper, what recommendations for improvements or possible solutions do the authors put forward? \\n\\n",
    "general_summary": "Please summarize the section of the paper that you have been assigned,the title of the section is [title_to_replace]\\n\\n"
  },
  "global summary": {
    "system": "You are an experienced article summarizer and rewriter proficient in consolidating various sections.\\n\\n",
    "overview": "Given a series of summaries, your task is to logically reorganize and refine them into an engaging narrative. The summaries are encapsulated within backticks (`), and your role is to extract, reorder, and reshape the content to generate a superior quality article overview in markdown format . This task goes beyond merely piecing together the sections; you are required to transform them into a  coherent and smoothly flowing storyline, while preserving the main essence of the original content.\\n\\n",
    "old_overview": "Craft a comprehensive summary drawing from the distinct segments of the original abstract, and provide a summarization of the entire content. Additionally, assign a specific score (out of 10 points) to the article, serving as a recommendation guide for potential readers.\\n\\n",
    "resummary": "Small sections of the paper have already been summarised for you, given between triple backticks in an array. Unite these summaries into a larger complete summary.\\n\\n",
    "score": "Hello, ChatGPT. I have a paper that requires your evaluation. Please assess it based on the following main criteria: clarity of the paper's theme and objectives, appropriateness and detail of the research methods, accuracy of the data and results, depth of the discussion and conclusion, and overall writing quality (including grammar, spelling, clarity, etc.). Please provide a score from 1-10 for each criterion, with 1 being very poor and 10 being excellent. Then, calculate the average of all scores to give an overall rating for the paper. "
  }
}
```

具体prompt如下（其中chunks为list，每个元素也即对应的subsection，chunk[0]为subsection中的subtitle，chunk[1]为对应subtext，通过assgin_prompts得到对应的prompt以及subtitle，如果section summary中的key 是否 in subtitle，则subtitle=key 否则subtitle = 原论文中的subtitle，此处也可以看出，<u>当用的是原论文中的subtitle时（markdown格式，会带#）在summary.mmd中会出现## ## subtitle的情况，而使用section summary中的key时，与## 结合，没有多余的#</u>）：

```python
for i,chunk in enumerate(chunks):
     subtitle,summary_prompt = assgin_prompts(prompts,chunk[0])
     input_prompt = summary_prompt + chunk[1]
     resp = chat_with_openai(input_prompt,messages,model,temperature,max_tokens,top_p,
                             frequency_penalty,presence_penalty,response_only=True,prompt_factor=prompt_factor)

     respons.append('\n## ' + subtitle + ':\n' + resp)
```



## re_summary

re_summary阶段：目前使用的prompt是 `global summary['system'] `+` global summary['overview'] `+ 所有summary拼接 , score的prompt为此前的`message + global summary['score']`

具体代码如下图所示（`respons`的每个元素为每个部分得到的subtitle+section summary,`titles`为经过nougat得到的论文总标题）：

```python
full_resp = ''.join(respons)
messages = init_messages('system',resummry_prompt['system'])

prompt = resummry_prompt['overview'] + '\n\n ```' + titles + full_resp + '```'

re_respnse = chat_with_openai(prompt,messages,model,temperature,max_tokens,top_p,
                              frequency_penalty,presence_penalty,response_only=True,
                              prompt_factor=prompt_factor,reset_messages=False)
score = chat_with_openai(resummry_prompt['score'],messages,model,temperature,max_tokens,top_p,
                         frequency_penalty,presence_penalty,response_only=True,
                         prompt_factor=prompt_factor,reset_messages=False)
logging.info(f'finished resummry the article,with the titles:{titles},authors:{authors}')
```



# results

> raw_mmd: [2310_03026.mmd](raw_mmd\2310_03026.mmd) 
>
> summary_mmd:  [2310_03026.mmd](summary\2310_03026.mmd) 
>
> resummary_mmd: [2310_03026.mmd](resummary_mmd\2310_03026.mmd) 

>  raw_mmd:  [2310_08102.mmd](raw_mmd\2310_08102.mmd) 
>
> summary_mmd:   [2310_08102.mmd](summary\2310_08102.mmd) 
>
> resummary_mmd: [2310_08102.mmd](resummary_mmd\2310_08102.mmd)

>  raw_mmd:   [2310_08365.mmd](raw_mmd\2310_08365.mmd) 
>
> summary_mmd:   [2310_08365.mmd](summary\2310_08365.mmd) 
>
> resummary_mmd:  [2310_08365.mmd](resummary_mmd\2310_08365.mmd) 

> raw_mmd:   [2310_08582.mmd](raw_mmd\2310_08582.mmd) 
>
> summary_mmd:   [2310_08582_202310151900.mmd](summary\2310_08582_202310151900.mmd) 
>
> resummary_mmd:  [2310_08582_202310151900.mmd](resummary_mmd\2310_08582_202310151900.mmd) 