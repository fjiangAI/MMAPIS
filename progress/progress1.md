# Progress1

- [x] [arxiv_extractor](../tools/arxiv_extractor)  
- [x] [chatgpt](../tools/chatgpt)  
- [ ] [nougat](../tools/nougat)  
- [x] [pdffigure](../tools/pdffigure)  
- [x] [tts](../tools/tts)  
- [x] [utils](../tools/utils) 
- [ ] [preprocessing](../server/preprocessing) 
     - [x] [summarization](../server/summarization)
          - [x] 按照Grid进行summary，例如按照最低为三级标题进行summary，得到的summary也是最高精度为三级，图片的精度主要看pdffigure
          - [x] alignment按照希望的grid进行alignment，如果section summary是三级精度，而这里alignment是二级，则最后生成的精度是二级

- [ ] [downstream](../server/downstream) 
     - [x]  [audio_broadcast](../server/downstream/audio_broadcast) 
     - [x]  [blog_generation](../server/downstream/blog_generation) 
     - [ ] [multimodal_qa](../server/downstream/multimodal_qa)  
     - [x] [paper_recommendation](../server/downstream/paper_recommendation) 
- [ ] streamlit
     - [ ] 寻找streamlit 多图显示
     - [ ] 将上传文件部分调好

