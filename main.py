import os
from config.config import GENERAL_CONFIG, OPENAI_CONFIG, ARXIV_CONFIG, NOUGAT_CONFIG,INTEGRATE_PROMPTS,SECTION_PROMPTS,ALIGNMENT_CONFIG,LOGGER_MODES,TTS_CONFIG, APPLICATION_PROMPTS
from MMAPIS.tools import ArxivCrawler, NougatPredictor
from MMAPIS.server.summarization import Section_Summarizer,Summary_Integrator
from MMAPIS.server.preprocessing import img_txt_alignment
from MMAPIS.tools.utils import init_logging
from MMAPIS.server.downstream import BroadcastTTSGenerator, Paper_Recommender, Blog_Generator
from pathlib import Path

def main():
    logger = init_logging(logger_mode=LOGGER_MODES)
    if NOUGAT_CONFIG['pdf']:
        pdf_ls = NOUGAT_CONFIG['pdf']
        if not isinstance(pdf_ls, list):
            pdf_ls = [pdf_ls]
        if any([os.path.isdir(pdf) for pdf in pdf_ls]):
            temp_ls = []
            for pdf in pdf_ls:
                if os.path.isdir(pdf):
                    temp_ls.extend([os.path.join(pdf, file) for file in os.listdir(pdf) if file.endswith('.pdf')])
                else:
                    temp_ls.append(pdf)
            pdf_ls = temp_ls
    else:
        arxiv_crawler = ArxivCrawler(proxy=GENERAL_CONFIG['proxy'],headers=GENERAL_CONFIG['headers'],download=True,save_dir=GENERAL_CONFIG['save_dir'])
        if ARXIV_CONFIG['key_word']:
            article_ls = arxiv_crawler.run_keyword_crawler(key_word=ARXIV_CONFIG["key_word"], max_return= ARXIV_CONFIG["max_return"],return_md=False)
        else:
            article_ls = arxiv_crawler.run_daily_crawler(daily_type=ARXIV_CONFIG["daily_type"], max_return= ARXIV_CONFIG["max_return"],return_md=False)
        pdf_ls = [article.pdf_path for article in article_ls]

    nougat_predictor = NougatPredictor()
    files = nougat_predictor.pdf2md_text(pdfs=pdf_ls)

    api_key = OPENAI_CONFIG["api_key"]
    base_url = OPENAI_CONFIG["base_url"]
    model_config = OPENAI_CONFIG["model_config"]
    section_summarizer = Section_Summarizer(
                                        api_key=api_key,
                                        base_url=base_url,
                                        model_config=model_config,
                                        proxy=GENERAL_CONFIG["proxy"],
                                        prompt_ratio=OPENAI_CONFIG["prompt_ratio"],
                                        rpm_limit=OPENAI_CONFIG["rpm_limit"],
                                        num_processes=OPENAI_CONFIG["num_processes"],
                                        ignore_titles=OPENAI_CONFIG["ignore_title"],
                                        )
    integrator = Summary_Integrator(api_key=api_key, base_url=base_url, model_config=model_config,proxy=GENERAL_CONFIG["proxy"])
    recommender = Paper_Recommender(api_key=api_key, base_url=base_url, model_config=model_config,proxy=GENERAL_CONFIG["proxy"])
    blog_generator = Blog_Generator(api_key=api_key, base_url=base_url, model_config=model_config,proxy=GENERAL_CONFIG["proxy"])
    broadcast_generator = BroadcastTTSGenerator(
        llm_api_key=OPENAI_CONFIG["api_key"],
        llm_base_url=OPENAI_CONFIG["base_url"],
        tts_api_key=TTS_CONFIG["api_key"],
        tts_base_url=TTS_CONFIG["base_url"],
        model_config=model_config,
        proxy=GENERAL_CONFIG["proxy"],
        app_secret=TTS_CONFIG["app_secret"],
    )
    for i,file in enumerate(files):
        file_name = file.file_name.rsplit(".",1)[0]
        img_txt_alignment(
            text = file.content,
            pdf = pdf_ls[i],
            file_name = file_name+"_raw_aligned",
            save_dir= os.path.join(GENERAL_CONFIG["save_dir"],file_name),
            init_grid=ALIGNMENT_CONFIG["init_grid"],
            max_grid=ALIGNMENT_CONFIG["max_grid"],
            threshold=ALIGNMENT_CONFIG["threshold"],
            img_width=ALIGNMENT_CONFIG["img_width"],
        )
        section_summaries = section_summarizer.section_summarize(
            article_text=file.content,
            file_name=file_name,
            summary_prompts=SECTION_PROMPTS,
            init_grid=3,
            max_grid=4)
        img_txt_alignment(
            text = section_summaries,
            raw_md_text=file.content,
            pdf = pdf_ls[i],
            file_name = file_name+"_section_summarized",
            save_dir= os.path.join(GENERAL_CONFIG["save_dir"],file_name),
            init_grid=ALIGNMENT_CONFIG["init_grid"],
            max_grid=ALIGNMENT_CONFIG["max_grid"],
            threshold=ALIGNMENT_CONFIG["threshold"],
            img_width=ALIGNMENT_CONFIG["img_width"],
        )

        _, integration = integrator.integrate_summary(
            section_summaries=section_summaries,
            integrate_prompts=INTEGRATE_PROMPTS,
            response_only=True,
            reset_messages=True)
        path1 = img_txt_alignment(
            text=integration,
            pdf=pdf_ls[i],
            file_name=file_name+"_document_summarized",
            save_dir= os.path.join(GENERAL_CONFIG["save_dir"],file_name),
            init_grid=ALIGNMENT_CONFIG["init_grid"],
            max_grid=ALIGNMENT_CONFIG["max_grid"],
            threshold=ALIGNMENT_CONFIG["threshold"],
            img_width=ALIGNMENT_CONFIG["img_width"],
            raw_md_text=file.content,
        )


        recommendation_flag, recommendation = recommender.recommendation_generation(
            document_level_summary=integration,
            article= file.content,
            score_prompts= APPLICATION_PROMPTS["score_prompts"],
            reset_messages=True,
            response_only=True)
        blog_flag, blog_post = blog_generator.blog_generation(
            pdf=pdf_ls[i],
            document_level_summary=integration,
            section_summaries=section_summaries,
            blog_prompts=APPLICATION_PROMPTS["blog_prompts"],
            reset_messages=True,
            response_only=True,
            file_name=file_name+"_blog",
            init_grid=ALIGNMENT_CONFIG["init_grid"],
            max_grid=ALIGNMENT_CONFIG["max_grid"],
            threshold=ALIGNMENT_CONFIG["threshold"],
            img_width=ALIGNMENT_CONFIG["img_width"],
            save_dir= os.path.join(GENERAL_CONFIG["save_dir"],file_name),
        )

        broadcast_flag,broadcast_script, tts_bytes = broadcast_generator.broadcast_tts_generation(document_level_summary=integration,
                                                                                 section_summaries=section_summaries,
                                                                                 broadcast_prompts=APPLICATION_PROMPTS["broadcast_prompts"],
                                                                                 return_bytes=True,
                                                                                    )
        with open(os.path.join(GENERAL_CONFIG["save_dir"],file_name,"recommendation.md"), "w") as f:
            f.write(recommendation)
        with open(os.path.join(GENERAL_CONFIG["save_dir"],file_name,"broadcast.md"), "w") as f:
            f.write(broadcast_script)
        with open(os.path.join(GENERAL_CONFIG["save_dir"],file_name,"broadcast.mp3"), "wb") as f:
            f.write(tts_bytes)





if __name__ == "__main__":
    main()


