import os
import re
import requests
import argparse
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from MMAPIS.backend.config.config import (
    GENERAL_CONFIG,
    OPENAI_CONFIG,
    ARXIV_CONFIG,
    NOUGAT_CONFIG,
    DOCUMENT_PROMPTS,
    SECTION_PROMPTS,
    ALIGNMENT_CONFIG,
    LOGGER_MODES,
    APPLICATION_PROMPTS,
)
from MMAPIS.backend.preprocessing import ArxivCrawler, NougatPredictor, Aligner, PDFFigureExtractor
from MMAPIS.backend.summarization import Summarizer
from MMAPIS.backend.downstream import BroadcastTTSGenerator, PaperRecommender, BlogGenerator
from MMAPIS.backend.tools import init_logging

def align_text_with_paths(text, img_paths, raw_md_text: str = None, min_grained_level: int = 3, max_grained_level: int = 4, img_width: int = 400, margin: int = 10, threshold: float = 0.9):
    """Align text with images using predefined thresholds and granularity levels."""
    aligner = Aligner()
    aligned_text = aligner.align(
        text=text,
        img_paths=img_paths,
        raw_md_text=raw_md_text,
        min_grained_level=min_grained_level,
        max_grained_level=max_grained_level,
        img_width=img_width,
        margin=margin,
        threshold=threshold,
    )
    return aligned_text



def save_file(file_path, content,is_bytes=False):
    if is_bytes:
        with open(file_path, "wb") as f:
            f.write(content)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

def load_file(file_path,is_bytes=False):
    if is_bytes:
        with open(file_path, "rb") as f:
            content = f.read()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    return content


def clean_img_url(text: str) -> str:
    """Clean image URLs by removing div tags in the provided text."""
    div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
    return re.sub(div_pattern, '', text)


def process_pdf_list(pdf_ls):
    """Process a list of PDFs or directories, extracting individual PDF files."""
    if not isinstance(pdf_ls, list):
        pdf_ls = [pdf_ls]
    temp_ls = []
    for pdf in pdf_ls:
        if os.path.isdir(pdf):
            temp_ls.extend([os.path.join(pdf, file) for file in os.listdir(pdf) if file.endswith('.pdf')])
        else:
            temp_ls.append(pdf)
    return temp_ls

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description='Process PDF documents for summarization, recommendation, and blog generation.')
    parser.add_argument('-p', '--pdf', nargs='*', default=NOUGAT_CONFIG['pdf'],
                        help='List of PDF files or directories or Arxiv links')
    parser.add_argument('-k', '--api_key', type=str, default=OPENAI_CONFIG['api_key'], help='OpenAI API key')
    parser.add_argument('-b', '--base_url', type=str, default=OPENAI_CONFIG['base_url'], help='Base URL for OpenAI API')
    parser.add_argument('-kw', '--keyword', type=str, default=ARXIV_CONFIG['key_word'],
                        help='Keyword for Arxiv search if PDFs are not provided')
    parser.add_argument('-dt', '--daily_type', type=str, default=ARXIV_CONFIG['daily_type'],
                        help='Type of daily Arxiv papers if PDFs are not provided and keyword is not provided')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Download PDFs from Arxiv if PDFs are not provided')
    parser.add_argument('-s', '--save_dir', type=str, default=GENERAL_CONFIG['save_dir'],
                        help='Directory to save the results')
    parser.add_argument('--recompute', action='store_true', help='Recompute the results')
    parser.add_argument('--all', action='store_true',
                        help='Process all downstream tasks after summarization, including recommendation, blog, and broadcast')
    parser.add_argument('--app', action="store", type=str,
                        help="Specify the downstream task to run, choose from recommendation, blog, broadcast",
                        choices=["recommendation", "blog", "broadcast"])

    args = parser.parse_args()

    # Initialize logging
    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(LOGGER_MODES)

    # Load configurations, giving priority to command line arguments
    api_key = args.api_key
    base_url = args.base_url
    pdf_ls = args.pdf
    save_dir = args.save_dir
    keyword = args.keyword
    download = args.download
    daily_type = args.daily_type
    run_all = args.all
    specific_app = args.app

    # Validate API key and base URL
    if not (api_key and base_url):
        raise ValueError("API key and base URL must be provided either via --api_key and --base_url or in the config file.")

    # Process PDFs
    if pdf_ls:
        pdf_ls = process_pdf_list(pdf_ls)  # Encapsulated logic for handling PDFs
    else:
        # Use ArxivCrawler to fetch PDFs based on keyword or daily type
        arxiv_crawler = ArxivCrawler(proxy=GENERAL_CONFIG['proxy'],headers=GENERAL_CONFIG['headers'],download=download, save_dir=save_dir)
        if keyword:
            logger.info("Running Arxiv keyword crawler with keyword: %s", keyword)  # Logging keyword crawler action
            article_ls = arxiv_crawler.run_keyword_crawler(key_word=keyword, max_return= ARXIV_CONFIG["max_return"],return_md=False)
        else:
            logger.info("Running Arxiv daily crawler with daily_type: %s", daily_type)  # Logging daily crawler action
            article_ls = arxiv_crawler.run_daily_crawler(daily_type=daily_type, max_return= ARXIV_CONFIG["max_return"],return_md=False)
        pdf_ls = [article.pdf_path for article in article_ls]

    nougat_predictor = NougatPredictor()
    files = nougat_predictor.pdf2md_text(pdfs=pdf_ls)


    # Initialize generators
    model_config = OPENAI_CONFIG["model_config"]
    summarizer = Summarizer(
                            api_key=api_key,
                            base_url=base_url,
                            model_config=model_config,
                            proxy=GENERAL_CONFIG["proxy"],
                            prompt_ratio=OPENAI_CONFIG["prompt_ratio"],
                            rpm_limit=OPENAI_CONFIG["rpm_limit"],
                            num_processes=OPENAI_CONFIG["num_processes"],
                            ignore_titles=OPENAI_CONFIG["ignore_title"],
                            )
    recommender = PaperRecommender(api_key=api_key, base_url=base_url, model_config=model_config,
                                   proxy=GENERAL_CONFIG["proxy"])
    blog_generator = BlogGenerator(api_key=api_key, base_url=base_url, model_config=model_config,
                                   proxy=GENERAL_CONFIG["proxy"])
    broadcast_generator = BroadcastTTSGenerator(
        llm_api_key=api_key,
        llm_base_url=base_url,
        model_config=model_config,
        proxy=GENERAL_CONFIG["proxy"],
        prompt_ratio=OPENAI_CONFIG["prompt_ratio"],
    )

    for i,file in enumerate(files):
        file_name = file.file_name.split("/")[-1].replace('.', '_')
        file_save_dir = os.path.join(save_dir,file_name)
        os.makedirs(file_save_dir,exist_ok=True)
        if pdf_ls[i].endswith(".pdf"):
            pdf_path = pdf_ls[i]
        else:
            pdf_content = requests.get(pdf_ls[i]).content
            pdf_path = os.path.join(file_save_dir,f"{file_name}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)

        pdf_parser = PDFFigureExtractor(pdf_path)
        img_paths = pdf_parser.extract_save_figures(save_dir=file_save_dir)
        del pdf_parser

        aligned_raw_md_path = os.path.join(file_save_dir,"aligned_raw_md_text.md")
        if not args.recompute and os.path.exists(aligned_raw_md_path):
            logger.info(f"File {file_name} is already processed, raw text alignment will be skipped")
        else:
            aligned_raw_md_text = align_text_with_paths(
                                    text = file.content,
                                    img_paths = img_paths,
                                    raw_md_text = file.content,
                                    min_grained_level = ALIGNMENT_CONFIG["min_grained_level"],
                                    max_grained_level = ALIGNMENT_CONFIG["max_grained_level"],
                                    img_width = ALIGNMENT_CONFIG["img_width"],
                                    margin = ALIGNMENT_CONFIG["margin"],
                                    threshold = ALIGNMENT_CONFIG["threshold"],
                                )
            save_file(aligned_raw_md_path,aligned_raw_md_text)

        aligned_section_summary_path = os.path.join(file_save_dir,"aligned_section_level_summary.md")
        aligned_document_summary_path = os.path.join(file_save_dir,"aligned_document_level_summary.md")
        if not args.recompute and os.path.exists(aligned_section_summary_path) and os.path.exists(aligned_document_summary_path):
            logger.info(f"File {file_name} is already processed, summarization will be skipped")
            section_level_summary = clean_img_url(load_file(aligned_section_summary_path))
            document_level_summary = clean_img_url(load_file(aligned_document_summary_path))

        else:
            flag, summaries = summarizer.generate_summary(
                text=file.content,
                section_prompts=SECTION_PROMPTS,
                document_prompts=DOCUMENT_PROMPTS,
                min_grained_level=ALIGNMENT_CONFIG["min_grained_level"],
                max_grained_level=ALIGNMENT_CONFIG["max_grained_level"],
            )
            if not flag:
                logger.error(f"Failed to generate summaries for {file_name}, due to {summaries}")
                continue

            section_level_summary = summaries["section_level_summary"]
            document_level_summary = summaries["document_level_summary"]

            # Align and save section summary
            aligned_section_level_summary = align_text_with_paths(
                text=section_level_summary,
                img_paths=img_paths,
                raw_md_text=file.content,
                min_grained_level=ALIGNMENT_CONFIG["min_grained_level"],
                max_grained_level=ALIGNMENT_CONFIG["max_grained_level"],
                img_width=ALIGNMENT_CONFIG["img_width"],
                margin=ALIGNMENT_CONFIG["margin"],
                threshold=ALIGNMENT_CONFIG["threshold"],
            )
            save_file(aligned_section_summary_path,aligned_section_level_summary)
            logger.info(f"Section-level summary is saved to {aligned_section_summary_path}")

            aligned_document_level_summary = align_text_with_paths(
                text=document_level_summary,
                img_paths=img_paths,
                raw_md_text=file.content,
                min_grained_level=ALIGNMENT_CONFIG["min_grained_level"],
                max_grained_level=ALIGNMENT_CONFIG["max_grained_level"],
                img_width=ALIGNMENT_CONFIG["img_width"],
                margin=ALIGNMENT_CONFIG["margin"],
                threshold=ALIGNMENT_CONFIG["threshold"],
            )
            save_file(aligned_document_summary_path,aligned_document_level_summary)
            logger.info(f"Document-level summary is saved to {aligned_document_summary_path}")

        # Recommendations and blog generation
        recommendation_path = os.path.join(file_save_dir,"recommendation.md")
        blog_path = os.path.join(file_save_dir,"blog.md")
        broadcast_path = os.path.join(file_save_dir,"broadcast.md")

        # Handle recommendations
        if not args.recompute and os.path.exists(recommendation_path) :
            logger.info(f"File {file_name} is already processed or not needed, recommendation generation will be skipped")
        elif specific_app == "recommendation" or run_all:
            recommendation_flag, recommendation = recommender.recommendation_generation(
                document_level_summary=document_level_summary,
                raw_md_text= file.content,
                score_prompts= APPLICATION_PROMPTS["score_prompts"],
                reset_messages=True,
                response_only=True)
            if not recommendation_flag:
                logger.error(f"Failed to generate recommendation for {file_name}, due to {recommendation}")
            else:
                recommendation_text = ""
                for item in recommendation:
                    for i, v in enumerate(item.values()):
                        if i == 0:
                            recommendation_text += f"- {v}\n"
                        else:
                            recommendation_text += f"  - {v}\n"
                save_file(recommendation_path,recommendation_text)
                logger.info(f"Recommendation is saved to {recommendation_path}")

        # Handle blog generation
        if not args.recompute and os.path.exists(blog_path):
            logger.info(f"File {file_name} is already processed or not needed, blog generation will be skipped")
        elif specific_app == "blog" or run_all:
            blog_flag, blog_post = blog_generator.blog_generation(
                document_level_summary=document_level_summary,
                section_level_summary=section_level_summary,
                blog_prompts=APPLICATION_PROMPTS["blog_prompts"],
                reset_messages=True,
                response_only=True,
            )
            if not blog_flag:
                logger.error(f"Failed to generate blog for {file_name}, due to {blog_post}")
            else:
                aligned_blog_post = align_text_with_paths(
                    text=blog_post,
                    img_paths=img_paths,
                    raw_md_text=file.content,
                    min_grained_level=ALIGNMENT_CONFIG["min_grained_level"],
                    max_grained_level=ALIGNMENT_CONFIG["max_grained_level"],
                    img_width=ALIGNMENT_CONFIG["img_width"],
                    margin=ALIGNMENT_CONFIG["margin"],
                    threshold=ALIGNMENT_CONFIG["threshold"],
                )
                save_file(os.path.join(file_save_dir,"blog.md"),aligned_blog_post)
                logger.info(f"Blog is saved to {blog_path}")

        # Handle broadcast generation
        if not args.recompute and os.path.exists(broadcast_path):
            logger.info(f"File {file_name} is already processed or not needed, broadcast generation will be skipped")
        elif specific_app == "broadcast" or run_all:
            broadcast_flag, broadcast_script, tts_bytes = broadcast_generator.broadcast_tts_generation(
                document_level_summary=document_level_summary,
                section_level_summary=section_level_summary,
                broadcast_prompts=APPLICATION_PROMPTS["broadcast_prompts"],
                return_bytes=True,
            )
            if not broadcast_flag:
                logger.error(f"Failed to generate broadcast for {file_name}, due to {broadcast_script}")
            else:
                save_file(os.path.join(file_save_dir,"broadcast.md"),broadcast_script)
                if tts_bytes:
                    save_file(os.path.join(file_save_dir,"broadcast.mp3"),tts_bytes,is_bytes=True)
                else:
                    logger.error("TTS bytes is empty")
                logger.info(f"Broadcast is saved to {broadcast_path}")
        logger.info(f"File {file_name} finished processing")





if __name__ == "__main__":
    main()


