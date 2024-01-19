from submodule.arxiv_links import *
from submodule.openai_api import *
from submodule.nougat_main import *
from submodule.my_utils import *
from submodule.img_parser import parser_img_from_pdf
import yaml
import os
import sys
from pathlib import Path
import argparse
import json
from submodule.nougat_main.nougat.utils.checkpoint import get_checkpoint
from tqdm import tqdm
from submodule.my_utils import init_logging

logger = init_logging(logging_path='./logging.ini')





def get_args(nougat_info, arxiv_info):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize","-b",type=int,default=get_batch_size(),help="Batch size to use.")
    parser.add_argument("--checkpoint", "-c", type=Path, default=Path(nougat_info["check_point"]),help="Path to checkpoint directory.")
    parser.add_argument("--out", "-o", type=Path, default=Path(nougat_info["out"]),help="Output(Full text) directory")
    parser.add_argument("--recompute", action="store_true", default=True,help="Recompute already computed PDF, discarding previous predictions.")
    parser.add_argument("--markdown", action="store_true", default=True,help="Add postprocessing step for markdown compatibility.")
    parser.add_argument("--kw", nargs='+',default=arxiv_info['key_word'],help="keywords to search arxiv")
    parser.add_argument("--pdf", nargs="+", type=Path,default=[Path(i) for i in nougat_info["pdf"]],help="PDF(s)/PDF path to process, you can manually input the path of pdf file or the directory of pdf files.")
    parser.add_argument("--num_process","-num_pro", type=str,default=3,help= "num of process to summary")
    parser.add_argument("--rate_limit", "-rate", action="store_false", default=openai_info["rate_limit"],help="Whether is limited by openai rate limit 3 requests per minute.")

    args = parser.parse_args()
    logger.info("input args: %r", args)
    if args.checkpoint is None or not args.checkpoint.exists():
        print("No checkpoint found. Downloading checkpoint.")
        pass
        args.checkpoint = get_checkpoint(checkpoint_path=args.checkpoint)
        # args.checkpoint = get_checkpoint()
        print("Checkpoint downloaded to", args.checkpoint)
    if args.out is None:
        logger.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logger.info("Output directory does not exist. Creating output directory.")
            # if path does not exist, create it
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logger.error("Output has to be directory.")
            sys.exit(1)
    if args.pdf[0].is_dir():
        pdf_files = list(args.pdf[0].glob('*.pdf'))
        args.pdf = [i for i in pdf_files]

    return args


if __name__ == "__main__":

    yaml_path = './config.yaml'
    if os.path.exists(yaml_path):
        print(f'Loading config file from {yaml_path}')
        with open(yaml_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        print(f'Config file not found at {yaml_path}')
        sys.exit(1)

    # get info from yaml
    openai_info = config["openai"]
    with open(openai_info['prompts_path'], 'r') as f:
        prompts = json.load(f)
    arxiv_info = config['arxiv']
    nougat_info = config["nougat"]
    proxy = arxiv_info['proxy']
    arxiv_info = config['arxiv']
    ignore_titles = openai_info['ignore_title']
    args = get_args(nougat_info, arxiv_info)
    headers = arxiv_info['headers']
    base_url = openai_info['base_url']
    print("args.kw:",args.kw,"type:",type(args.kw))
    print("openai_info:",openai_info["rate_limit"],"type:",type(openai_info["rate_limit"]))
    if isinstance(args.kw, list):
        args.kw = ' '.join(args.kw)

    # links,titles,abs,authors = get_arxiv_links(key_word=args.kw,
    #                                            proxies=proxy,
    #                                            max_num=50,
    #                                            headers=headers,
    #                                            daily_type="math",
    #                                            )
    #
    # if links:
    #     args.pdf = links

    logger.info(f"Processing {len(args.pdf)} PDF(s),inculding :\n {args.pdf}")

    full_text, file_names = nougat_predict(args,proxy=proxy,headers=headers)
    logger.info(f"Processing {len(full_text)} PDF(s),inculding :\n {file_names}")

    rate_limit = 3 if args.rate_limit else None
    if file_names and args.out:
        processing_bar = tqdm(zip(full_text, file_names), leave=True, position=0)
        for i, (article, file_name_with_suffix) in enumerate(processing_bar):
            filename = file_name_with_suffix.split(".")[0]
            processing_bar.set_description(f"Processing {i + 1} file,filename: {filename},save_dir: {args.out}")
            # api_key=openai_info["api_key"]
            summerizer = OpenAI_Summarizer(openai_info, proxy, summary_prompts=prompts['section summary'],
                                           resummry_prompts=prompts["blog summary"], ignore_titles=ignore_titles,
                                           acquire_mode='url',num_processes=args.num_process,base_url=base_url,requests_per_minute=rate_limit,
                                           model_config = openai_info['model_config']
                                             )
            titles, authors, affiliations, summaries, re_respnse = summerizer.summary_with_openai(article,filename,init_grid=2)
            save_dir = str(args.out) + f"/{filename}"
            if "http" in str(args.pdf[i]):
                pdf_path = download_pdf(pdf_url=links[i], pdf_name=filename, save_dir=save_dir)
                if pdf_path:
                    img_paths = parser_img_from_pdf(pdf_path=pdf_path)
                else:
                    logger.error(f"Download pdf failed, pdf_url:{links[i]}")
                    continue
            else:
                img_paths = parser_img_from_pdf(pdf_path=args.pdf[i],save_dir=os.path.join(save_dir,"img"))
            img_content = "# Images & Tables:\n\n"
            print("img_paths:",img_paths)
            for i,img_path in enumerate(img_paths):
                img_path =  f"./img/" + os.path.basename(img_path)
                img_content += f"![Image {i}]({img_path})\n\n"
            table = "# Tables:\n"+"\n\n\n".join(re_respnse[-1])
            re_respnse = "\n\n".join(re_respnse[:-1])
            summaries = img_content + table + summaries
            re_respnse = img_content + table + re_respnse
            raw_file_name = Path(file_name_with_suffix).stem + "_raw_" + ".mmd"
            inter_file_name = Path(file_name_with_suffix).stem + "_integrate_" + ".mmd"
            file_paths = save_mmd_file(save_texts=[summaries, re_respnse], file_names=[raw_file_name, inter_file_name],
                                      save_dirs=[save_dir, save_dir])
            logger.info(f"Save abstract to {file_paths[0]},re_summary to {file_paths[1]}")
    else:
        logger.error("No PDF(s) found.")
        sys.exit(1)
