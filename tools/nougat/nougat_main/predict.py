"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import urllib.error
from pathlib import Path
import logging
import argparse
import re
from functools import partial
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from .nougat import NougatModel
from .nougat.utils.dataset import LazyDataset
from .nougat.postprocessing import markdown_compatible
import fitz

logging.basicConfig(level=logging.INFO)

import torch


def latex2md_math(text):
    """
    convert latex math to markdown math
    Args:
        text: text with latex math

    Returns: text with markdown math

    """
    text = re.sub(r'\\\[(.*?)\\\]',r'$$\1$$',text)
    text = re.sub(r'\\\((.*?)\\\)',r'$\1$',text)
    text = re.sub(r'\\mathbbm',r"\\mathbb",text)
    return text


def nougat_predict(args,proxy=None,headers=None,pdf_name:str = None):
    # args = get_args()
    model = NougatModel.from_pretrained(args.checkpoint,ignore_mismatched_sizes=True).to(torch.bfloat16)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    datasets = []
    pdf_process_bar = tqdm(args.pdf)
    for i,pdf in enumerate(pdf_process_bar):
        pdf_process_bar.set_description(f"[nougat]Extracting raw content from  {i+1} pdf,name: {pdf if not isinstance(pdf,bytes) else pdf_name}")
        if isinstance(pdf,Path):
            if not pdf.exists():
                print(f"File {pdf} does not exist.")
                continue
            if args.out:
                out_path = args.out /Path("raw_mmd")/ pdf.with_suffix(".mmd").name
                # if exists and not recompute, skip
                if out_path.exists() and not args.recompute:
                    logging.info(
                        f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                    )
                    continue
        try:
        # pdf: Path, prepare_input: Callable[[Image.Image], torch.Tensor]
        # input: pdf + prepare_input(img)
        # dataset: [num_pages]
            dataset = LazyDataset(
                pdf, partial(model.encoder.prepare_input, random_padding=False),proxy=proxy,headers=headers,pdf_name= pdf_name
            )
            logging.info(f"Loaded {str(pdf) if not isinstance(pdf,bytes) else pdf_name},length: {len(dataset)}")
        # if pdf is corrupted, skip
        except fitz.fitz.FileDataError:
            logging.info(f"Could not load file in path {str(pdf)}.")
            continue
        except urllib.error.URLError:
            logging.info(f"Could not load file in URL {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        logging.error("No PDFs to process.")
        return None,None
    # shape
    # datasets: [num_files*num_pages]
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0
    # [sample(img) , is_last_page] is data from a pdf file (1 page)
    # sample: [batch_size,num_channels(3),h,w],is_last_page = [batch_size] in ['' or pdf_name]
    article_ls = []
    file_names = []

    dataloader_bar = tqdm(dataloader,leave=True,position=0)
    for i, items in enumerate(dataloader_bar):
        dataloader_bar.set_description(f"[nougat]now processing[{i+1}] img")
        if items is None:
            logging.warning(f"Ignore empty batch in page {page_num+1}.")
            continue
        else:
            (sample, is_last_page) = items
        model_output = model.inference(image_tensors=sample)
        # check if model output is faulty
        # output [batch_Size,decoder_seq_len]
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1

            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                continue
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                out = latex2md_math(out)
                if args.out:
                    file_name = Path(is_last_page[j]).with_suffix(".mmd").name
                    out_path = args.out /Path("raw_mmd")/file_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")

                    file_names.append(file_name)
                    article_ls.append(out)
                else:
                    print(out, "\n\n")
                predictions = []
                page_num = 0
                file_index += 1

    return article_ls,file_names

