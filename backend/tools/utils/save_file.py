import sys
from pathlib import Path
import logging
import logging.config
from typing import List,Union
from datetime import datetime
import requests
from retrying import retry

def save_mmd_file(save_texts: Union[str,List[str]],
                  file_names: Union[str, Path,List[str],List[Path]], # file name with suffix
                  save_dirs: Union[str, Path,List[str],List[Path]],
                  ):
    assert save_texts is not None, "summary_text is None"
    if isinstance(save_texts,str):
        save_texts = [save_texts]
    if isinstance(file_names,str):
        file_names = [file_names]
    if isinstance(save_dirs,str):
        save_dirs = [save_dirs]*len(save_texts)
    print(f"save_text:{save_texts},file_name:{file_names},save_dir:{save_dirs}")
    assert len(save_texts) == len(file_names) == len(save_dirs), f"length of save_text,file_name,save_dir must be equal,length of save_text:{len(save_texts)},file_name:{len(file_names)},save_dir:{len(save_dirs)}"
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    save_path = []
    for text,file_name,save_dir in zip(save_texts,file_names,save_dirs):
        file_name, save_dir = Path(file_name), Path(save_dir)
        filename = file_name.stem + f"_{timestamp}" + file_name.suffix
        file_path = save_dir / filename
        logging.info(f"file {str(file_name)} save to {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
        save_path.append(file_path)
    return save_path

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def download_pdf(pdf_url: Union[str,Path],
                 save_dir: Union[str,Path],
                 pdf_name: Union[str, Path] = None,
                 temp_file: bool = False,
                 ):
    """
    download pdf from url
    Args:
        pdf_url: pdf url
        save_dir: save directory
        pdf_name: pdf name
        temp_file: if True, add timestamp to the file name

    Returns:

    """
    try:
        if pdf_name is None:
            pdf_name = pdf_url.split("/")[-1].replace('.', '_')
        save_dir = Path(save_dir) / Path(pdf_name).stem
        if temp_file:
            temp_t = datetime.now().strftime("%Y%m%d_%H%M")
            pdf_name = f"{temp_t}_{pdf_name}"
        pdf_path = Path(save_dir) / (str(pdf_name) + ".pdf")
        if not pdf_path.parent.exists():
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_path.exists():
            logging.info(f"pdf {pdf_path} already exists")
            return True, pdf_path
        print(f"download pdf from {pdf_url} to {pdf_path}")
        response = requests.get(pdf_url)
        print("request status  code:",response.status_code)
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        logging.info(f"download pdf from {pdf_url} to {pdf_path}")
        return True, pdf_path
    except Exception as e:
        error_msg = f"Donwload pdf from {pdf_url} error: {e}"
        logging.error(error_msg)
        return False, error_msg




