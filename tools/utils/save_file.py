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
                 pdf_name: Union[str,Path],
                 save_dir: Union[str,Path],
                 ):
    """
    download pdf from url
    Args:
        pdf_url: pdf url
        save_dir: save dir

    Returns: save path

        """
    try:
        response = requests.get(pdf_url)
        pdf_path = Path(save_dir) / pdf_name + ".pdf"
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"PDF downloaded successfully to {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None



if __name__ == '__main__':

    file_list = ["./data/111.pdf",Path("./data/222.pdf")]
    text_list = ["111","222"]
    print(save_mmd_file(save_texts=text_list,file_names=file_list,save_dirs="./res/out"))

