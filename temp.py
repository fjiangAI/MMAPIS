import os
import re
import requests
from tqdm import tqdm





file_dir = "./res/cs_2017/raw_mmd"
file_list = os.listdir(file_dir)
link_ls = [f"https://arxiv.org/pdf/{x.replace('.mmd','').replace('_','.')}.pdf" for x in file_list]

save_path = file_dir+"/pdf"
# 确保保存目录存在
if not os.path.exists(save_path):
    os.makedirs(save_path)
link_ls = tqdm(link_ls,position=0,leave=True)
# 对于列表中的每个链接
for link in link_ls:
    # 发送一个HTTP请求
    response = requests.get(link)

    # 从链接中获取文件名
    file_name = os.path.basename(link)

    # 打开一个新的文件进行写入
    with open(os.path.join(save_path, file_name), 'wb') as file:
        file.write(response.content)

print("所有文件下载完毕！")