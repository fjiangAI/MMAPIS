import  torch
from functools import singledispatch
from typing import List
import io
from pathlib import Path
import fitz
import logging
import zipfile
import sys
import requests
from datetime import datetime
import json
import tiktoken
import re



def get_best_gpu(choice_list=None):
    total_gpus = torch.cuda.device_count()

    if not choice_list:
        choice_list = list(range(total_gpus))

    choice_list = list(filter(lambda x: x < total_gpus, choice_list))
    max_memory = 0
    best_gpu = None

    for gpu in choice_list:
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = meminfo.free
        if free_memory > max_memory:
            max_memory = free_memory
            best_gpu = gpu

    return best_gpu, max_memory/1024**3



def get_batch_size():
    if torch.cuda.is_available():
        best_gpu, free_memory = get_best_gpu([0, 1])
        BATCH_SIZE = int(
            free_memory * 0.3
        )
        # BATCH_SIZE = int(
        #     torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        # )
        logging.info(f"Best GPU: {best_gpu}. Batch size: {BATCH_SIZE}")
        if BATCH_SIZE < 1:
            logging.error("Not enough GPU memory, can not load model.")
            sys.exit(1)
    else:
        # don't know what a good value is here. Would not recommend to run on CPU
        BATCH_SIZE = 5
        logging.warning("No GPU found. Conversion on CPU is very slow.")
    return BATCH_SIZE


def get_pdf_list(uploaded_files:List):
    pdf_list, file_names = [], []
    for uploaded_file in uploaded_files:
        tmp_pdf_list, tmp_file_names = process_single_file(uploaded_file = uploaded_file)
        print("tmp_file_names",tmp_file_names)
        pdf_list.extend(tmp_pdf_list)
        file_names.extend(tmp_file_names)
    return pdf_list, file_names


def process_single_file(uploaded_file):
    if uploaded_file.name.endswith(".zip"):
        logging.info(f"get zip file{uploaded_file.name}")
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            file_content = [zip_ref.read(name) for name in file_names]
        return file_content, file_names

    ## not recommend to use ra, dur to unrar error
    elif uploaded_file.name.endswith(".rar"):
        logging.error("rar file not support")
        # file_bytes = io.BytesIO(uploaded_file.getvalue())
        # with rarfile.RarFile(file_bytes, 'r') as rar_ref:
        #     file_names = rar_ref.namelist()
        #     file_content = [rar_ref.read(name) for name in file_names]
        # return file_content, file_names

    elif uploaded_file.name.endswith(".pdf"):
        logging.info(f"get pdf file{uploaded_file.name}")

        file_content = [uploaded_file.getvalue()]
        file_names = [uploaded_file.name]
        return file_content, file_names
    else:
        logging.error("File type not supported.")
        sys.exit(1)


@singledispatch
def get_pdf_doc(pdf,proxy=None,headers=None,pdf_name=None):
    # pdf options:
    # 1. [str] url
    # 2. [str] path
    # 3. [Path] path
    # 4. bytes
    raise NotImplementedError("Unsupported type")

@get_pdf_doc.register(str)
def _(pdf,proxy=None,headers=None,pdf_name=None):

    if "http" in pdf:
        logging.info(f"Opening PDF to rasterize from url format")
        name = pdf.split("/")[-1].replace('.', '_')
        # self.size = len(fitz.open(stream=urllib.request.urlopen(pdf).read(), filetype="pdf"))
        pdf_doc_obj = fitz.open(stream=requests.get(pdf, proxies=proxy, headers=headers).content, filetype="pdf")
        size = len(pdf_doc_obj)

    else:
        logging.info(f"Opening PDF to rasterize from str format")
        name = pdf.split("/")[-1]
        pdf_doc_obj = fitz.open(Path(pdf), filetype="pdf")
        size = len(pdf_doc_obj)

    return pdf_doc_obj, size, name

@get_pdf_doc.register(bytes)
def _(pdf,proxy=None,headers=None,pdf_name=None):

    logging.info(f"Opening PDF to rasterize from bytes format")
    pdf_doc_obj = fitz.open(stream=pdf, filetype="pdf")
    size = len(pdf_doc_obj)
    name = pdf_name if pdf_name else f"unk_pdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return pdf_doc_obj, size, name

@get_pdf_doc.register(Path)
def _(pdf,proxy=None,headers=None,pdf_name=None):
    logging.info(f"Opening PDF to rasterize from Path format")
    pdf_doc_obj = fitz.open(pdf, filetype="pdf")
    name = pdf.name
    size = len(pdf_doc_obj)
    return pdf_doc_obj, size, name



def custom_response_handler(response: requests.Response,
                            func_name:str=''):

    try:
        if 'application/json' in response.headers['content-type']:
            json_info = response.json()
            if response.status_code == 200 and json_info:
                logging.info(f"{func_name} process success")
                return json_info['message']
            elif response.status_code == 400:
                logging.error(f"request body error[{func_name}], status: {json_info.get('status', response.status_code)}")
            elif response.status_code == 500:
                logging.error(f"internal error[{func_name}], status: {json_info.get('status', response.status_code)}")
            else:
                logging.error(f"unknown response error[{func_name}], status code: {response.status_code}")
            error_msg =  json_info.get('status', '') + " "+ json_info.get('message', 'Unknown error')
            error_status = json_info.get('status', response.status_code)
            json_msg = {'error': error_msg, 'status': error_status}
            return json_msg

        elif 'audio/mp3' in response.headers['content-type']:
            logging.info(f"{func_name} process success")
            return response.content

        else:
            logging.error(f"unknown response type:{response.headers.get('content-type','unknown')} error[{func_name}], status code: {response.status_code}")
            error_msg = response.text
            return {'error': error_msg, 'status': response.status_code}

    except Exception as e:
        logging.error(f"response error[{func_name}], an unexpected error occurred: {str(e)}")
        return {'error': 'Unknown error', 'status': response.status_code if response else 502}


def dict_filter_none(d:dict):
    return {k:v for k,v in d.items() if v is not None}


def handle_request(url:str,parameters = None,proxy=None, headers = None):
        success = False
        response = None
        try:
            if proxy is None:
                raw_response = requests.post(url, headers=headers, json=parameters)
            else:
                raw_response = requests.post(url, headers=headers, json=parameters, proxies=proxy)

            raw_response.raise_for_status()
            response = json.loads(raw_response.content.decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            success = True
        except requests.exceptions.RequestException as e:
            content = f"Request Error: {str(e)}"
        except json.JSONDecodeError as e:
            content = f"JSON Decode Error: {str(e)}"
        except KeyError as e:
            content = f"KeyError: {str(e)}"
        except Exception as e:
            content = f"Unexpected Error: {str(e)}"

        return response,content, success


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
    """
        Returns the number of tokens used by a list of messages.
        Args:
            messages: A list of messages. Each message is:
                dict (with keys "role" and "content".)
                string (in which case the role is assumed to be "user".)
                list (already normalized into a list of dicts.)
            model: The model to use. Defaults to "gpt-3.5-turbo-16k-0613".
        Returns:
            The number of tokens used by the messages.
    """
    # try:
    #     encoding = tiktoken.encoding_for_model(model)
    encode_model = None
    map_dict = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "text-embedding-ada": "cl100k_base",
        "text-davinci": "p50k_base",
        "Codex": "p50k_base",
        "davinci": "p50k_base",
    }
    for key in map_dict.keys():
        if key in model:
            encode_model = map_dict[key]
            break

    if not encode_model:
        logging.error(f"model {model} not found,load default model: cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")
    else:
        encoding = tiktoken.get_encoding(encode_model)

    if model in "gpt-3.5-turbo-16k-0613":  # note: future models may deviate from this
        if isinstance(messages, dict):
            messages = [messages]
        elif isinstance(messages, list):
            pass
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def strip_title(title):
    title = re.sub(r'^[^a-zA-Z]*', '', title)
    title = re.sub(r'[^a-zA-Z]*$', '', title)
    return title


if __name__ == "__main__":
    text = """
    self.messages: [{'role': 'system', 'content': "You excel in crafting academic paper summary, and you need to strictly follow the prescribed method steps to segmentally summarize the given academic paper. Adhere strictly to the following system steps:\n\n1. **Draft Initial Summary**\n\n   
    - Start with the tag `[Raw Summary Content]`. Ensure a detailed analysis of the paper's significant sections, emphasizing comprehensive coverage of all crucial aspects.\n\n   - Use professional and understandable language to enhance readers' understanding of the research.\n\n   - Avoid nebulous and overgeneralized descriptions.\n\n2. **Provide Constructive Feedback**\n\n   Start with the tag `[Feedback content]` and suggest improvements to the generated draft. Clearly point out necessary improvements in the following areas:\n\n      (i) **Redundancy**: Identify repeated statements and propose strategies for simplification to enhance conciseness.\n \n      (ii) **Missing Entities**: Highlight omitted crucial information (** including key mathematical formulas if provided**) and provide supplementary suggestions.\n\n      (iii) **Quantitative Suggestions**: Advocate the usage of numerical measurements as substitutes for descriptive generalizations.\n \n      (iv) **Off-topic Content**:  Locate and suggest the removal of sections straying from the main topic.\n\n      (v) **Consistency**: Propose improvements in language fluency.\n\n      (vi) **Authenticity**: Identify statements inconsistent with the source content and provide suggestions for improvement.\n\n3. **Generate Final Summary** \n\n   - Start with the tag `[Final Summary Content]` and modify the initial draft based on feedback to produce the final segmented summary.\n    \n   - Ensure comprehensive presentation of all key findings and specific details(especially key mathematical formulas if provided) of the paper.\n    \n   - Use precise expressions, eliminate redundancy, ambiguity content.\n   \n   - Use clear, technical, and unbiased **academic expression** suitable for professional readers.\n\n   -   Keep **appropriate Markdown formatting** for a visually appealing output.\n   \nThroughout the entire process, underscore key points and meet the user role content requirements and queries, and make clear each summary section's contribution to the paper as a whole."}, {'role': 'user', 'content': ["Distill the essence of the paper's abstract, underscoring the research focus, complex dilemmas tackled and solutions provided. Enumerate the primary attributions professed in the abstract, along with demonstrable impacts if presented.\n\n```## Abstract\nControllable Image Captioning (CIC) -- generating image descriptions following designated control signals -- has received unprecedented attention over the last few years. To emulate the human ability in controlling caption generation, current CIC studies focus exclusively on control signals concerning objective properties, such as contents of interest or descriptive patterns. However, we argue that almost all existing objective control signals have overlooked two indispensable characteristics of an ideal control signal: 1) Event-compatible: all visual contents referred to in a single sentence should be compatible with the described activity. 2) Sample-suitable: the control signals should be suitable for a specific image sample. To this end, we propose a new control signal for CIC: Verb-specific Semantic Roles (VSR). VSR consists of a verb and some semantic roles, which represents a targeted activity and the roles of entities involved in this activity. Given a designated VSR, we first train a grounded semantic role labeling (GSRL) model to identify and ground all entities for each role. Then, we propose a semantic structure planner (SSP) to learn human-like descriptive semantic structures. Lastly, we use a role-shift captioning model to generate the captions. Extensive experiments and ablations demonstrate that our framework can achieve better controllability than several strong baselines on two challenging CIC benchmarks. Besides, we can generate multi-level diverse captions easily. The code is available at: [https://github.com/mad-red/VSR-guided-CIC](https://github.com/mad-red/VSR-guided-CIC).```", 'Concisely summarize the core essence of the introduction. Clarify the academic background and research motivation, outlining how this section establishes the foundational framework for the entire paper. Emphasize key research questions and objectives, providing rationale for the research questions and logical connections with existing literature.\n\n\n\n```## 1 Introduction\nImage captioning, _i.e_., generating fluent and meaningful descriptions to summarize the salient contents of an image, is a classic proxy task for comprehensive scene understand\n\n\n\n\n\ning [21]. With the release of several large scale datasets and advanced encoder-decoder frameworks, current captioning models plausibly have already achieved "super-human" performance in all accuracy-based evaluation metrics. However, many studies have indicated that these models tend to produce generic descriptions, and fail to control the caption generation process as humans, _e.g._, referring to different contents of interest or descriptive patterns. In order to endow the captioning models with human-like controllability, a recent surge of efforts [16, 10, 19, 76, 46, 75, 27, 20] resort to introducing extra control signals as constraints of the generated captions, called **Controllable Image Captioning** (CIC). As a byproduct, the CIC models can easily generate diverse descriptions by feeding different control signals.\n\nEarly CIC works mainly focus on _subjective_ control signals, such as sentiments [40], emotions [41, 22], and personality [14, 52], _i.e._, the linguistic styles of sentences. Although these stylized captioning models can eventually produce style-related captions, they remain hard to control the generation process effectively and precisely. To further improve the controllability, recent CIC works gradually put a more emphasis on _objective_ control signals. More specifically, they can be coarsely classified into two categories: 1) _Content-controlled_: the control signals are about the contents of interest which need to be described. As the example shown in Figure 1 (a), given the region set (\\(\\raisebox{-0.86pt}{\\includegraphics[]{figures/1.eps}}\\raisebox{-0.86pt}{ \\includegraphics[]{figures/2.eps}}\\raisebox{-0.86pt}{\\includegraphics[]{figures/ 3.eps}}\\)) as a control signal, we hope that the generated caption can cover all regions (_i.e._, man, wave, and surfboard). So far, various types of content-controlled signals have been proposed, such as visual relations [27], object regions [16, 34], scene graphs [10, 76], and mouse trace [46]. 2) _Structure-controlled_: the control signals are about the semantic structures of sentences. For instance, the length-level [19], part-of-speech tags [20], or attributes [77] of the sentence (cf. Figure 1 (b)) are some typical structure-controlled signals.\n\nNevertheless, all existing objective control signals (_i.e._, both content-controlled and structure-controlled) have overlooked two indispensable characteristics of an ideal control signal towards "human-like" controllable image captioning: 1) **Event-compatible**: all visual contents referred to in a single sentence should be compatible with the described activity. Imaging how humans describe images -- our brains always quickly structure a descriptive pattern like "sth do sth at someplace" first, and then fill in the detailed description [54, 44, 29, 69], _i.e._, we have subconsciously made sure that all the mentioned entities are event-compatible (_e.g._, man, wave, surfboard are all involved in activity riding in Figure 1 (a)). To further see the negative impact of dissatisfying this requirement, suppose that we deliberately utilize two more objects (hand and sky, _i.e._, \\(\\raisebox{-0.86pt}{\\includegraphics[]{figures/1.eps}}\\raisebox{-0.86pt}{ \\includegraphics[]{figures/2.eps}}\\)) as part of the control signal, and the model generates an incoherent and illogical caption. 2) **Sample-suitable**: the control signals should be suitable for the specific image sample. By "suitable", we mean that there do exist reasonable descriptions satisfying the control signals, _e.g._, a large length-level may not be suitable for an image with a very simple scene. Unfortunately, it is always very difficult to decide whether a control signal is sample-suitable in advance. For example in Figure 1 (b), although the two control signals (_i.e._, length-levels 3 and 4) are quite close, the quality of respectively generated captions varies greatly.\n\nIn this paper, we propose a new event-oriented objective control signal, _Verb-specific Semantic Roles_ (VSR), to meet both event-compatible and sample-suitable requirements simultaneously. VSR consists of a verb (_i.e._, predicate [8]) and some user-interested semantic roles [30]. As shown in Figure 2, the verb captures the scope of a salient activity in the image (_e.g._, eating), and the corresponding semantic roles1 (_e.g._, agent, food, container, and tool) categorize how objects participate in this activity, _i.e._, a child (agent) is eating (activity) a pancake (food) from a plate (container) with a fork (tool). Thus, VSR is designed to guarantee that all the mentioned entities are event-compatible. Meanwhile, unlike the existing structure-controlled signals which directly impose constraints on the generated captions, VSR only restricts the involved semantic roles, which is theoretically suitable for all the images with the activity, _i.e._, sample-suitable.\n\nFootnote 1: We use PropBank-style annotations of semantic roles (_e.g._, Arg0, Arg1) in all experiments (cf. Figure 1). The FrameNet-style annotations of semantic roles (_e.g._, Agent) here are just for a more intuitive illustration. In the PropBank-style annotations, Arg denotes “argument”, MNR denotes “manner”, DIR denotes “directional”, and LOC denotes “location”. We leave more details in the supplementary material.\n\nIn order to generate sentences with respect to the designated VSRs, we first train a grounded semantic role labeling (GSRL) model to identify and ground all entities for each role. Then, we propose a semantic structure planner (SSP) to rank the given verb and semantic roles, and output some human-like descriptive semantic structures, _e.g._, Arg0\\({}_{\\text{reader}}\\)\\(-\\)read\\(-\\)Arg1\\({}_{\\text{thing}}\\)\\(-\\)LOC in Figure 1 (c). Finally, we combine the grounded entities and semantic structures, and use an RNN-based role-shift captioning model to generate the captions by sequentially focusing on different roles.\n\nAlthough these are no available captioning datasets with the VSR annotations, they can be easily obtained by off-the-shelf semantic role parsing toolkits [51]. Extensive experiments on two challenging CIC benchmarks (_i.e._, COCO\n\n\n\n\n\nEntities [16] and Flickr30K Entities [45]) demonstrate that our framework can achieve better controllability given designated VSRs than several strong baselines. Moreover, our framework can also realize diverse image captioning and achieve a better trade-off between quality and diversity.\n\nIn summary, we make three contributions in this paper:\n\n1. We propose a new control signal for CIC: Verb-specific Semantic Roles (VSR). To the best of our knowledge, VSR is the first control signal to consider both event-compatible and sample-suitable requirements2. Footnote 2: When using control signals extracted from GT captions, existing control signals can always meet both requirements and generate reasonable captions. However, in more general settings (_e.g._, construct control signals without GT captions), the form of VSR is more human-friendly, and it is easier to construct signals which meet both requirements compared with all existing forms of control signals, which is the main advantage of VSR.\n2. We can learn human-like verb-specific semantic structures automatically, and abundant visualization examples demonstrate that these patterns are reasonable.\n3. We achieve state-of-the-art controllability on two challenging benchmarks, and generate diverse captions by using different verbs, semantic roles, or structures.```', 'Summarize academic progress and background knowledge in the research field, explaining how this paper builds upon existing literature and contributes innovatively. Evaluate the strengths and limitations of previous research, emphasizing the clear innovation and contribution of this study.\n\n\n\n```## 2 Related Work\n**Controllable Image Captioning.** Compared with conventional image captioning [61, 66, 9, 25, 13], CIC is a more challenging task, which needs to consider extra constraints. Early CIC works are mostly about stylized image captioning, _i.e._, constraints are the linguistic styles of sentences. According to the requirements of parallel training samples, existing solutions can be divided into two types: models using parallel stylized image-caption data [40, 11, 52, 1] or not [22, 41]. Subsequently, the community gradually shifts the emphasis to controlling described contents [16, 75, 27, 10, 76, 46, 34] or structures [20, 19, 73, 74] of the sentences. In this paper, we propose a novel control signal VSR, which is the first control signal to consider both the event-compatible and sample-suitable requirements.\n\n**Diverse and Distinctive Image Captioning.** Diverse image captioning, _i.e._, describing the image contents with diverse wordings and rich expressions, is an essential property of human-like captioning models. Except from feeding different control signals to the CIC models, other diverse captioning methods can be coarsely grouped into four types: 1) GAN-based [17, 50, 31]: they use a discriminator to force the generator to generate human-indistinguishable captions. 2) VAE-based [63, 7]: the diversity obtained with them is by sampling from a learned latent space. 3) RL-based [38]: they regard diversity as an extra reward in the RL training stage. 4) BS-based [60]: they decode a list of diverse captions by optimizing a diversity-augmented objective.\n\nMeanwhile, distinctive image captioning is another close research direction [18, 58, 36, 35, 62], which aims to generate discriminative and unique captions for individual images. Unfortunately, due to the subjective nature of diverse and distinctive captions, effective evaluation remains as an open problem, and several new metrics are proposed, such as SPICE-U [65], CIDErBtw [62], self-CIDEr [64], word recall [56], mBLEU [50]. In this paper, we can easily generate diverse captions in both lexical-level and syntactic-level.\n\n**Semantic Roles in Images.** Inspired from the semantic role labeling task [6] in NLP, several tasks have been proposed to label the roles of each object in an activity in an image:\n\n_Visual Semantic Role Labeling (VSRL)_, also called situation recognition, is a generalization of action recognition and human-object interaction, which aims to label an image with a set of verb-specific action _frames_[71]. Specifically, each action frame describes details of the activity captured by the verb, and it consists of a fixed set of verb-specific semantic roles and their corresponding values. The values are the entities or objects involved in the activity and the semantic roles categorize how objects participate in the activity. The current VSRL methods [23, 71, 39, 32, 70, 55, 15] usually learn an independent action classifier first, and then model the role inter-dependency by RNNs or GNNs.\n\n_Grounded Semantic Role Labeling (GSRL)_, also called grounded situation recognition, builds upon the VSRL task, which requires the models not only to label a set of frames, but also to localize each role-value pair in the image [47, 53, 68, 23]. In this paper, we use the GSRL model as a bridge to connect the control signals (VSR) and related regions. To the best of our knowledge, we are the first captioning work to benefit from the verb lexicon developed by linguists.```', 'Summarize the 3 Proposed Approach section of the paper. Focus on the core points in the paper\'s narrative thread, the central viewpoints, and key content in this section. Thoroughly explore the unique significance of the key technical details for academic contributions.\n\n\n\n```## 3 Proposed Approach\nFor human-like controllable image captioning, we first propose the Verb-specific Semantic Roles (VSR) as the control signal for generating customized captions. As shown in Figure 3, we formally represent a control signal VSR as:\n\n\\[\\mathcal{VSR}=\\{v,<s_{1},n_{1}>,...,<s_{m},n_{m}>\\}, \\tag{1}\\]\n\nwhere \\(v\\) is a **verb** capturing the scope of a salient activity in the image (_e.g._, ride), \\(s_{i}\\) is a **semantic role** of verb \\(v\\) (_e.g._, LOC), and \\(n_{i}\\) is the number of interested entities in the role \\(s_{i}\\). For example, for \\(\\mathcal{VSR}=\\{\\text{ride},<\\text{Arg0},1>,<\\text{Arg1},1>,<\\text{Loc},2>\\}\\), we hope to generate a caption which not only focuses on describing the ride activity, but also contains one entity respectively in the role \\(\\text{Arg0}_{\\text{rider}}\\) and \\(\\text{Arg1}_{\\text{xeed}}\\), and two entities in the role LOC. Thus, VSR can effectively control the amount of information carried in the whole sentence and each role, _i.e._, the level of details.\n\nIt is convenient to construct VSRs automatically or manually. For the verbs, they can be accurately predicted by an off-the-shelf action recognition network with a predefined verb vocabulary. For the verb-specific semantic roles, they can be easily retrieved from the verb lexicon such as PropBank or FrameNet. Then, the users can easily select a subset of roles or an automatic sampling to generate a subset of roles, and randomly assign the entity number for each role.\n\nGiven an image \\(\\mathbf{I}\\) and a control signal \\(\\mathcal{VSR}\\), the controllable image captioning model aims to describe \\(\\mathbf{I}\\) by a textual sentence \\(\\mathbf{y}=\\{y_{1},...,y_{T}\\}\\), _i.e._, modeling the probability \\(p(\\mathbf{y}|\\mathbf{I},\\mathcal{VSR})\\). Inspired from the human habit of describing images, we decompose this task into two steps: structuring a descriptive pattern and filling in detailed captions:\n\n\\[p(\\mathbf{y}|\\mathbf{I},\\mathcal{VSR})=p(\\mathbf{y}|\\text{pattern})p(\\text{pattern}|\\mathbf{I},\\mathcal{VSR}). \\tag{2}\\]\n\nFurther, we utilize two sequences \\(\\mathcal{S}=(s_{1}^{b},...,s_{K}^{b})\\) and \\(\\mathcal{R}=(\\mathbf{r}_{1},...,\\mathbf{r}_{K})\\) to model the descriptive patterns. Specifically, \\(\\mathcal{S}\\) is a semantic structure of the sentence and each \\(s_{i}^{b}\\in\\mathcal{S}\\) is a sub-role. By "sub-role", we mean that each role \\(s_{i}\\in\\mathcal{VSR}\\) can be divided into \\(n_{i}\\) sub-roles, and when \\(n_{i}=1\\), role \\(s_{i}\\) itself is a sub-role. Thus, VSR in Figure 3 can be rewritten as Arg0, Arg1, LOC-1, and LOC-2. \\(\\mathcal{R}\\) is a sequence of visual features of the corresponding grounded entities for each sub-role in \\(\\mathcal{S}\\) (_e.g._, \\(\\mathbf{r}_{i}\\) is the features of visual regions referring to \\(s_{i}^{b}\\)). Particularly, for presentation conciseness, we regard the verb in \\(\\mathcal{VSR}\\) as a special type of sub-role, and since there are no grounded visual regions referring to the verb, we use the global image feature as the grounded region feature in \\(\\mathcal{R}\\). Meanwhile, we use \\(\\mathcal{\\tilde{R}}\\) to denote a set of all elements in the sequence \\(\\mathcal{R}\\). Thus, we further decompose this task into three components:\n\n\\[p(\\mathbf{y}|\\mathbf{I},\\mathcal{VSR})=\\underbrace{p(\\mathbf{y}|\\mathcal{S},\\mathcal{R})}_ {\\text{Captioner}}\\underbrace{p(\\mathcal{S},\\mathcal{R}|\\mathcal{\\tilde{R}}, \\mathcal{VSR})}_{\\text{SSP}}\\underbrace{p(\\mathcal{\\tilde{R}}|\\mathbf{I},\\mathcal{ VSR})}_{\\text{GSRL}}. \\tag{3}\\]\n\nIn this section, we first introduce each component of the whole framework of the VSR-guided controllable image captioning model sequentially in Section 3.1 (cf. Figure 3), including a grounded semantic role labeling (GSRL) model, a semantic structure planner (SSP), and a role-shift captioning model. Then, we demonstrate the details about all training objectives and the inference stage in Section 3.2, including extending from a single VSR to multiple VSRs.\n\n### Controllable Caption Generation with VSR\n\n#### 3.1.1 Grounded Semantic Role Labeling (GSRL)\n\nGiven an image \\(\\mathbf{I}\\), we first utilize an object detector [48] to extract a set of object proposals \\(\\mathcal{B}\\). Each proposal \\(\\mathbf{b}_{i}\\in\\mathcal{B}\\) is associated with a visual feature \\(\\mathbf{f}_{i}\\) and a class label \\(c_{i}\\in\\mathcal{C}\\). Then, we group all these proposals into \\(N\\) disjoint sets, _i.e._, \\(\\mathcal{B}=\\{\\mathcal{B}_{1},...,\\mathcal{B}_{N}\\}\\)3, and each proposal set \\(\\mathcal{B}_{i}\\) consists of one or more proposals. In this GSRL step, we need to refer each sub-role in the \\(\\mathcal{VSR}\\) to a proposal set in \\(\\mathcal{B}\\). Specifically, we calculate the similarity score \\(a_{ij}\\) between semantic role \\(s_{i}\\) and proposal set \\(\\mathcal{B}_{j}\\) by:\n\nFootnote 3: Due to different annotation natures of specific CIC datasets, we group proposals by different principles. Details are shown in Section 4.2.\n\n\\[\\mathbf{q}_{i}=\\left[\\mathbf{e}_{v}^{g};\\mathbf{e}_{s_{i}}^{g};\\mathbf{\\tilde{f}}\\right],\\quad a _{ij}=F_{a}(\\mathbf{q}_{i},\\mathbf{\\tilde{f}}_{j}), \\tag{4}\\]\n\nwhere \\(\\mathbf{e}_{v}^{g}\\) and \\(\\mathbf{e}_{s_{i}}^{g}\\) are the word embedding features of verb \\(v\\) and semantic role \\(s_{i}\\), \\(\\mathbf{\\tilde{f}}\\) and \\(\\mathbf{\\tilde{f}}_{j}\\) represent the average-pooled visual features of proposal set \\(\\mathcal{B}\\) and \\(\\mathcal{B}_{j}\\), [;] is a concatenation operation, and \\(F_{a}\\) is a learnable similarity function4.\n\nFootnote 4: For conciseness, we leave the details in the supplementary material.\n\nAfter obtaining the grounding similarity scores \\(\\{a_{ij}\\}\\) between semantic role \\(s_{i}\\) and all proposal sets \\(\\{\\mathcal{B}_{j}\\}\\), we then select the top \\(n_{i}\\) proposal sets with the highest scores as the grounding results for all sub-roles of \\(s_{i}\\). \\(\\mathcal{\\tilde{R}}\\) in Eq. (3) is the set of visual features of all grounded proposal sets.\n\n#### 3.1.2 Semantic Structure Planner (SSP)\n\nSemantic structure planner (SSP) is a hierarchical semantic structure learning model, which aims to learn a reasonable sequence of sub-roles \\(\\mathcal{S}\\). As shown in Figure 3, it consists of two subnets: an S-level SSP and an R-level SSP.\n\n**S-level SSP.** The sentence-level (S-level) SSP is a coarse-grained structure learning model, which only learns a sequence of all involved general semantic roles (including the verb) in \\(\\mathcal{VSR}\\) (_e.g._, ride, Arg0\\({}_{\\text{rider}}\\), Arg1\\({}_{\\text{steed}}\\) and LOC in Figure 3). To this end, we formulate this sentence-level structure learning as a role sequence generation task, as long as we constrain that each output role token belongs to the\n\n\n\n\n\ngiven role set and each role can only appear once. Specifically, we utilize a three-layer Transformer [57]5 to calucate the probability of roles \\(p(s_{i}|\\mathcal{VSR})\\) at each time step \\(t^{4}\\):\n\nFootnote 5: More comparison results between Transformer and Sinkhorn networks [42, 16] are left in supplementary material.\n\n\\[\\begin{split}\\mathbf{H}&=\\text{Transformer}_{\\text{enc} }\\left(\\{\\text{FC}_{a}(\\mathbf{e}_{v}^{i}+\\mathbf{e}_{s_{i}}^{i})\\}\\right),\\\\ p(s_{t}|\\mathcal{VSR})&=\\text{Transformer}_{\\text{dec }}\\left(\\mathbf{H},\\mathbf{e}_{s_{c1}}^{o}\\right),\\end{split} \\tag{5}\\]\n\nwhere Transformer, are the encoder (enc) and decoder (dec) of the standard multi-head transformer. \\(\\mathbf{e}_{v}^{i}\\) and \\(\\mathbf{e}_{s_{i}}^{i}\\) are the word embedding features of verb \\(v\\) and semantic role \\(s_{j}\\), respectively. FC\\({}_{a}\\) is a learnable fc-layer to obtain the embedding of each input token. \\(\\mathbf{e}_{s_{c1}}^{o}\\) is the sequence of embeddings of previous roles. Based on \\(p(s_{t}|\\mathcal{VSR})\\), we can predict a role at time step \\(t\\) and obtain an initial role sequence,, \\(\\text{Arg}_{\\text{order}}-\\text{ride}-\\text{Arg}_{\\text{t-ated}}-\\text{LOC}\\) in Figure 3.\n\n**R-level SSP.** The role-level (R-level) SSP is a fine-grained structure model which aims to rank all sub-roles within the same semantic role (, LOC-1 and LOC-2 are two sub-roles of role Loc in Figure 3). Since the only differences among these sub-roles are the grounded visual regions, we borrow ideas from the Sinkhorn networks [42, 16], which use a differentiable Sinkhorn operation to learn a _soft_ permutation matrix \\(\\mathbf{P}\\). Specifically, for each role \\(s_{i}\\) with multiple sub-roles (, \\(n_{i}>1\\)), we first select all the corresponding grounded proposal sets for these sub-roles, denoted as \\(\\hat{\\mathcal{B}}=\\{\\hat{\\mathcal{B}}_{1},...,\\hat{\\mathcal{B}}_{n_{i}}\\}\\). And for each proposal \\(\\mathbf{b}_{*}\\in\\hat{\\mathcal{B}}\\), we encode a feature vector \\(\\mathbf{z}_{*}=[\\mathbf{z}_{*}^{v};\\mathbf{z}_{*}^{s_{i}};\\mathbf{z}_{*}^{l}]\\), where \\(\\mathbf{z}_{*}^{v}\\) is a transformation of its visual feature \\(\\mathbf{f}_{*}\\), \\(\\mathbf{z}_{*}^{s_{i}}\\) is the word embedding feature of the semantic role \\(s_{i}\\), and \\(\\mathbf{z}_{*}^{l}\\) is a 4-d encoding of the spatial position of proposal \\(\\mathbf{b}_{*}\\). Then, we transform each feature \\(\\mathbf{z}_{*}\\) into \\(n_{i}\\)-d, and average-pooled all features among the same proposal set,, we can obtain an \\(n_{i}\\)-d feature for each \\(\\hat{\\mathcal{B}}_{i}\\). We concatenate all these features to get an \\(n_{i}\\times n_{i}\\) matrix \\(\\mathbf{Z}\\). Finally, we use the Sinkhorn operation to obtain the soft permutation matrix \\(\\mathbf{P}^{4}\\):\n\n\\[\\mathbf{P}=\\text{Sinkhorn}(\\mathbf{Z}). \\tag{6}\\]\n\nAfter the two SSP subnets (, S-level and R-level), we can obtain the semantic structure \\(\\mathcal{S}\\) (cf. Eq. (3)). Based on the sequence of \\(\\mathcal{S}\\) and the set of proposal features \\(\\tilde{\\mathcal{R}}\\) from the GSRL model, we re-rank \\(\\tilde{\\mathcal{R}}\\) based on \\(\\mathcal{S}\\) and obtain \\(\\mathcal{R}\\).\n\n#### 3.1.3 Role-shift Caption Generation\n\nGiven the semantic structure sequence \\(\\mathcal{S}=(s_{1}^{b},...,s_{K}^{b})\\) and corresponding proposal feature sequence \\(\\mathcal{R}=(\\mathbf{r}_{1},...,\\mathbf{r}_{K})\\), we utilize a two-layer LSTM to generate the final caption \\(\\mathbf{y}\\). At each time step, the model focuses on one specific sub-role \\(s_{t}^{b}\\) and its grounded region set \\(\\mathbf{r}_{t}\\), and then generates the word \\(y_{t}\\). Therefore, we take inspirations from previous CIC methods [16, 10], and predict two distributions simultaneously: \\(p(g_{t}|\\mathcal{S},\\mathcal{R})\\) for controlling the shift of sub-roles, and \\(p(y_{t}|\\mathcal{S},\\mathcal{R})\\) to predict the distribution of a word.\n\nAs for the role-shift, we use an adaptive attention mechanism [37] to predict the probability of shifting6:\n\nFootnote 6: Note that the proposed method uses the same semantic role \\(s_{t}^{b}\\) and its grounded region set \\(\\mathbf{r}_{t}\\), and then generates the word \\(y_{t}\\). Therefore, we take inspirations from previous CIC methods [16, 10], and predict two distributions simultaneously: \\(p(g_{t}|\\mathcal{S},\\mathcal{R})\\) for controlling the shift of sub-roles, and \\(p(y_{t}|\\mathcal{S},\\mathcal{R})\\) to predict the distribution of a word.\n\nAs for the role-shift, we use an adaptive attention mechanism [37] to predict the probability of shifting7:\n\nFootnote 7: Note that the proposed method uses the same semantic role \\(s_{t}^{b}\\) and its grounded region set \\(\\mathbf{r}_{t}\\), and then generates the word \\(y_{t}\\). Therefore, we take inspirations from previous CIC methods [16, 10], and predict two distributions simultaneously: \\(p(g_{t}|\\mathcal{S},\\mathcal{R})\\) for controlling the shift of sub-roles, and \\(p(y_{t}|\\mathcal{S},\\mathcal{R})\\) to predict the distribution of a word.\n\n\\[\\alpha_{t}^{g},\\mathbf{\\alpha}_{t}^{r},\\mathbf{s}\\mathbf{r}_{t}^{g}=\\text{AdaptiveAttn}_{a}( \\mathbf{x}_{t},\\mathbf{r}_{t}), \\tag{7}\\]\n\nwhere \\(\\text{AdaptiveAttn}_{a}\\) is an adaptive attention network, \\(\\mathbf{x}_{t}\\) is the input query for attention, \\(\\mathbf{s}\\mathbf{r}_{t}^{g}\\) is a spatial vector, \\(\\alpha_{t}^{g}\\) and \\(\\mathbf{\\alpha}_{t}^{r}\\) are the attention weights for the spatial vector and region features, respectively. We directly use attention weight \\(\\alpha_{t}^{g}\\) as the probability of shifting sub-roles,, \\(p(g_{t}|\\mathcal{S},\\mathcal{R})=\\alpha_{t}^{g}\\). Based on probability \\(p(g_{t}|\\mathcal{S},\\mathcal{R})\\), we can sample a gate value \\(g_{j}\\in\\{0,1\\}\\), and the focused sub-role at time step \\(t\\) is:\n\n\\[s_{t}^{b}\\leftarrow\\mathcal{S}[i],\\text{where }i=\\min\\left(1+\\sum_{j=1}^{t-1}g_{j},K \\right). \\tag{8}\\]\n\nDue to the special nature of sub-role "verb", we fix \\(g_{t+1}=1\\) when \\(s_{t}^{b}\\) is the verb.\n\nFor each sub-role \\(s_{t}^{b}\\), we use the corresponding proposal set features \\(\\mathbf{r}_{t}\\) and a two-layer LSTM to generate word \\(y_{t}\\):\n\n\\[\\begin{split}\\mathbf{h}_{t}^{1}&=\\text{LSTM}_{1}\\left( \\mathbf{h}_{t-1}^{1},\\{y_{t-1},\\bar{\\mathbf{f}},\\mathbf{h}_{t-1}^{2}\\}\\right),\\\\ \\mathbf{h}_{t}^{2}&=\\text{LSTM}_{2}\\left(\\mathbf{h}_{t-1}^{ 2},\\{\\mathbf{h}_{t}^{1},\\mathbf{c}_{t}\\}\\right),\\\\ y_{t}&\\sim p(y_{t}|\\mathcal{S},\\mathcal{R})=\\text{ FC}_{b}(\\mathbf{h}_{t}^{2}),\\end{split} \\tag{9}\\]\n\nwhere \\(\\mathbf{h}_{t}^{1}\\) and \\(\\mathbf{h}_{t}^{2}\\) are hidden states of the first- and second-layer LSTM (, LSTM\\({}_{1}\\) and LSTM\\({}_{2}\\)), FC\\({}_{b}\\) is a learnable fc-layer, and \\(\\mathbf{c}_{t}\\) is a context vector. To further distinguish the textual and visual words, we use another adaptive attention network to obtain the context vector \\(\\mathbf{c}_{t}\\)6:\n\nFootnote 6: Note that the proposed method uses the same semantic role \\(s_{t}^{b}\\) and its grounded region set \\(\\mathbf{c}_{t}\\), and then generates the word \\(y_{t}\\).\n\n\\[\\begin{split}\\alpha_{t}^{v},\\mathbf{\\alpha}_{t}^{r},\\mathbf{s}\\mathbf{r}_{t}^{ v}&=\\text{AdaptiveAttn}_{b}(\\mathbf{x}_{t},\\mathbf{r}_{t}),\\\\ \\mathbf{c}_{t}&=\\alpha_{t}^{v}\\cdot\\mathbf{s}\\mathbf{r}_{t}^{ v}+\\sum_{i}\\mathbf{\\alpha}_{t,i}^{r}\\cdot\\mathbf{r}_{t,i},\\end{split} \\tag{10}\\]\n\nwhere \\(\\mathbf{x}_{t}\\) is the query for adaptive attention (, the input of the LSTM\\({}_{1}\\)), \\(\\mathbf{s}\\mathbf{r}_{t}^{v}\\) is a spatial vector, and \\(\\alpha_{t}^{v}\\) and \\(\\mathbf{\\alpha}_{t}^{r}\\) are the attention weights for the spatial vector and region features.\n\n### Training and Inference\n\n**Training Stage.** In the training stage, we train the three components (GSRL, SSP and captioning model) separately:\n\n_Training objective of GSRL._ For the GSRL model, we use a binary cross-entropy (BCE) loss between the predicted similarity scores \\(\\hat{a}_{ij}\\) and its ground truth \\(a_{ij}^{*}\\) as the training loss:\n\n\\[L_{\\text{GSRL}}=\\sum_{ij}\\text{BCE}(\\hat{a}_{ij},a_{ij}^{*}). \\tag{11}\\]\n\n_Training objective of SSP._ For S-level SSP, we use a cross-entropy (XE) loss between prediction \\(\\hat{s}_{t}\\) and its ground truth \\(s_{t}^{*}\\) as the training objective. For R-level SSP, we use a mean square (MSE) loss between prediction \\(\\mathbf{\\hat{P}}_{t}\\) and its ground truth \\(\\mathbf{P}_{t}^{*}\\) as the training objective:\n\n\\[L_{\\text{SSP}}^{S}=\\sum_{t}\\text{XE}(\\hat{s}_{t},s_{t}^{*}),L_{\\text{SSP}}^{R}= \\sum_{t}\\mathbf{1}_{(n_{t}>1)}\\text{MSE}(\\mathbf{\\hat{P}}_{t},\\mathbf{P}_{t}^{*}), \\tag{12}\\]here \\(\\mathbf{1}_{(n_{t}>1)}\\) is an indicator function, being 1 if \\(n_{t}>1\\) and 0 otherwise.\n\n_Training objective of captioning model._ We follow the conventions of previous captioning works and use a two-stage training scheme: XE and RL stages. In the XE stage, we use an XE loss between predicted words and ground truth words as the training loss. In the RL stage, we use a self-critical baseline [49]. At each step, we sample from \\(p(y_{t}|\\mathcal{S},\\mathcal{R})\\) and \\(p(g_{t}|\\mathcal{S},\\mathcal{R})\\) to obtain the next word \\(y_{t+1}\\) and sub-role \\(s^{b}_{t+1}\\). Then we calculate the reward \\(r(\\mathbf{y}^{s})\\) of the sampled sentence \\(\\mathbf{y}^{s}\\). Baseline \\(b\\) is the reward of the greedily generated sentence. Thus, the gradient expression of the training loss is:\n\n\\[\\nabla_{\\theta}L=-(r(\\mathbf{y}^{s})-b)(\\nabla_{\\theta}\\log p(\\mathbf{y}^{s})+\\nabla_{ \\theta}\\log p(\\mathbf{g}^{s})), \\tag{13}\\]\n\nwhere \\(\\mathbf{g}^{s}\\) is the sequence of role-shift gates.\n\n**Inference.** In testing stage, given an image and one \\(\\mathcal{VSR}\\), we sequentially use the GSRL, SSP, and captioning model to generate the final captions. Meanwhile, our framework can be easily extended from one \\(\\mathcal{VSR}\\) to multiple \\(\\mathcal{VSR}\\)s as the control signal. Taking an example of two \\(\\mathcal{VSR}\\)s, we first use GSRL and SSP to obtain semantic structures and grounded regions features: \\((\\mathcal{S}^{a},\\mathcal{R}^{a})\\) and \\((\\mathcal{S}^{b},\\mathcal{R}^{b})\\). Then, as shown in Figure 4, we merge them by two steps4: (a) find the sub-roles in both \\(\\mathcal{S}^{a}\\) and \\(\\mathcal{S}^{b}\\) which refer to the same visual regions (_e.g._, \\(s^{a}_{1}\\) and \\(s^{b}_{1}\\) refer to the same proposal set); (b) insert all other sub-roles between the nearest two selected sub-roles (_e.g._, \\(s^{*}_{2}\\) are still between \\(s^{*}_{1}\\) and \\(s^{*}_{3}\\)). Concerning the order of sub-roles from different verbs, we follow the rank of two verbs (_e.g._, \\(s^{a}_{2}\\) is in front of \\(s^{b}_{2}\\)).\n\nFootnote 4: All baselines use the same visual regions as models with VSRs.```', 'Thoroughly analyze the core experimental techniques and procedures used in the research. Clearly state the innovative technical means and paths used in data collection and processing, and analyze their key role in achieving research goals. Focus on specific experimental setups, data acquisition and processing techniques, and how to ensure the scientificity, reliability, and validity of technical methods.\n\n\n\n```## 4 Experiments\n### Datasets and Metrics\n\n**Flickr30K Entities [45].** It builds upon the Flickr30K [72] dataset, by manually grounding each noun phrase in the descriptions with one or more visual regions. It consists of 31,000 images, and each image is associated with five captions. We use the same splits as [26] in our experiments.\n\n**COCO Entities [16].** It builds upon the COCO [12] dataset which consists of 120,000 images and each image is annotated with five captions. Different from Flickr30K Entities where all grounding entities are annotated by humans, all annotations in COCO Entities are detected automatically. Especially, they align each entity to all the detected proposals with the same object class.\n\nAlthough we only assume that there exists at least one verb (_i.e._, activity) in each image; unfortunately, there are still a few samples (_i.e._, 3.26% in COCO Entities and 0.04% in Flickr30K Entities) having no verbs in their captions. We use the same split as [16] and further drop the those samples with no verb in the training and testing stages5. We will try to cover these extreme cases and leave it for future work.\n\nFootnote 5: All baselines use the same visual regions as models with VSRs.\n\n### Implementation Details\n\n**Proposal Generation and Grouping.** We utilize a Faster R-CNN [48] with ResNet-101 [24] to obtain all proposals for each image. Especially, we use the model released by [3], which is finetuned on VG dataset [28]. For COCO Entities, since the "ground truth" annotations for each noun phrase are the proposals with the same class, we group the proposals by their detected class labels. But for Flickr30K Entities, we directly regard each proposal as a proposal set.\n\n**VSR Annotations.** Since there are no ground truth semantic role annotations for CIC datasets, we use a pretrained SRL tool [51] to annotate verbs and semantic roles for each caption, and regard them as ground truth annotations. For each detected verb, we convert it into its base form and build a verb dictionary for each dataset. The dictionary sizes for COCO and Flickr30K are 2,662 and 2,926, respectively. There are a total of 24 types of semantic roles for all verbs.\n\n**Experimental Settings.** For the S-level SSP, the head number of multi-head attention is set to 8, and the hidden size of the transformer is set to 512. The length of the transformer is set to 10. For the R-level SSP, we set the maximum number of entities for each role to 10. For the RL training of the captioning model, we use CIDEr-D [59] score as the training reward. Due to the limited space, we leave more detailed parameter settings in the supplementary material.\n\n### Evaluation on Controllability\n\n**Settings.** To evaluate the controllability of proposed framework, we followed the conventions of prior CIC works [16, 76, 10], and utilized the VSR aligned with ground truth captions as the control signals. Specifically, we compared the proposed framework with several carefully designed baselines6: 1) **C-LSTM**: It is a Controllable LSTM model [61]. Given the features of all grounded visual regions, it first averages all region features, and then uses an LSTM to generate the captions. 2) **C-UpDn**: It is a Controllable UpDn model [3], which uses an adaptive attention to generate the captions. 3) **SCT**[16]: It regards the set of visual regions as a control signal, and utilizes a chunk-shift captioning model to generate the captions. 4) **Ours _w/o_ verb**: We ablate our model by removing the verb information in both the SSP\n\n\n\n\n\n[MISSING_PAGE_FAIL:7]\n\ntwo settings: 1) Given a VSR and grounded visual regions of each role aligned with the ground truth caption, we first use an SSP to select two semantic structures, and then respectively generate two diverse captions. For fair comparisons, 
    we utilize the same set of visual regions on two strong baselines: a) **BS**: an UpDn model uses beam search to produce two captions, and b) **SCT**: an SCT model takes a permutation of all region sets to generate two captions. 2) For each verb, we can randomly sample a subset of all semantic roles to construct new VSRs. Specifically, we sample two more sets of semantic roles, and generate two diverse captions for each role set following the same manner.\n\n**Evaluation Metrics.** We used two types of metrics to evaluate the diverse captions: 1) Accuracy-based: we followed the conventions of the previous works [16, 20, 63] and reported the best-1 accuracy, _i.e._, the generated caption with the maximum score for each metric is chosen. Analogously, we evaluate the generated captions against the single ground truth caption. 2) Diversity-based: we followed [10] and used two metrics which only focus on the language similarity: Div-n (D-n) [4, 20] and self-CIDEr (s-C) [64].\n\n**Quantitative Results.** The quantitative results are reported in Table 2. From Table 2, we can observe that the diverse captions generated by our framework in both two settings have much higher accuracy (_e.g._, CIDEr 267.3 vs. 222.5 in SCT), and that the diversity is slightly behind SCT (_e.g._, self-CIDEr 67.0 vs. 69.1 in SCT). This is because SCT generates captions by randomly shuffling regions. Instead, we tend to learn more reasonable structures. Thus, we can achieve much higher results on accuracy, _i.e._, our method can achieve a better trade-off between quality and diversity on diverse image captioning than the two strong baselines.\n\n**Visualizations.** We further illustrate the generated captions of two images with different VSRs in Figure 7. The captions are generated effectively according to the given VSR, and the diversity of VSR leads to significant diverse captions.```', 'Highlight the conclusion of this section, emphasizing key points and significance, clearly stating the research contributions and prospects for future research. Subdivide parallel statements, focusing on the significant contributions of this section to the entire paper, the response to the main thematic question, and areas requiring further improvement and expansion.\n\n\n\n```## 5 Conclusions & Future Works\nIn this paper, we argued that all existing objective control signals for CIC have overlooked two indispensable characteristics: event-compatible and sample-suitable. To this end, we proposed a novel control signal called VSR. VSR consists of a verb and several semantic roles, _i.e._, all components are guaranteed to be event-compatible. Meanwhile, VSR only restricts the involved semantic roles, which is also sample-suitable for all the images containing the activity. We have validated the effectiveness of VSR through extensive experiments. Moving forward, we will plan to 1) design a more effective captioning model to benefit more from the VSR signals; 2) extend VSR to other controllable text generation tasks, _e.g._, video captioning [67]; 3) design a more general framework to cover the images without verbs.\n\n**Acknowledgements.** This work was supported by the National Natural Science Foundation of China (U19B2043,61976185), Zhejiang Natural Science Foundation (LR19F020002), Zhejiang Innovation Foundation (2019R52002), and Fundamental Research Funds for Central Universities.\n\n\n\n\n\n\n```']}]
    """
    print(num_tokens_from_messages(model="gpt-3.5-turbo-16k-0613",messages=text))