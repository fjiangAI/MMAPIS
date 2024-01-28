# Human-like Controllable Image Captioning with Verb-specific Semantic Roles

- Authors: Long Chen<sup>2,3*,</sup>    Zhihong Jiang<sup>1*</sup>    Jun Xiao\({}^{1\dagger}\)   Wei Liu<sup>4</sup>

- Affiliations: <sup>1</sup>Zhejiang University   <sup>2</sup>Tencent AI Lab   <sup>3</sup>Columbia University   <sup>4</sup>Tencent Data Platform<br><br>zjuchenlong@gmail.com, {zju_jiangzhihong, junx}@zju.edu.cn, wl2223@columbia.edu<br><br>denotes equal contributions,   <sup>&dagger;</sup> denotes the corresponding author.

![img](img/Evaluation on Diversity_0.png)

![img](img/Evaluation on Diversity_1.png)

## Abstract


This paper introduces a novel approach called Verb-specific Semantic Roles (VSR) for controllable image captioning. The VSR control signal consists of a verb and semantic roles, capturing the activity and the roles of entities involved in the image. The proposed approach includes three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In the GSRL component, an object detector is used to extract object proposals, and similarity scores between semantic roles and proposal sets are calculated. The SSP component consists of S-level and R-level subnets, which learn the sequence of semantic roles and rank the sub-roles within each role. The Role-shift Caption Generation component generates captions by focusing on specific sub-roles and their grounded regions.

The training stage involves training each component separately using objective functions such as binary cross-entropy, cross-entropy, mean square, and self-critical baseline loss. In the testing stage, the GSRL, SSP, and captioning model are sequentially used to generate captions. The proposed approach can also be extended to multiple VSRs as control signals.

The key technical contributions of this paper include the introduction of VSR as a control signal for controllable image captioning, the development of GSRL for grounding semantic roles, the SSP for learning semantic structures, and the role-shift caption generation model. Experimental results demonstrate the effectiveness of the proposed approach in achieving better controllability and generating diverse captions.
## Introduction


The proposed approach in this paper focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL component, object proposals are associated with visual features and class labels, and similarity scores between semantic roles and proposal sets are calculated. This allows for grounding the control signals in the image. The SSP component learns a reasonable sequence of sub-roles using a sentence-level SSP and a role-level SSP. This helps in structuring the descriptive pattern of the captions. The Role-shift Caption Generation component generates the final caption by focusing on specific sub-roles and grounded region sets, enabling controllability in the caption generation process. The training stage involves training the three components separately, optimizing their respective objectives. In the inference stage, the GSRL, SSP, and captioning model are used to generate captions based on the control signals. The framework can be extended to multiple VSRs as control signals, allowing for more diverse and customized captions. Overall, the proposed approach offers a novel and effective method for achieving controllable image captioning.
## Related Work


The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In GSRL, object proposals are extracted, and similarity scores between semantic roles and proposal sets are calculated. SSP learns the semantic structure by predicting role sequences and ranking sub-roles within each role. Role-shift Caption Generation generates captions by focusing on specific sub-roles and their grounded region sets. The training stage involves training each component separately, and the inference stage utilizes the trained models to generate captions. The approach can be extended to multiple VSRs as control signals.
## Proposed Approach
![img](img/Proposed Approach_0.png)

The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. GSRL predicts similarity scores between semantic roles and proposal sets, while SSP learns the sequence of semantic roles and ranks sub-roles within each role. The Role-shift Caption Generation component generates captions based on the semantic structure sequence and proposal feature sequence using adaptive attention and LSTM. The proposed approach can handle multiple VSRs as control signals. However, more explanation and examples of GSRL and quantitative results would enhance the understanding and validation of the proposed approach.
### Controllable Caption Generation with VSR


The proposed approach in this paper focuses on controllable image captioning with Verb-specific Semantic Roles (VSR) as the control signal. The framework consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between semantic roles and proposal sets. The SSP component consists of an S-level SSP and an R-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation component utilizes LSTM-based models to generate captions by focusing on specific sub-roles and their grounded regions.

During training, the GSRL, SSP, and captioning model are trained separately with objectives such as binary cross-entropy loss, cross-entropy loss, and mean square loss. In the inference stage, the framework can be extended to multiple VSRs as control signals. The proposed approach achieves better controllability in generating captions and allows for diverse captioning.

The significance of this approach lies in its consideration of event compatibility and sample suitability in control signals, which previous methods have overlooked. The use of VSRs enables more precise and effective control over the caption generation process. The experimental results demonstrate the effectiveness of the proposed framework in achieving better controllability and generating diverse captions.
### Training and Inference
![img](img/Training and Inference_0.png)

The proposed approach in this paper focuses on Controllable Image Captioning (CIC) and introduces a novel control signal called Verb-specific Semantic Roles (VSR) to generate customized captions. The VSR consists of a verb and semantic roles, representing a targeted activity and the roles of entities involved in this activity. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In GSRL, an object detector is used to extract object proposals, and a similarity score is calculated between semantic roles and proposal sets. The SSP consists of an S-level SSP and an R-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation utilizes a two-layer LSTM to generate captions based on the semantic structure sequence and proposal feature sequence. 

During training, separate training objectives are used for each component. In the inference stage, the GSRL, SSP, and captioning model are sequentially used to generate captions. The approach can also be extended to multiple VSRs as control signals. 

The proposed approach makes several contributions, including the introduction of VSR as a control signal for CIC, the learning of human-like semantic structures, and achieving state-of-the-art controllability on challenging benchmarks. The approach also enables the generation of diverse captions by using different verbs, semantic roles, or structures.
## Experiments



### Datasets and Metrics


The research utilizes two datasets, namely Flickr30K Entities and COCO Entities, for evaluation. The Flickr30K Entities dataset is an extension of the Flickr30K dataset, where each noun phrase in the image descriptions is manually grounded with one or more visual regions. It contains 31,000 images, each associated with five captions. The COCO Entities dataset, on the other hand, is an extension of the COCO dataset and consists of 120,000 images, with each image annotated with five captions. In this dataset, all annotations are automatically detected.

To ensure the scientificity, reliability, and validity of the technical methods, the research follows the same splits as previous works for both datasets. However, there are a few samples in both datasets (3.26% in COCO Entities and 0.04% in Flickr30K Entities) that do not have any verbs in their captions. To address this issue, the research drops these samples during the training and testing stages. It is noted that all baselines use the same visual regions as the models with VSRs, ensuring a fair comparison between different methods.
### Implementation Details


The implementation details of the research are as follows:

1. Proposal Generation and Grouping: The Faster R-CNN model with ResNet-101 is used to generate proposals for each image. The model is finetuned on the VG dataset. For COCO Entities, proposals are grouped based on their detected class labels. For Flickr30K Entities, each proposal is treated as a separate proposal set.

2. VSR Annotations: Since there are no ground truth semantic role annotations for CIC datasets, a pretrained Semantic Role Labeling (SRL) tool is used to annotate verbs and semantic roles for each caption. These annotations are considered as ground truth. A verb dictionary is built for each dataset, with sizes of 2,662 for COCO and 2,926 for Flickr30K. There are a total of 24 types of semantic roles for all verbs.

3. Experimental Settings: For the S-level SSP, the multi-head attention has 8 heads, and the hidden size of the transformer is set to 512. The length of the transformer is set to 10. For the R-level SSP, the maximum number of entities for each role is set to 10. During reinforcement learning (RL) training of the captioning model, the CIDEr-D score is used as the training reward.

These implementation details ensure the scientificity, reliability, and validity of the technical methods used in the research. The use of a well-established object detection model for proposal generation, along with the finetuning on a relevant dataset, ensures accurate and reliable proposal generation. The VSR annotations obtained from a pretrained SRL tool provide a reliable source of ground truth for training and evaluation. The specific experimental settings, such as the parameters used in the SSP models and the choice of training reward in RL, are carefully chosen to optimize performance and achieve the research goals.
### Evaluation on Controllability
![img](img/Evaluation on Controllability_0.png)

![img](img/Evaluation on Controllability_1.png)

![img](img/Evaluation on Controllability_2.png)

The researchers evaluated the controllability of their proposed framework by comparing it with several baselines. They used the Verb-specific Semantic Roles (VSR) aligned with ground truth captions as control signals. The baselines included the C-LSTM model, the C-UpDn model, and the SCT model. They conducted evaluations in two settings: (1) given a VSR and grounded visual regions aligned with ground truth captions, they used the SSP to select two semantic structures and generated diverse captions; and (2) for each verb, they randomly sampled a subset of semantic roles to construct new VSRs and generated diverse captions for each role set.

The evaluation metrics used were accuracy-based metrics and diversity-based metrics. The accuracy-based metrics included CIDEr, which measures caption quality, and the diversity-based metrics included Div-n (D-n) and self-CIDEr (s-C), which focus on language similarity.

The quantitative results showed that the diverse captions generated by the proposed framework had higher accuracy compared to the SCT baseline. The diversity was slightly behind SCT due to the framework's focus on learning more reasonable structures.

The researchers also provided visualizations of the generated captions for two images with different VSRs, demonstrating the effective generation of captions according to the given VSR and the significant diversity achieved through different VSRs.

Overall, the experimental techniques and procedures used in this research involved comparing the proposed framework with baselines, evaluating controllability using various metrics, and providing visualizations to demonstrate the effectiveness and diversity of the generated captions. These techniques ensured the scientificity, reliability, and validity of the technical methods used in the study.
## Conclusions & Future Works


The proposed approach in this paper introduces Verb-specific Semantic Roles (VSR) as a control signal for controllable image captioning. The authors argue that existing control signals overlook the crucial characteristics of event compatibility and sample suitability. VSR consists of a verb and semantic roles, ensuring that all components are compatible with the described activity. The approach includes grounded semantic role labeling (GSRL), semantic structure planning (SSP), and role-shift caption generation. However, specific mathematical formulas and details of these components are not provided in this section. The authors validate the effectiveness of VSR through extensive experiments, although no quantitative results are presented in this section. Future work includes improving the captioning model, extending VSR to other text generation tasks, and designing a more general framework.