# Human-like Controllable Image Captioning with Verb-specific Semantic Roles

- Authors: Long Chen<sup>2,3*,</sup>    Zhihong Jiang<sup>1*</sup>    Jun Xiao\({}^{1\dagger}\)   Wei Liu<sup>4</sup>

- Affiliations: <sup>1</sup>Zhejiang University   <sup>2</sup>Tencent AI Lab   <sup>3</sup>Columbia University   <sup>4</sup>Tencent Data Platform<br><br>zjuchenlong@gmail.com, {zju_jiangzhihong, junx}@zju.edu.cn, wl2223@columbia.edu<br><br>denotes equal contributions,   <sup>&dagger;</sup> denotes the corresponding author.

![img](img/Evaluation on Diversity_0.png)

![img](img/Evaluation on Diversity_1.png)

## Abstract


This paper proposes a novel approach for controllable image captioning using Verb-specific Semantic Roles (VSR) as the control signal. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL component, object proposals are extracted, and similarity scores between semantic roles and proposal sets are calculated. The SSP component learns a reasonable sequence of sub-roles, and the Role-shift Caption Generation component generates the final caption by focusing on specific sub-roles and their grounded region sets. The approach is trained separately for each component using different training objectives. In the testing stage, the approach can be extended to multiple VSRs. Experimental results demonstrate the effectiveness of the proposed approach in achieving better controllability and generating diverse captions.
## Introduction


The proposed approach introduces Verb-specific Semantic Roles (VSR) as a control signal for controllable image captioning. VSR consists of a verb and semantic roles representing an activity and the entities involved. The approach comprises three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. GSRL grounds semantic roles to visual proposals, while SSP learns a sequence of sub-roles. Role-shift Caption Generation uses adaptive attention to control sub-role shifting and generate captions. The approach's mathematical details are captured in equations (1), (4), (5), (6), (7), (9), (10), (11), (12), and (13). Experimental results should be included to support the claims made.
## Related Work


The proposed approach in this paper focuses on Controllable Image Captioning (CIC) using Verb-specific Semantic Roles (VSR) as the control signal. The VSR consists of a verb and semantic roles that represent a targeted activity and the roles of entities involved in the activity. The approach includes three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation.

In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between each semantic role and proposal set. The top proposal sets are selected as the grounding results for the sub-roles. The SSP component consists of an S-level SSP and an R-level SSP. The S-level SSP learns a sequence of general semantic roles, while the R-level SSP ranks sub-roles within the same semantic role using a soft permutation matrix. The Role-shift Caption Generation component generates the final caption by focusing on specific sub-roles and their grounded region sets. An adaptive attention mechanism is used to control the shift of sub-roles, and a two-layer LSTM generates the words.

In the training stage, the GSRL, SSP, and captioning model are trained separately with respective objectives. The GSRL is trained with binary cross-entropy loss, the SSP with cross-entropy and mean square loss, and the captioning model with cross-entropy and reinforcement learning using a self-critical baseline. In the inference stage, the GSRL, SSP, and captioning model are sequentially used to generate captions. The framework can be extended to multiple VSRs by merging semantic structures and grounded region features.

The proposed approach contributes to controllable image captioning by introducing VSR as a control signal that considers event-compatibility and sample-suitability requirements. The use of GSRL, SSP, and Role-shift Caption Generation enables better controllability and the generation of diverse captions. The approach achieves state-of-the-art controllability on challenging benchmarks and generates captions with a better trade-off between quality and diversity.
## Proposed Approach
![img](img/Proposed Approach_0.png)

The proposed approach introduces Verb-specific Semantic Roles (VSR) as a control signal for generating customized captions in controllable image captioning. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL component, object proposals are extracted using an object detector, and similarity scores between semantic roles and proposal sets are calculated. The SSP component includes an S-level SSP and an R-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation component utilizes an adaptive attention mechanism and LSTM to generate captions based on the semantic structure and grounded region features. The training stage involves training the GSRL, SSP, and captioning model separately using various loss functions, while the inference stage uses the trained models to generate captions. The framework can also be extended to multiple VSRs as control signals. The paper could provide more specific details about the algorithms or models used in each component and include quantitative evaluations to demonstrate the effectiveness of the proposed approach.
### Controllable Caption Generation with VSR


The proposed approach in this paper focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as the control signal. It consists of three main components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL step, object proposals are extracted using an object detector, and a similarity score is calculated between each semantic role and proposal set. The SSP includes an S-level SSP and an R-level SSP, which learn the sequence of semantic roles at the sentence level and rank the sub-roles within each role. The Role-shift Caption Generation model generates the final caption by focusing on specific sub-roles and their corresponding grounded region sets. The training stage involves training each component separately using different loss functions, while the inference stage uses the trained models to generate captions. The proposed approach can also be extended to multiple VSRs as control signals.
### Training and Inference
![img](img/Training and Inference_0.png)

The proposed approach in this paper focuses on Controllable Image Captioning (CIC), aiming to generate image descriptions with designated control signals. The authors argue that existing objective control signals in CIC studies overlook two crucial characteristics: event compatibility and sample suitability. To address this, they propose a new control signal called Verb-specific Semantic Roles (VSR), which consists of a verb and semantic roles representing the targeted activity and the roles of entities involved. The proposed approach includes three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. GSRL grounds the semantic roles to visual proposals, SSP learns the descriptive semantic structure, and Role-shift Caption Generation generates the final captions. The training stage involves training each component separately, while the inference stage utilizes the GSRL, SSP, and captioning model sequentially. The proposed approach can be extended to multiple VSRs as control signals. Experimental results demonstrate that the framework achieves better controllability and generates diverse captions compared to strong baselines.
## Experiments



### Datasets and Metrics


The researchers conducted experiments using two datasets: Flickr30K Entities and COCO Entities. The Flickr30K Entities dataset is an extension of the Flickr30K dataset, where each noun phrase in the image descriptions is manually grounded with visual regions. It contains 31,000 images, each associated with five captions. The COCO Entities dataset, on the other hand, is an extension of the COCO dataset, where the annotations are automatically detected. It consists of 120,000 images, each with five captions.

To ensure consistency in the evaluation process, the researchers used the same splits as previous works for both datasets. However, they noted that a small percentage of samples (3.26% in COCO Entities and 0.04% in Flickr30K Entities) did not contain any verbs in their captions. These samples were dropped from the training and testing stages to maintain consistency with the assumption that there is at least one verb (activity) in each image.

It is worth mentioning that all baselines used the same visual regions as the models with VSRs, ensuring a fair comparison between different approaches.
### Implementation Details


The research paper provides details on the implementation of the proposed approach. The following are the key experimental techniques and procedures used:

1. **Proposal Generation and Grouping**: A Faster R-CNN model with ResNet-101 is employed to generate proposals for each image. The model is fine-tuned on the VG dataset. For COCO Entities, proposals are grouped based on their detected class labels. For Flickr30K Entities, each proposal is treated as a separate proposal set.

2. **VSR Annotations**: Since there are no ground truth semantic role annotations for CIC datasets, a pretrained Semantic Role Labeling (SRL) tool is used to annotate verbs and semantic roles for each caption. These annotations are considered as ground truth. A verb dictionary is created for each dataset, and the base form of each detected verb is used. There are 24 types of semantic roles for all verbs.

3. **Experimental Settings**: For the S-level SSP, the multi-head attention has 8 heads, and the transformer's hidden size is set to 512. The length of the transformer is set to 10. For the R-level SSP, the maximum number of entities for each role is set to 10. During reinforcement learning (RL) training of the captioning model, the CIDEr-D score is used as the training reward.

These experimental techniques and procedures ensure the scientificity, reliability, and validity of the technical methods used in the research. The use of a well-established object detection model for proposal generation, along with the fine-tuning on a relevant dataset, enhances the accuracy of the proposed approach. The VSR annotations provide ground truth information for training and evaluation. The specific experimental settings, such as the number of heads in multi-head attention and the maximum number of entities, are carefully chosen to optimize the performance of the models. Overall, these techniques and procedures contribute to the successful implementation and evaluation of the proposed approach.
### Evaluation on Controllability
![img](img/Evaluation on Controllability_0.png)

![img](img/Evaluation on Controllability_1.png)

![img](img/Evaluation on Controllability_2.png)

### Evaluation on Controllability
To evaluate the controllability of the proposed framework, the authors used the VSR aligned with ground truth captions as control signals. Several baselines were compared, including C-LSTM, C-UpDn, and SCT. The evaluation metrics used were accuracy-based metrics and diversity-based metrics.

In the first setting, given a VSR and grounded visual regions aligned with the ground truth caption, two diverse captions were generated using the SSP and role-shift captioning model. The same set of visual regions was used for two strong baselines: BS and SCT. In the second setting, new VSRs were constructed by randomly sampling a subset of all semantic roles for each verb.

The quantitative results showed that the diverse captions generated by the proposed framework had higher accuracy than the baselines, while the diversity was slightly behind SCT. The framework achieved a better trade-off between quality and diversity in diverse image captioning.

Visualizations of the generated captions for two images with different VSRs were also provided, demonstrating that the captions effectively followed the given VSR and exhibited significant diversity based on the VSR.
## Conclusions & Future Works


The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In GSRL, object proposals are extracted and similarity scores are calculated between semantic roles and proposal sets. SSP includes S-level and R-level models for learning a sequence of general semantic roles and ranking sub-roles within the same semantic role. Role-shift Caption Generation utilizes adaptive attention and LSTM to generate captions, focusing on specific sub-roles. The approach aims to improve controllability in image captioning.