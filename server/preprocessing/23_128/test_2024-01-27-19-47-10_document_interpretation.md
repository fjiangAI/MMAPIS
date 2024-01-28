# Human-like Controllable Image Captioning with Verb-specific Semantic Roles

![img](img/Evaluation on Diversity_0.png)

![img](img/Evaluation on Diversity_1.png)

## Abstract


This paper introduces a novel approach called Verb-specific Semantic Roles (VSR) for controllable image captioning. The VSR control signal consists of a verb and semantic roles, capturing the activity and the roles of entities involved in the image. The proposed approach includes three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. The key technical contributions of this paper include the introduction of VSR as a control signal for controllable image captioning, the development of GSRL for grounding semantic roles, the SSP for learning semantic structures, and the role-shift caption generation model. Experimental results demonstrate the effectiveness of the proposed approach in achieving better controllability and generating diverse captions.
## Introduction


The proposed approach in this paper focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. The GSRL component uses an object detector to extract object proposals and calculate similarity scores between semantic roles and proposal sets, grounding the control signals in the image. The SSP component learns a reasonable sequence of sub-roles using a sentence-level SSP and a role-level SSP, structuring the descriptive pattern of the captions. The Role-shift Caption Generation component generates the final caption by focusing on specific sub-roles and grounded region sets, enabling controllability in the caption generation process. The proposed approach offers a novel and effective method for achieving controllable image captioning.
## Related Work


The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. The GSRL component calculates similarity scores between semantic roles and proposal sets, grounding the control signals. The SSP component learns the sequence of semantic roles and ranks sub-roles within each role. The Role-shift Caption Generation generates captions by focusing on specific sub-roles and their grounded region sets. The approach can be extended to multiple VSRs as control signals.
## Proposed Approach
![img](img/Proposed Approach_0.png)

The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. The GSRL component extracts object proposals using an object detector and calculates similarity scores between semantic roles and proposal sets. The SSP component learns the sequence of semantic roles and ranks sub-roles within each role. The Role-shift Caption Generation component generates captions based on the semantic structure sequence and proposal feature sequence using adaptive attention and LSTM. The proposed approach can handle multiple VSRs as control signals.
### Controllable Caption Generation with VSR


The proposed approach focuses on controllable image captioning with Verb-specific Semantic Roles (VSR) as the control signal. The framework consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between semantic roles and proposal sets. The SSP component consists of an S-level SSP and an R-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation component utilizes LSTM-based models to generate captions by focusing on specific sub-roles and their grounded regions.

During training, the GSRL, SSP, and captioning model are trained separately with objectives such as binary cross-entropy loss, cross-entropy loss, and mean square loss. In the inference stage, the framework can be extended to multiple VSRs as control signals. The proposed approach achieves better controllability in generating captions and allows for diverse captioning.

The significance of this approach lies in its consideration of event compatibility and sample suitability in control signals, which previous methods have overlooked. The use of VSRs enables more precise and effective control over the caption generation process. The experimental results demonstrate the effectiveness of the proposed framework in achieving better controllability and generating diverse captions.
### Training and Inference
![img](img/Training and Inference_0.png)

The proposed approach focuses on Controllable Image Captioning (CIC) and introduces a novel control signal called Verb-specific Semantic Roles (VSR) to generate customized captions. The VSR consists of a verb and semantic roles, representing a targeted activity and the roles of entities involved in this activity. The approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In GSRL, an object detector is used to extract object proposals, and a similarity score is calculated between semantic roles and proposal sets. The SSP consists of an S-level SSP and an R-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation utilizes a two-layer LSTM to generate captions based on the semantic structure sequence and proposal feature sequence. 

During training, separate training objectives are used for each component. In the inference stage, the GSRL, SSP, and captioning model are sequentially used to generate captions. The approach can also be extended to multiple VSRs as control signals. 

The proposed approach makes several contributions, including the introduction of VSR as a control signal for CIC, the learning of human-like semantic structures, and achieving state-of-the-art controllability on challenging benchmarks. The approach also enables the generation of diverse captions by using different verbs, semantic roles, or structures.
## Experiments



### Datasets and Metrics


The research utilizes two datasets, namely Flickr30K Entities and COCO Entities, for evaluation. The Flickr30K Entities dataset is an extension of the Flickr30K dataset, where each noun phrase in the image descriptions is manually grounded with one or more visual regions. The COCO Entities dataset, on the other hand, is an extension of the COCO dataset and consists of 120,000 images, with each image annotated with five captions. In this dataset, all annotations are automatically detected.

The evaluation metrics used were accuracy-based metrics and diversity-based metrics. The accuracy-based metrics included CIDEr, which measures caption quality, and the diversity-based metrics included Div-n (D-n) and self-CIDEr (s-C), which focus on language similarity.
### Implementation Details


The implementation details of the research are as follows:

1. Proposal Generation and Grouping: The Faster R-CNN model with ResNet-101 is used to generate proposals for each image. The model is finetuned on the VG dataset. For COCO Entities, proposals are grouped based on their detected class labels. For Flickr30K Entities, each proposal is treated as a separate proposal set.

2. VSR Annotations: A pretrained Semantic Role Labeling (SRL) tool is used to annotate verbs and semantic roles for each caption. These annotations are considered as ground truth. A verb dictionary is built for each dataset.

3. Experimental Settings: For the S-level SSP, the multi-head attention has 8 heads, and the hidden size of the transformer is set to 512. The length of the transformer is set to 10. For the R-level SSP, the maximum number of entities for each role is set to 10.

These implementation details ensure the scientificity, reliability, and validity of the technical methods used in the research.
### Evaluation on Controllability
![img](img/Evaluation on Controllability_0.png)

![img](img/Evaluation on Controllability_1.png)

![img](img/Evaluation on Controllability_2.png)

The researchers evaluated the controllability of their proposed framework by comparing it with several baselines. They used the Verb-specific Semantic Roles (VSR) aligned with ground truth captions as control signals. The baselines included the C-LSTM model, the C-UpDn model, and the SCT model. They conducted evaluations in two settings: (1) given a VSR and grounded visual regions aligned with ground truth captions, they used the SSP to select two semantic structures and generated diverse captions; and (2) for each verb, they randomly sampled a subset of semantic roles to construct new VSRs and generated diverse captions for each role set.

The quantitative results showed that the diverse captions generated by the proposed framework had higher accuracy compared to the SCT baseline. The diversity was slightly behind SCT due to the framework's focus on learning more reasonable structures.

The researchers also provided visualizations of the generated captions for two images with different VSRs, demonstrating the effective generation of captions according to the given VSR and the significant diversity achieved through different VSRs.

Overall, the experimental techniques and procedures used in this research involved comparing the proposed framework with baselines, evaluating controllability using various metrics, and providing visualizations to demonstrate the effectiveness and diversity of the generated captions. These techniques ensured the scientificity, reliability, and validity of the technical methods used in the study.
## Conclusions & Future Works


The proposed approach in this paper introduces Verb-specific Semantic Roles (VSR) as a control signal for controllable image captioning. The authors argue that existing control signals overlook the crucial characteristics of event compatibility and sample suitability. VSR consists of a verb and semantic roles, ensuring that all components are compatible with the described activity. The approach includes grounded semantic role labeling (GSRL), semantic structure planning (SSP), and role-shift caption generation. The authors validate the effectiveness of VSR through extensive experiments, although no quantitative results are presented in this section. Future work includes improving the captioning model, extending VSR to other text generation tasks, and designing a more general framework.