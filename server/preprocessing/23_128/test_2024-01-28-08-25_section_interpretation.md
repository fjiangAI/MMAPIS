# Human-like Controllable Image Captioning with Verb-specific Semantic Roles

- Authors: Long Chen<sup>2,3*,</sup>    Zhihong Jiang<sup>1*</sup>    Jun Xiao\({}^{1\dagger}\)   Wei Liu<sup>4</sup>

- Affiliations: <sup>1</sup>Zhejiang University   <sup>2</sup>Tencent AI Lab   <sup>3</sup>Columbia University   <sup>4</sup>Tencent Data Platform<br><br>zjuchenlong@gmail.com, {zju_jiangzhihong, junx}@zju.edu.cn, wl2223@columbia.edu<br><br>denotes equal contributions,   <sup>&dagger;</sup> denotes the corresponding author.

![img](img/Evaluation on Diversity_0.png)

![img](img/Evaluation on Diversity_1.png)

## Abstract


This paper proposes a novel approach for controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. The VSR consists of a verb and semantic roles that represent the targeted activity and the roles of entities involved in the activity. The proposed approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation.

In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between each semantic role and proposal set. The top proposal sets with the highest scores are selected as grounding results for each sub-role.

The SSP component includes an S-level SSP and an R-level SSP. The S-level SSP learns a sequence of general semantic roles, while the R-level SSP ranks sub-roles within the same semantic role. Sinkhorn networks are used to learn a soft permutation matrix for ranking the sub-roles.

The Role-shift Caption Generation component generates the final caption based on the semantic structure sequence and proposal feature sequence. An adaptive attention mechanism is used to control the shift of sub-roles, and a two-layer LSTM is employed to generate words based on the focused sub-role and grounded region set.

During training, each component is trained separately with specific objectives. The GSRL model is trained using binary cross-entropy loss, the SSP models use cross-entropy and mean square loss, and the captioning model is trained using cross-entropy loss in the XE stage and self-critical baseline in the RL stage.

In the testing stage, the proposed framework can generate captions based on one or multiple VSRs as control signals. Multiple VSRs can be merged by finding sub-roles that refer to the same visual regions and inserting other sub-roles between them.

Overall, this approach introduces VSR as a new control signal for controllable image captioning and combines GSRL, SSP, and Role-shift Caption Generation to achieve better controllability and generate diverse captions. The proposed framework is evaluated on challenging benchmarks and demonstrates superior performance compared to baselines.
## Introduction


The paper proposes a novel approach for controllable image captioning using Verb-specific Semantic Roles (VSR) as control signals. The VSR consists of a verb and semantic roles that represent a targeted activity and the roles of entities involved in the activity. The proposed approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between each semantic role and proposal set. The SSP component includes a sentence-level SSP and a role-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation component uses an adaptive attention mechanism to control the shift of sub-roles and generate the final captions.

During the training stage, the GSRL, SSP, and captioning model are trained separately using respective training objectives. In the testing stage, the framework can be extended to multiple VSRs as control signals. The proposed approach achieves better controllability in generating customized captions and can easily generate diverse captions. Experimental results on challenging benchmarks demonstrate the effectiveness of the proposed framework.

The key technical contributions of this paper include the introduction of VSR as a control signal, the use of GSRL for grounding semantic roles to visual regions, the SSP for learning semantic structures, and the role-shift caption generation model. The proposed approach addresses the limitations of existing objective control signals and provides a more human-like controllability in image captioning.
## Related Work


The proposed approach focuses on controllable image captioning using Verb-specific Semantic Roles (VSR) as the control signal. It consists of three main components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. GSRL utilizes an object detector to extract object proposals and calculates a similarity score between semantic roles and proposal sets. SSP includes an S-level SSP and an R-level SSP to learn the sequence of semantic roles and rank sub-roles within each role. Role-shift Caption Generation generates captions based on the learned semantic structure and grounded region features. The approach can be extended to handle multiple VSRs as control signals. The summary lacks key mathematical formulas related to similarity score calculation and training objectives for each component. Including numerical measurements or examples to demonstrate the approach's performance would enhance the summary's clarity.
## Proposed Approach
![img](img/Proposed Approach_0.png)

This paper presents a novel approach, Verb-specific Semantic Roles (VSR), for controllable image captioning. VSR is a control signal that considers event-compatibility and sample-suitability requirements. The proposed approach consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. GSRL grounds semantic roles to visual regions, SSP learns the semantic structure of the sentence, and Role-shift Caption Generation generates the final captions. The training objectives for each component are defined, and a two-stage training scheme is used for the captioning model. In the testing stage, the framework can be extended to multiple VSRs. However, the paper lacks specific details on the models and algorithms used in each component. The authors claim better controllability and diversity, but quantitative results are not provided to support these claims.
### Controllable Caption Generation with VSR


The proposed approach in this paper aims to achieve controllable image captioning by introducing a novel control signal called Verb-specific Semantic Roles (VSR). The VSR consists of a verb and semantic roles that represent a targeted activity and the roles of entities involved in the activity. The approach consists of three main components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. 

In the GSRL component, an object detector is used to extract object proposals from the image, and a similarity score is calculated between each semantic role and proposal set. The SSP component includes a sentence-level SSP and a role-level SSP, which learn the sequence of semantic roles and rank sub-roles within each role, respectively. The Role-shift Caption Generation component generates the final caption by focusing on specific sub-roles and their grounded region sets.

During the training stage, the GSRL, SSP, and captioning model are trained separately with respective loss functions. In the inference stage, the framework can be extended to multiple VSRs as control signals. The proposed approach offers a novel solution for controllable image captioning, allowing users to generate captions tailored to specific activities and entities in the image.

The key technical details, such as the use of object proposals, similarity scores, adaptive attention mechanisms, and training with cross-entropy and reinforcement learning objectives, contribute to the innovative nature of this research. The proposed approach demonstrates improved controllability compared to existing baselines and enables the generation of diverse captions.
### Training and Inference
![img](img/Training and Inference_0.png)

The proposed approach in this paper introduces Verb-specific Semantic Roles (VSR) as a control signal for controllable image captioning. It consists of three components: Grounded Semantic Role Labeling (GSRL), Semantic Structure Planner (SSP), and Role-shift Caption Generation. In the GSRL step, similarity scores are calculated between semantic roles and object proposals. The SSP includes sentence-level and role-level models to learn the sequence of semantic roles and rank sub-roles within each role. The Role-shift Caption Generation uses adaptive attention to control the shift of sub-roles and generate captions. Each component is trained separately using appropriate objective functions. The framework can be extended to multiple VSRs as control signals.
## Experiments



### Datasets and Metrics


The research utilizes two benchmark datasets: Flickr30K Entities and COCO Entities. The Flickr30K Entities dataset is an extension of the Flickr30K dataset, where each noun phrase in the captions is manually grounded with visual regions. It contains 31,000 images, each associated with five captions. The COCO Entities dataset, on the other hand, consists of 120,000 images, and the annotations are automatically detected.

To ensure the scientificity, reliability, and validity of the technical methods, the research follows the same splits as previous studies for both datasets. However, it is noted that a small percentage of samples (3.26% in COCO Entities and 0.04% in Flickr30K Entities) do not contain any verbs in their captions. To address this, the samples without verbs are dropped during the training and testing stages. The study acknowledges this limitation and suggests that future work should explore covering these extreme cases.

It is worth mentioning that all baselines use the same visual regions as models with VSRs, ensuring fair comparisons between different methods.
### Implementation Details


The researchers utilized a Faster R-CNN model with ResNet-101 to generate proposals for each image. They used a pretrained SRL (Semantic Role Labeling) tool to annotate verbs and semantic roles for each caption, treating them as ground truth annotations. They built a verb dictionary for each dataset, with sizes of 2,662 for COCO and 2,926 for Flickr30K. 

In terms of experimental settings, the S-level SSP (Semantic Structure Planner) used a multi-head attention with 8 heads and a transformer with a hidden size of 512. The length of the transformer was set to 10. For the R-level SSP, the maximum number of entities for each role was set to 10. 

During the reinforcement learning (RL) training of the captioning model, the researchers used the CIDEr-D score as the training reward. Further details on parameter settings can be found in the supplementary material.

These implementation details are crucial for ensuring the scientificity, reliability, and validity of the technical methods used in the research. They provide a clear description of the experimental setups, data acquisition, and processing techniques employed by the researchers, allowing other researchers to replicate and validate their findings.
### Evaluation on Controllability
![img](img/Evaluation on Controllability_0.png)

![img](img/Evaluation on Controllability_1.png)

![img](img/Evaluation on Controllability_2.png)

The proposed approach in this paper introduces Verb-specific Semantic Roles (VSR) as a new control signal for controllable image captioning. VSR addresses the overlooked characteristics of event-compatibility and sample-suitability in existing control signals. It consists of a verb and semantic roles, ensuring event-compatibility, and only restricts the involved semantic roles, making it sample-suitable. The approach includes a grounded semantic role labeling model, a semantic structure planner, and a role-shift caption generation model. The summary emphasizes the need for a more effective captioning model, the extension of VSR to other tasks, and the development of a more general framework. Experimental results support the effectiveness of VSR.
## Conclusions & Future Works


This paper proposes a novel approach for controllable image captioning (CIC) by introducing a new control signal called Verb-specific Semantic Roles (VSR). The VSR consists of a verb and semantic roles that ensure event compatibility and sample suitability. The approach includes a grounded semantic role labeling (GSRL) model, a semantic structure planner (SSP), and a role-shift captioning model. Experimental results demonstrate the effectiveness of the VSR approach in achieving better controllability and generating diverse captions. Future work includes improving the captioning model, extending VSR to other tasks such as video captioning, and designing a more general framework for images without verbs.