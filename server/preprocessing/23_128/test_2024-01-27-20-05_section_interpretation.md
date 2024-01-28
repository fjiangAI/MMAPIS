# Addressing Token Uniformity in Transformers via Singular Value Transformation

- Authors: Hanqi Yan<sup>1</sup>    Lin Gui<sup>1</sup>    Wenjie Li<sup>2</sup>    Yulan He<sup>1,3,</sup>

- Affiliations: <sup>1</sup>Department of Computer Science, University of Warwick, United Kingdom<br><br>   <sup>2</sup>Department of Computing, The Hong Kong Polytechnic University, China<br><br>   <sup>3</sup>The Alan Turing Institute, United Kingdom

![img](img/SINGULAR VALUE DISTRIBUTION OF_0.png)

![img](img/TRANSFORMATION FUNCTION_0.png)

![img](img/EXPERIMENTS_0.png)

![img](img/EXPERIMENTS_1.png)

![img](img/EXPERIMENTS_2.png)

![img](img/EXPERIMENTS_3.png)

![img](img/EXPERIMENTS_4.png)

![img](img/EXPERIMENTS_5.png)

## Abstract


The "Singular Value Distribution of Transformer Block Outputs" section explores the token uniformity problem in transformer-based models. The authors propose a geometric interpretation, highlighting the vanishing singular values in transformers and bounding the embedding space based on these values. Empirical studies on BERT confirm the presence of skewed singular value distributions in intermediate transformer blocks, indicating a higher degree of token uniformity with increasing network depth. The authors also observe an increase in token uniformity as the network layer goes deeper, as measured by the average cosine similarity between tokens. This section provides valuable insights into the token uniformity issue and its implications for transformer models.
## Introduction


The "Singular Value Distribution of Transformer Block Outputs" section explores the token uniformity problem in transformer-based language models. It presents the information propagation process in a transformer block and discusses the limitations of previous approaches. The section proposes studying the singular value distribution to understand token uniformity. A geometric interpretation reveals that the embedding space is bounded by singular values. An empirical study on BERT demonstrates that the singular value distribution becomes steeper with deeper layers, indicating increased token uniformity. Additional specific details about the empirical study could enhance the summary.
## Related Work


This section explores the singular value distribution of transformer block outputs and its relation to token uniformity. The authors provide a geometric interpretation of vanishing singular values in transformers and propose a transformation function to address token uniformity while preserving the local neighborhood structure. An empirical study on BERT demonstrates that the degree of skewness of singular value distributions correlates with token uniformity. The findings highlight the importance of considering singular value distributions in addressing token uniformity issues in transformer-based models.
## Singular Value Distribution of Transformer Block Outputs


The "Singular Value Distribution of Transformer Block Outputs" section explores the problem of token uniformity in transformer-based models. It introduces the concept of singular value distribution and its significance in understanding the distortion or stretching of input signals. The section provides a geometric interpretation of the vanishing singular values in transformers and establishes a bound on the embedding space. The authors propose a novel approach to studying token uniformity by analyzing the skewness of the singular value distribution. They conduct an empirical study on BERT, showing that the singular value distribution becomes steeper with increasing network depth, indicating token uniformity. The section contributes to the academic field by providing insights into the token uniformity problem and establishing the importance of the singular value distribution in transformer models.
### Singular Value Vanishing in Transformer


The "Singular Value Distribution of Transformer Block Outputs" section explores the token uniformity problem in transformer-based models. The authors provide a geometric interpretation of the problem based on the singular value distribution and establish that the embedding space is bounded by the largest singular value and an upper bound for small singular values, forming a cone-like hypersphere. They conduct an empirical study on BERT, observing skewed singular value distributions in intermediate transformer blocks, indicating a higher degree of token uniformity as the network depth increases. The authors highlight the significance of the singular value distribution in understanding and addressing the token uniformity problem. Key mathematical formulas related to singular value decomposition and the definition of the subspace \(\mathcal{S}^{l}_{[1,k]}\) are omitted in the summary. The empirical study quantifies the degree of token uniformity using measurements such as average cosine similarity and median singular values. The summary maintains language fluency, consistency, and authenticity with the source content.
### Empirical Study of Token Uniformity in Bert


The "Singular Value Distribution of Transformer Block Outputs" section investigates the token uniformity problem in transformer-based language models. The authors propose a novel approach to studying token uniformity by analyzing the distribution of singular values in transformer block outputs. They provide a geometric interpretation, demonstrating that the embedding space is bounded by the largest singular value and an upper bound for small singular values. An empirical study using BERT confirms that the singular value distribution becomes steeper with increasing network depth, indicating higher token uniformity. The authors also observe the vanishing of smaller singular values and measure token uniformity using average cosine similarity between tokens. This section contributes to understanding the relationship between singular value distributions and token uniformity in transformer models.
## Transformation Function


In this section, the paper proposes a singular value transformation function to alleviate the token uniformity problem in transformer-based models. The authors provide insights into the design of this function based on their empirical analysis of the changes in singular value distributions and measures of token uniformity across transformer blocks. The transformation function aims to modify the singular value distribution to make it less skewed towards small values while preserving the local neighborhood structure in the original embedding space. This innovative approach addresses the limitations of existing methods by effectively reducing token uniformity without sacrificing the preservation of important information.
### Motivation


The 4 Transformation Function section of the paper focuses on addressing the token uniformity problem by proposing a transformation function to adjust the skewness of singular value distributions. The motivation behind this is that existing normalization methods preserve the trace of the covariance matrix but do not control higher moments of the distribution, such as the skewness, resulting in a potential imbalance in singular values.

The proposed transformation function aims to modify small singular values to prevent dimension vanishing and alleviate the token uniformity problem. The function is designed to adjust the skewness of singular value distributions, making them less skewed towards small singular values. By doing so, it helps to balance the contributions of different features and mitigate the anisotropic shape of the embedded feature space.

The paper does not provide specific details about the mathematical formulation or implementation of the transformation function in this section. However, it highlights the significance of controlling higher moments of the singular value distribution, such as skewness, in order to achieve a more balanced representation.

Overall, this section emphasizes the novel contribution of proposing a transformation function that addresses the token uniformity problem by adjusting the skewness of singular value distributions. This innovation builds upon existing normalization methods and aims to provide a more comprehensive solution for achieving balanced representations in transformer-based models.
### Properties of desirable singular value transformation


The "Properties of desirable singular value transformation" section presents three crucial properties for an effective singular value transformation function to mitigate token uniformity in PTLMs. Firstly, the function should be monotonically increasing to preserve the order of singular values. Secondly, the second-order derivative of the function should be monotonically decreasing to balance the transformed distribution. Lastly, the largest singular value should remain unchanged to maintain the bounded embedding space. These properties ensure that the transformation effectively addresses token uniformity while preserving important characteristics of the original data manifold.
### Softdecay Function


The paper proposes a non-linear and trainable transformation function called SoftDecay, which is built on the soft-exponential function. The function is defined by Equation (2) and has desirable properties for addressing the token uniformity problem. The SoftDecay function ensures that the transformed singular values are always greater than or equal to the original singular values, preserving the local neighborhood structure.

To apply the transformation, the original representations are first decomposed using Singular Value Decomposition (SVD). Then, the SoftDecay function is applied to the singular values, resulting in transformed singular values. A rescaling factor is computed to ensure that the maximum transformed singular value matches the maximum original singular value. Finally, the transformed representations are obtained by multiplying the transformed singular values with the original decomposition matrices.

The proposed transformation is summarized in Algorithm 1, which outlines the steps for applying SoftDecay to the representations. This transformation function allows for alleviating token uniformity while preserving the local neighborhood structure in the original embedding space.
### Transformed Feature Evaluation


The "Transformed Feature Evaluation" section introduces the evaluation metrics for assessing the quality of transformed features. The paper argues that the evaluation should consider both the uniformity and the preservation of the local neighborhood structure in the original embedding space.

To measure distribution uniformity, three different metrics are proposed. First, the "TokenUni" metric calculates the cosine similarity between transformed features. Second, the "RBF_dis" metric uses the Radial Basis Function (RBF) kernel to measure feature similarity. Finally, the "EV_k" metric computes the explained variance by comparing variances in different directions or singular values.

To evaluate the preservation of the local neighborhood structure, the paper introduces the Local Structure Discrepancy Score (LSDS). This metric measures how well the transformed features preserve the linear combination of nearest neighbors in the original space.

These evaluation metrics provide a comprehensive assessment of both the uniformity and preservation of local structure in the transformed features, ensuring that the proposed transformation function achieves the desired properties.
## Experiments


The experiments in this paper focus on evaluating the proposed transformation functions on four widely-used Pre-Trained Language Models (PTLMs): BERT, ALBERT, RoBERTa, and DistilBERT. The evaluation is conducted on both semantic textual similarity (STS) datasets and General Language Understanding Evaluation (GLUE) tasks.

To ensure the scientificity and reliability of the experimental results, the authors provide model training details and additional results in the supplementary material. This allows for transparency and reproducibility of the experiments.

The specific experimental setup and data acquisition and processing techniques are not mentioned in this section. However, it can be assumed that the authors follow standard procedures for training the PTLMs on large-scale datasets and fine-tuning them on specific downstream tasks.

The use of STS datasets allows for the evaluation of the proposed transformation functions in capturing semantic similarity between sentences. This helps assess the effectiveness of the functions in addressing the token uniformity problem while preserving the local neighborhood structure.

The GLUE tasks provide a comprehensive evaluation of the PTLMs' performance on various natural language understanding tasks, including sentiment analysis, textual entailment, and question answering. By applying the proposed transformation functions to these PTLMs and evaluating their performance on GLUE tasks, the authors demonstrate the effectiveness of their method across a range of NLP tasks.

Overall, the experimental techniques and procedures used in this research involve training and fine-tuning PTLMs on large-scale datasets, evaluating their performance on STS datasets and GLUE tasks, and applying the proposed transformation functions to observe improvements in performance. These techniques ensure the scientificity, reliability, and validity of the technical methods used in this study.
### Unsupervised Evaluation on STS


The researchers conducted unsupervised evaluations on the Semantic Textual Similarity (STS) task to assess the performance of their proposed method, SoftDecay, in adjusting anisotropy. They compared SoftDecay with other unsupervised methods, including BERT-flow, SBERT-WK, BERT-whitening, and WhiteBERT. BERT-flow aimed to transform representations learned by PTLMs into a standard Gaussian distribution, while SBERT-WK trained the top transformation layer using Natural Language Inference datasets. BERT-whitening and WhiteBERT analyzed BERT-based word models geometrically to achieve isotropic representation distribution.

The results, as shown in Table 1, demonstrated that SoftDecay outperformed the baselines significantly across all seven datasets. It achieved a 23.5% improvement over the base PTLMs and a 5% improvement over the best baseline among BERT-based methods. The improvements were more pronounced in ALBERT-based models (23%) compared to DistilBERT-based models (8%), which aligns with the expectation that deeper models are more susceptible to token uniformity issues. The cross-layer parameter sharing in ALBERT potentially exacerbates token uniformity and benefits more from mitigation strategies.

To further investigate how SoftDecay alleviates token uniformity, the researchers visualized the cumulative density function (CDF) of singular values from DistilBERT and ALBERT before and after applying SoftDecay. The CDF plots showed that the singular value distribution of the last layer output of ALBERT became less skewed after applying SoftDecay, indicating a reduction in token uniformity compared to DistilBERT.

Additionally, the researchers evaluated the features for the STS task by visualizing the sentence representations in STS-15 using tSNE and assessing them with proposed metrics. BERT-whitening achieved perfect isotropy but failed to preserve the local neighborhood structure of BERT embeddings. In contrast, SoftDecay significantly improved uniformity while maintaining a similar distribution shape, effectively preserving the local neighborhood structure. This preservation of local structure contributed to the superior performance of SoftDecay compared to other methods.

Overall, the experimental techniques and procedures involved conducting evaluations on STS datasets, comparing results with baselines, visualizing singular value distributions, and assessing feature characteristics. These techniques ensured the scientificity, reliability, and validity of the proposed method and demonstrated its effectiveness in addressing token uniformity issues.
### Supervised Evaluation on Glue Datasets


The researchers conducted supervised evaluations on the GLUE datasets to assess the effectiveness of their proposed method. They focused on five sentence-level classification datasets: CoLA for grammar acceptability assessment, SST2 for sentiment classification, MRPC for paraphrase detection, QNLI for natural language inference, and RTE for recognizing textual entailment.

The researchers applied their SoftDecay method on the last encoder layer of BERT, ALBERT, and DistilBERT. They fine-tuned the weights of the pretrained transformer language models (PTLMs) along with the hyperparameter alpha (\(\alpha\)) specific to each task. They compared their method with two baselines: Sentence-BERT (S-BERT) and BERT-CT. S-BERT added a pooling operation to derive a sentence embedding, while BERT-CT incorporated contrastive loss in the training objective to retain distinguishable sentence representations.

Since the GLUE test set was not publicly available, the researchers submitted their trained models to the GLUE leaderboard to obtain the test results. The results showed that SoftDecay was more effective on BERT-based models, while giving less noticeable improvement on DistilBERT, which aligns with the observations from the STS tasks. BERT performed better than the other models on single-sentence tasks, except for MRPC. All models achieved better results on inference tasks, particularly on the smaller RTE dataset. The cumulative distribution function (CDF) of singular value distributions before and after applying SoftDecay further confirmed the effectiveness of their proposed transformation function. The researchers also noted that models trained on larger training sets tended to generate more similar representations. On MRPC, SoftDecay was effective on BERT but resulted in a slight performance drop on ALBERT and DistilBERT, possibly due to the smaller training set size. Overall, SoftDecay outperformed both S-BERT and BERT-CT across all tasks according to the GLUE test results.
## Conclusion and Future Work


The paper addresses the issue of token uniformity in transformer-based models by proposing a transformation function based on the distribution of singular values. The authors empirically show that the degree of skewness of singular value distributions correlates with token uniformity. Their proposed approach effectively alleviates token uniformity while preserving the local neighborhood structure. Experimental results demonstrate improved performance on semantic textual similarity evaluation and GLUE tasks. Future research directions include extending the approach to encoder-decoder structures and exploring its impact on language generation tasks.