# Attention Is All You Need

- Authors: Ashish Vaswani avaswani@google.com &Noam Shazeer1 noam@google.com &Niki Parmar1 nikip@google.com &Jakob Uszkoreit1 &Llion Jones1 llion@google.com &Aidan N. Gomez1 &Lukasz Kaiser1 lukaszkaiser@google.com illia.polosukhin@gmail.com

- Affiliations: Google Brain Google Brain Google Research Google Research usz@google.com Google Research University of Toronto aidan@cs.toronto.edu Google Brain &Illia Polosukhin1 Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.Work performed while at Google Brain.Work performed while at Google Research.Work performed while at Google Research. Footnote 1: footnotemark: Footnote 2: footnotemark:



## Overview

In today's post, we shall unpack the nuances of a fascinating research piece titled **"Attention Is All You Need"** penned by Ashish Vaswani et al. This research introduces the Transformer, an innovative network architecture for sequence transduction tasks that replaces recurrent and convolutional neural networks with an attention mechanism-based approach.

The primary motivation behind this research is to overcome the limitations of traditional models in learning dependencies between distant positions and achieving higher effective resolution. The Transformer model incorporates self-attention as the basic building block, revolutionizing the field of sequence transduction. By leveraging attention mechanisms, the Transformer effectively models dependencies between input and output sequences without considering their distance.

The contributions of this paper are both significant and groundbreaking. The researchers showcase the superiority of the Transformer model over existing models in terms of translation quality, parallelizability, and reduced training time. Experimental results demonstrate that the Transformer outperforms previous models on machine translation tasks, achieving a BLEU score of 28.4 on the WMT 2014 English-to-German translation task and a BLEU score of 41.8 on the WMT 2014 English-to-French translation task.

Now, let's delve into the core content of this paper and explore its innovative techniques and remarkable insights.

## Model Architecture
<div style="width: 100%; overflow-x: auto; white-space: nowrap; padding: 20px 0;">
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Model Architecture_0.png" alt="Image 0" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img0</div>
    </div>
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Model Architecture_1.png" alt="Image 1" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img1</div>
    </div>
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Model Architecture_2.png" alt="Image 2" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img2</div>
    </div>
</div>


The Transformer model introduces a revolutionary approach to sequence transduction tasks through its encoder-decoder structure. The encoder maps input symbol representations to continuous representations, while the decoder generates an output sequence based on these continuous representations. Unlike traditional models that rely on recurrent or convolutional layers, the Transformer utilizes self-attention to compute representations without the need for sequence-aligned neural networks. This architectural shift enables the model to capture dependencies between distant positions more effectively.
### Encoder and Decoder Stacks

The heart of the Transformer lies in its encoder-decoder model, which consists of multiple layers. The encoder comprises six layers, each incorporating a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. To facilitate information flow and maintain output dimensions, residual connections and layer normalization are employed. Similarly, the decoder also consists of six layers, with an additional sub-layer performing multi-head attention over the encoder stack's output. Notably, the decoder's self-attention sub-layer is modified to prevent positions from attending to subsequent positions, ensuring predictions depend solely on known outputs.
### Attention

A pivotal aspect of the Transformer is its "Scaled Dot-Product Attention" mechanism, mathematically defined as:

$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V$

This formulation allows for efficient processing through matrix operations. The authors also introduce multi-head attention, enabling the model to attend to different representation subspaces:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}*{1},...,\text{head}*{h}) W^{O}$$

where each head is defined as:

$$\text{head}*{i} = \text{Attention}(QW*{i}^{Q},KW_{i}^{K},VW_{i}^{V})$$

The Transformer leverages attention in three key areas: encoder-decoder attention, self-attention in the encoder, and self-attention in the decoder. These attention mechanisms play a crucial role in capturing contextual information and improving the model's performance.
### Positionwise FeedForward Networks

Each layer incorporates a position-wise feed-forward network expressed as:

$\text{FFN}(x) = \max(0,xW_{1}+b_{1})W_{2}+b_{2}$

The input and output dimensions of these networks are both set to 512, with an inner-layer dimensionality of 2048. This design allows distinct parameter utilization across layers while ensuring consistent transformations across positions.

### Embeddings and Softmax

To facilitate sequence transduction, the paper adopts embeddings and softmax as part of the Transformer model. Input and output tokens are converted into vectors of dimension $d_{\text{model}}$ using learned embeddings. The decoder's output is transformed into predicted next-token probabilities through a linear transformation and softmax function. By sharing the weight matrix between the embedding layers and pre-softmax linear transformation, the model achieves enhanced performance. Additionally, multiplying the weights in the embedding layers by $\sqrt{d_{\text{model}}}$ further improves the model's capabilities.
### Positional Encoding

To incorporate sequence order in a model that lacks recurrence and convolution, the Transformer introduces positional encoding:

$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{\text{model}}}), \quad PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{\text{model}}})$

The authors utilize sine and cosine functions to generate positional encodings, with each dimension corresponding to a sinusoid. These sinusoidal positional encodings form a geometric progression, allowing the model to attend to relative positions within a sequence. Although learned positional embeddings were experimented with, the paper demonstrates that the sinusoidal version yields similar results. This approach enables the Transformer to handle sequences of varying lengths effectively.

## Why SelfAttention

The authors provide a critical examination of self-attention layers compared to recurrent and convolutional layers for mapping variable-length sequences. They identify three key advantages supporting the use of self-attention:

1. **Computational Complexity**: Self-attention connects all positions with a constant number of operations, while recurrent layers require O(n) operations.
2. **Parallelization**: The architecture allows for significantly more parallelizable computations compared to recurrent layers, optimizing training times.
3. **Path Length and Long-Range Dependencies**: Self-attention facilitates learning long-range dependencies through shorter paths between input-output positions.

Additionally, self-attention models offer enhanced interpretability, as demonstrated by attention distributions that reveal distinct tasks associated with individual attention heads.

## Training
<div style="width: 100%; overflow-x: auto; white-space: nowrap; padding: 20px 0;">
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Training_0.png" alt="Image 0" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img0</div>
    </div>
</div>


The paper provides insights into the training regime employed for the Transformer models used in their study. The machine translation models were trained on the WMT 2014 English-German and English-French datasets. The English-German dataset consisted of 4.5 million sentence pairs, while the English-French dataset contained 36 million sentences. Byte-pair encoding was used to encode the sentences, resulting in shared and word-piece vocabularies for English-German and English-French, respectively. The training data was batched based on approximate sequence length, with each training batch containing approximately 25,000 source tokens and 25,000 target tokens. The models were trained on a single machine with 8 NVIDIA P100 GPUs for either 12 hours or 3.5 days, depending on the model size. The Adam optimizer was employed with specific hyperparameters, including $\beta_{1}=0.9$, $\beta_{2}=0.98$, and $\epsilon=10^{-9}$. The learning rate was adjusted during training using a formula that accounted for the model size and step number. Regularization techniques such as Residual Dropout, Dropout on embeddings and positional encodings, and Label Smoothing were also employed.
## Results
<div style="width: 100%; overflow-x: auto; white-space: nowrap; padding: 20px 0;">
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Results_0.png" alt="Image 0" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img0</div>
    </div>
    <div style="display: inline-block; margin-right: 10px; text-align: center;">
      <img src=".\img\Results_1.png" alt="Image 1" style="max-height: 400px; width: auto; vertical-align: top;">
      <div style="margin-top: 10px; font-weight: bold;">img1</div>
    </div>
</div>


The Transformer model's performance was evaluated on machine translation tasks, specifically English-to-German and English-to-French translation. The big transformer model achieved a new state-of-the-art BLEU score of 28.4 on the English-to-German task, surpassing previously reported models by more than 2.0 BLEU. Even the base model outperformed all previously published models and ensembles at a fraction of the training cost. On the English-to-French task, the big model achieved a BLEU score of 41.0, outperforming previous single models at a significantly lower training cost. The paper also explored various model variations and demonstrated their impact on performance. For instance, increasing the maximum output length and using a beam size of 21 improved translation quality. The Transformer model's superiority over other architectures was evident in the results, positioning it as a significant contribution to sequence transduction tasks.
## Critical Review and Future Implications

The paper establishes the Transformer as a pioneering model in sequence transduction, showcasing significant contributions to efficiency and performance in NLP tasks.

### Strengths

- **Innovative Architecture**: By leveraging self-attention exclusively, the Transformer achieves remarkable efficiency and performance improvements.
- **Empirical Validation**: Rigorous experimentation demonstrates its superiority over traditional models across various tasks.

###  Potential Drawbacks

- **Scalability Challenges**: Further investigation is needed to optimize processing efficiency for very long sequences.
- **Broader Applicability**: More empirical studies are required to validate performance across diverse domains and tasks.

### Future Research Directions

The authors express enthusiasm about extending attention-based models beyond text to diverse modalities like images and audio. This opens avenues for innovative applications in various fields.

In conclusion, this paper not only establishes the Transformer as a groundbreaking model in sequence transduction but also sets the stage for future research directions aimed at expanding its applicability and efficiency across various domains. The significance of this work resonates throughout the research arena, promising transformative impacts on how we approach natural language processing and beyond.