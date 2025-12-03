# Attention and Transformers

---

## Mathematical Foundations

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Calculus & Linear Algebra</div>
        <div class="timeline-text">Basis for optimization algorithms and machine learning model operations</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1676:0,1805:0,1809:0,1847:0,1951:0">
        {{TIMELINE:timeline_calculus_linear_algebra}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Probability & Statistics</div>
        <div class="timeline-text">Basis for Bayesian methods, statistical inference, and generative models</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1815:0">
        {{TIMELINE:timeline_probability_statistics}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Information & Computation</div>
        <div class="timeline-text">Foundations of algorithmic thinking and information theory</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1843:0,1936:0,1947:0,1948:0">
        {{TIMELINE:timeline_information_computation}}
    </div>
</div>

<div class="fragment" data-fragment-index="1"></div>

---

## Early History of Neural Networks

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Architectures & Layers</div>
        <div class="timeline-text">Evolution of network architectures and layer innovations</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1943:0,1957:0,1965:0,1979:0,1982:0,1989:0,2012:0">
        {{TIMELINE:timeline_early_nn_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Methods for efficient learning and gradient-based optimization</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1967:0,1970:0,1986:0,1992:0,2009:0,2010:0,2012:0">
        {{TIMELINE:timeline_early_nn_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Datasets</div>
        <div class="timeline-text">Tools, platforms, and milestones that enabled practical deep learning</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="2002:0,2007:0,">
        {{TIMELINE:timeline_early_nn_software}}
    </div>
</div>

<div class="fragment" data-fragment-index="2"></div>

---

## The Deep Learning Era

<!-- Layers & Architectures Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Deep architectures</div>
        <div class="timeline-text">Deep architectures and generative models transforming AI capabilities</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2015:1,2016:0,2017:1">
        {{TIMELINE:timeline_deep_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Advanced learning techniques and representation learning breakthroughs</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2014:1,2015:0,2016:0">
        {{TIMELINE:timeline_deep_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2017:0,2018:1,2020:1,2022:1,2023:1">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>

---

## Recurrent Layers: Recap

<div style="font-size: 0.8em;">

**Vanilla RNN**:

<ul>
<li>Maintains hidden state across time steps to capture temporal dependencies</li>
<li>Suffers from vanishing/exploding gradients for long sequences</li>
<li>Formula: $\mathbf{h}_t = \sigma\left(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}\right)$</li>
</ul>

<div class="fragment" data-fragment-index="1">

**LSTM (Long Short-Term Memory)**:

- Uses gating mechanisms (forget, input, output gates) and separate cell state
- Better at capturing long-term dependencies, mitigates vanishing gradients
- More parameters and computational cost than vanilla RNN

</div>

<div class="fragment" data-fragment-index="2">

**GRU (Gated Recurrent Unit)**:

- Simplified variant with reset and update gates (no separate cell state)
- Fewer parameters and computational cost than LSTM while maintaining comparable performance

</div>

<div class="fragment image-overlay highlight" data-fragment-index="3" style="text-align: left; width: 70%; font-size: 1.25em;">

Limitations of Recurrent Layers:
<ul>
<li>Recurrent layer do only have direct connections to the previous time step, limiting long-range dependency capture</li>
<li>Sequential processing hinders parallelization, leading to long training times</li>
</ul>

</div>

---

## Attention Mechanism

- Addresses limitations of recurrent layers by allowing direct connections between all time steps
- Computes queries and keys to determine relevance between different positions in the sequence
- Enriches value representations by aggregating information from relevant time steps
- Enables modeling of long-range dependencies and parallel processing at once

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; top: 80%;">
    <img src="assets/images/06-attention_transformer/self_attention.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 90%; object-fit: contain;">
</div>

---

## Attention is All You Need (2017)

- Introduced the Transformer architecture, which relies solely on attention mechanisms, eliminating the need for recurrent or convolutional layers
- Utilizes multi-head self-attention to capture different aspects of relationships between tokens in a sequence
- Employs positional encoding to retain the order of tokens in the input sequence
- Achieved state-of-the-art results in machine translation tasks, significantly outperforming previous models
- Large language models like BERT, GPT, LLaMA, and others are based on the Transformer architecture

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/transformer_raw.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

---

## Embedding Layers

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/transformer_embedding.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

- Convert discrete tokens (words, subwords, characters, midi notes) into continuous vector representations
- Capture semantic relationships and contextual information
- Learned during training to optimize task performance
- Similar to a lookup table where each token maps to a dense vector
- Linear layer without bias and with one-hot encoded inputs

<div class="formula">
$$
\begin{aligned}
\mathtt{nn.Embedding}(i = \text{token index}) &= \mathbf{e}_{i}^{\top} \mathbf{W} \\
&= \begin{bmatrix}0 & \cdots & 1 & \cdots & 0\end{bmatrix} \mathbf{W} \\
&= \mathbf{W}_{i, :}
\end{aligned}
$$
</div>

where $\mathbf{W} \in \mathbb{R}^{V \times D}$ is the embedding matrix, $V$ is the vocabulary size, and $D$ is the embedding dimension.

</div>

---

## Output Projection

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/transformer_final_projection.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

- Maps the decoder's output back to the vocabulary space for token prediction
- Typically implemented as a linear layer followed by a softmax activation
- Shares weights with the embedding layer to reduce the number of parameters and improve performance (Press & Wolf, 2017)
- Converts the decoder's continuous representations into logits for each token in the vocabulary

<div class="formula">
$$
\begin{aligned}
\mathtt{nn.Linear}(\mathbf{h}_t) &= \mathbf{h}_t \mathbf{W}^{\top} + \mathbf{b} \\
&= \mathbf{h}_t \mathbf{W}^{\top} \quad \text{(if weights are shared, } \mathbf{b} = 0\text{)}\\
\mathbf{p}_t &= \mathrm{softmax}(\mathbf{h}_t \mathbf{W}^{\top}) = \frac{\exp(\mathbf{h}_t \mathbf{W}^{\top})}{\sum_{j=1}^{V} \exp((\mathbf{h}_t \mathbf{W}^{\top})_j)}
\end{aligned}
$$
</div>

where $\mathbf{W} \in \mathbb{R}^{V \times D}$ is the shared weight matrix from the embedding layer, $V$ is the vocabulary size, and $D$ is the model dimension.

---

## Positional Encoding

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 65%;">
    <img src="assets/images/06-attention_transformer/transformer_positional_encoding.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

- Since Transformers do not have inherent sequential processing, positional encodings are added to input embeddings to provide information about the order of tokens
- Can be implemented using fixed sinusoidal functions or learned embeddings
- Enables the model to capture the relative and absolute positions of tokens in the sequence
- **Vanilla sinusoidal positional encoding formula:**

<div class="formula">
$$
\begin{aligned}
\mathrm{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
\mathrm{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="3" style="text-align: center; top: 65%; width: 100%;">
    <img src="assets/images/06-attention_transformer/positional_encoding.png" alt="Transformer Architecture" style="max-width: 100%;">
</div>

---

## Self-Attention Mechanism

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/transformer_self_attention.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

Self-attention allows each token in the input sequence to attend to all other tokens, enabling the model to capture dependencies regardless of their distance in the sequence

</div>

<div class="fragment" data-fragment-index="4" style="font-size: 0.9em;">

**Step 1: Compute Queries, Keys, and Values**<br>

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="5" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_raw.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="6" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_qkv_compute.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="7" style="font-size: 0.9em;">

For each token, compute query ($\mathbf{Q}$), key ($\mathbf{K}$), and value ($\mathbf{V}$) vectors using learned linear projections.

<div class="formula">
$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V
$$
</div>

where $\mathbf{X} \in \mathbb{R}^{T \times D}$ is the input sequence matrix, and $\mathbf{W}_Q \in \mathbb{R}^{D \times D_k}$, $\mathbf{W}_K \in \mathbb{R}^{D \times D_k}$, $\mathbf{W}_V \in \mathbb{R}^{D \times D_v}$ are learned weight matrices. Each token has dimension $D$, and the sequence length is $T$.

</div>

</div>

</div>

---

## Scaled Dot-Product Attention

<div style="font-size: 0.9em;">

**Step 2: Compute Attention Scores**

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 40%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_scaled_dot_product.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="2" style="text-align: center; top: 40%;">
    <img src="assets/images/06-attention_transformer/scaled_dot_product_attention_raw.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="3" style="text-align: center; top: 40%;">
    <img src="assets/images/06-attention_transformer/scaled_dot_product_attention_attention_weights.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="4" style="font-size: 0.8em;">
Calculate attention scores using the scaled dot-product of queries and keys:
<div class="formula">
$$
\mathbf{A} = \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{D_k}}\right)
$$
</div>

where $d_k$ is the dimension of the key vectors, used for scaling to prevent large dot-product values that could lead to small gradients. $\mathbf{A} \in \mathbb{R}^{T \times T}$ contains the attention weights for each token pair in the sequence.
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="5" style="text-align: center; top: 40%;">
    <img src="assets/images/06-attention_transformer/self_attention.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
</div>

<div class="fragment" data-fragment-index="6" style="font-size: 0.9em;">

**Step 3: Compute Weighted Sum of Values**

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="7" style="text-align: center; top: 40%;">
    <img src="assets/images/06-attention_transformer/scaled_dot_product_attention_value_matmul.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="8" style="font-size: 0.8em;">
Finally, compute the output as a weighted sum of the value vectors:
<div class="formula">
$$
\mathbf{Z} = \mathbf{A} \mathbf{V}
$$
</div>

$\mathbf{Z} \in \mathbb{R}^{T \times D_v}$ captures information from all tokens in the sequence, weighted by their relevance to the query token

</div>
</div>

---

## Linear Output Layer

**Step 4: Final Linear Projection**<br>

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 80%;">
    <img src="assets/images/06-attention_transformer/output_projection.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

- After obtaining the output from the self-attention mechanism, a linear layer is applied to project the output back to the model dimension
- This linear transformation allows the model to learn complex combinations of the attended information
- The output of this layer is then passed through subsequent layers in the Transformer architecture

<div class="formula">
$$
\mathbf{h}_{\text{out}} = \mathbf{Z} \mathbf{W}_O + \mathbf{b}_O
$$
</div>

---

## Causal Self-Attention

<div class="fragment appear-vanish image-overlay" data-fragment-index="0" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/transformer_masked_self_attention.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="1" style="font-size: 0.9em;">

- In autoregressive models, causal self-attention ensures that each token can only attend to previous tokens in the sequence, preventing information leakage from future tokens
- This is typically implemented by applying a mask to the attention scores before the softmax operation

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="2" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/scaled_dot_product_attention_masked.png" alt="Causal Attention Mask" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="3" style="font-size: 0.9em;">
The masked attention scores are computed as follows:
<div class="formula">
$$
\begin{aligned}
\mathbf{M}_{i,j} &= \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases} \\
\mathbf{A}_{\text{masked}} &= \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{D_k}} + \mathbf{M}\right)
\end{aligned}
$$
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/self_attention_causal.png" alt="Causal Self-Attention" style="max-width: 90%; max-height: 80%; object-fit: contain;">
</div>

</div>

---

## Multi-Head Attention

<div class="fragment appear-vanish image-overlay" data-fragment-index="1" style="text-align: center; top: 60%;">
    <img src="assets/images/06-attention_transformer/transformer_multihead_attention.png" alt="Transformer Architecture" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="2" style="font-size: 0.9em;">

- Instead of performing a single attention function, multi-head attention runs multiple attention operations in parallel
- Each "head" learns different attention patterns, allowing the model to capture various aspects of relationships
- Enables the model to jointly attend to information from different representation subspaces

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="3" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_raw.png" alt="Multi-Head Attention" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_qkv_compute.png" alt="Multi-Head Attention QKV" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="5" style="font-size: 0.8em;">

**Step 1: Project inputs to multiple heads**

For each head $i$, compute separate Q, K, V projections:

<div class="formula">
$$
\mathbf{Q}_i = \mathbf{X} \mathbf{W}_Q^i, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_K^i, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_V^i
$$
</div>

where $\mathbf{W}_Q^i \in \mathbb{R}^{D \times D_k}$, $\mathbf{W}_K^i \in \mathbb{R}^{D \times D_k}$, $\mathbf{W}_V^i \in \mathbb{R}^{D \times D_v}$ are unique weight matrices for head $i$

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="6" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_scaled_dot_product.png" alt="Multi-Head Attention Computation" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="7" style="font-size: 0.8em;">

**Step 2: Compute attention for each head**

<div class="formula">
$$
\begin{aligned}
\mathbf{A}_i &= \mathrm{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^{\top}}{\sqrt{D_k}}\right) \\
\mathbf{Z}_i &= \mathbf{A}_i \mathbf{V}_i
\end{aligned}
$$
</div>

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="8" style="text-align: center; top: 50%;">
    <img src="assets/images/06-attention_transformer/multihead_attention_concatenate.png" alt="Multi-Head Attention Concatenation" style="max-width: 90%; max-height: 80%; object-fit: contain;">
    <div class="reference" style="margin: 10px; text-align: center;">
    Source: <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need</a>
    </div>
</div>

<div class="fragment" data-fragment-index="9" style="font-size: 0.8em;">

**Step 3: Concatenate heads and project**

<div class="formula">
$$
\begin{aligned}
\mathbf{Z}_{\text{concat}} &= \mathrm{Concat}(\mathbf{Z}_1, \mathbf{Z}_2, \ldots, \mathbf{Z}_h) \\
\mathbf{h}_{\text{out}} &= \mathbf{Z}_{\text{concat}} \mathbf{W}_O + \mathbf{b}_O
\end{aligned}
$$
</div>

where $h$ is the number of heads, and $\mathbf{W}_O \in \mathbb{R}^{h \cdot D_v \times D}$ projects the concatenated outputs back to the model dimension

</div>

---

# Python Implementation
