# Recurrent Layers

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
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1943:0,1957:0,1965:0,1979:0,2012:0,1982:1,1989:1">
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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2016:0">
        {{TIMELINE:timeline_deep_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Advanced learning techniques and representation learning breakthroughs</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2015:0,2016:0">
        {{TIMELINE:timeline_deep_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2017:0">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>

---

## Recap: Linear Layers

<div class="fragment appear-vanish" data-fragment-index="0">

- Linear layers are the building blocks of neural networks, consisting of weights and biases
- A linear layer with one output and a step activation function is named a perceptron
- In order to stack multiple linear layers and learn complex patterns, we need a differentiable non-linear activation function
- Common activation functions include ReLU, sigmoid, and tanh

</div>

<div class="fragment" data-fragment-index="0">

- The forward pass of a multi-layer perceptron (MLP) consists of multiple linear transformations followed by non-linear activations
- In the backward pass, we "go back" through the network to update the weights using gradient descent.

</div>

<div class="fragment" data-fragment-index="1" style="font-size: 0.70em;">

<div style="display: flex; gap: 40px;">

<div style="flex: 1;">

**Forward Pass**:
1. Input: $\mathbf{h}^{(0)} = \mathbf{x}$
2. For $l = 1, \ldots, L$:
   - $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$
   - $\mathbf{h}^{(l)} = \sigma(\mathbf{z}^{(l)})$
3. Output: $\hat{\mathbf{y}} = \mathbf{h}^{(L)}$
4. Loss: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$

</div>

<div style="flex: 1;">

**Backward Pass**:

1. Output layer: $\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \odot \sigma'(\mathbf{z}^{(L)})$
2. For $l = L-1, \ldots, 1$:
   - $\boldsymbol{\delta}^{(l)} = [(\mathbf{W}^{(l+1)})^\top \boldsymbol{\delta}^{(l+1)}] \odot \sigma'(\mathbf{z}^{(l)})$
3. Gradients for all layers:
   - $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^\top$
   - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$

</div>

</div>

</div>

<div class="image-overlay fragment highlight" style="width: 78%;">

Linear layers have limitations in modeling sequential data!

</div>

---

## Recap: Convolutional Layers

<div style="font-size: 0.85em;">

<div class="fragment" data-fragment-index="0">

- Convolutional layers apply learnable FIR filters, enabling translation invariance and local pattern recognition

</div>


<div class="fragment appear-vanish" data-fragment-index="0">

- They handle variable-length inputs with fewer parameters than fully connected layers
- Multi-channel convolutions learn diverse features through parallel kernels applied across all input channels
- Options include padding, stride, and dilation to control receptive field and output dimensions
- Pooling layers (max/average) reduce spatial dimensions, helping decrease computational load

</div>

<div class="fragment" data-fragment-index="0">

- Transposed convolutions upsample feature maps for generative models and segmentation tasks
- In backpropagation, weight gradients are computed via convolution, and error propagation uses full convolution with flipped kernels

</div>

</div>

<div class="fragment" data-fragment-index="1" style="font-size: 0.70em; display: flex; gap: 40px;">

<div style="flex: 1;">

**Forward Pass**:
1. Input: $\mathbf{h}^{(0)} = \mathbf{x}$
2. For $l = 1, \ldots, L$:
   - $z^{(l)}[n] = \sum_{k=0}^{M-1} w^{(l)}[k] h^{(l-1)}[n-k] + b^{(l)}$
   - $h^{(l)}[n] = \sigma(z^{(l)}[n])$
3. Output: $\hat{\mathbf{y}} = \mathbf{h}^{(L)}$
4. Loss: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$

</div>

<div style="flex: 1;">

**Backward Pass**:

1. Output layer: $\delta^{(L)}[n] = \frac{\partial \mathcal{L}}{\partial \hat{y}[n]} \cdot \sigma'(z^{(L)}[n])$
2. For $l = L-1, \ldots, 1$:
   - $\delta^{(l)}[n] = \left[\sum_{k=0}^{M-1} \delta^{(l+1)}[n+k] w^{(l+1)}[k]\right] \sigma'(z^{(l)}[n])$
3. Gradients for all layers:
   - $\frac{\partial \mathcal{L}}{\partial w^{(l)}[k]} = \sum_{n} \delta^{(l)}[n] h^{(l-1)}[n-k]$
   - $\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{n} \delta^{(l)}[n]$

</div>

</div>

<div class="image-overlay fragment highlight" style="width: 78%; text-align: left;">

Can convolutional layers capture long-range temporal dependencies?

- Limited: Standard convolutions have fixed, small receptive fields (kernel size $M$)
- Stacking layers increases receptive field, but grows linearly with depth
- Dilated convolutions expand receptive field exponentially, but still bound to fixed context windows

</div>

---

## WaveNet: Dilated Causal Convolutions

<div style="text-align: center; margin-top: 120px;">
    <img width="60%" src="assets/images/05-recurrent_layers/wavenet.gif">
    <div class="reference" style="margin-top: 10px; text-align: center;">
        Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio (No. arXiv:1609.03499). https://doi.org/10.48550/arXiv.1609.03499
    </div>
</div>

---

## Vanilla Recurrent Layers

- Recurrent layers are designed to handle sequential data by maintaining a hidden state that captures information from previous time steps
- They process input sequences one element at a time, updating the hidden state at each step
- The hidden state allows the network to retain memory of past inputs, enabling it to learn temporal dependencies

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/vanilla_rnn.png" alt="Different Space Regions" style="width: 70%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## Vanilla Recurrent Layers

- Recurrent layers are designed to handle sequential data by maintaining a hidden state that captures information from previous time steps
- They process input sequences one element at a time, updating the hidden state at each step
- The hidden state allows the network to retain memory of past inputs, enabling it to learn temporal dependencies

<div class="formula fragment appear-vanish" data-fragment-index="0">
$$
\begin{aligned}
\mathbf{h}_t &= f_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{h}_{t-1}) \\
\mathbf{h}_t &= \sigma(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_{x} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)
\end{aligned}
$$
</div>

<div class="formula fragment appear-vanish" data-fragment-index="1">
$$
\begin{aligned}
\mathbf{h}_t &= f_{\boldsymbol{\theta}}(\mathbf{x}_t, \mathbf{h}_{t-1}) \\
\mathbf{h}_t &= \sigma(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b})
\end{aligned}
$$
</div>

---

## Vanilla Recurrent Layer - Forward Propagation

<div style="font-size: 0.90em;">

**Forward Pass** (compute hidden states for $t = 1, \ldots, T$):

<div class="fragment" data-fragment-index="1">

<div class="formula" style="margin-top: 20px;">
$$
\begin{aligned}
\mathbf{z}_t &= \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b} \\
\mathbf{h}_t &= \sigma(\mathbf{z}_t)
\end{aligned}
$$
</div>

where:
- $\mathbf{W}_{xh} \in \mathbb{R}^{M \times N}$ maps input to hidden state
- $\mathbf{W}_{hh} \in \mathbb{R}^{M \times M}$ maps previous hidden to current hidden
- $\mathbf{b} \in \mathbb{R}^{M}$ is the bias vector
- $\sigma(\cdot)$ is the activation function (e.g., tanh, sigmoid)
- $\mathbf{h}_0$ is initialized (often to zeros)

</div>

</div>

---

## Backpropagation Through Time (BPTT)

<div class="highlight" style="margin-top: 150px; text-align: center;">
How do we train a recurrent neural network across multiple time steps?
</div>

<div class="fragment">
Compute gradients of the loss $\mathcal{L}$ with respect to parameters, accounting for dependencies across all time steps:

<div class="formula" style="margin-top: 40px;">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}}, \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}}
$$
</div>

</div>

---

## BPTT: Temporal Dependency Chain

The loss $\mathcal{L}$ has a **temporal dependency chain**:

<div style="margin-top: 40px; text-align: center;">
<br>
$\mathcal{L}$ depends on $\mathbf{h}_T$ (final hidden state)
<br>

<div class="fragment" data-fragment-index="1">
<br>
↓ which depends on $\mathbf{W}_{hh}$, $\mathbf{W}_{xh}$, $\mathbf{b}$, and $\mathbf{h}_{T-1}$
<br>
</div>

<div class="fragment" data-fragment-index="2">
<br>
↓ which depends on $\mathbf{W}_{hh}$, $\mathbf{W}_{xh}$, $\mathbf{b}$, and $\mathbf{h}_{T-2}$
<br>
</div>

<div class="fragment" data-fragment-index="3">
<br>
↓ and so on back to $\mathbf{h}_0$...
<br>
</div>

</div>

<div class="fragment highlight image-overlay" data-fragment-index="4" style="margin-top: 60px; text-align: left; padding: 50px; width: 78%;">
BPTT: Temporal Dependency Chain extends backpropagation to handle temporal dependencies by unrolling the recurrent network through time!
</div>

---

## BPTT: Temporal Dependency Chain

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/bptt_vanilla_rnn.png" alt="Different Space Regions" style="width: 50%;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## BPTT: Output Layer

<div style="font-size: 0.80em;">

<div><strong>MSE Loss</strong>: $\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\Vert\mathbf{y}_i - \hat{\mathbf{y}}_i\Vert^2 = \frac{1}{N}\sum_{i=1}^{N}\sum_{j}(y_{ij} - \hat{y}_{ij})^2$</div>

**Step 1**: Compute gradient w.r.t. final hidden state pre-activation $\mathbf{z}_T$ assuming we have only one sample $N=1$.

<div class="fragment" data-fragment-index="1">

Apply the **chain rule**: $\mathcal{L}$ depends on $\mathbf{z}_T$ through $\mathbf{h}_T$ and then $\hat{\mathbf{y}}$. For each hidden unit $k$, output dimension $j$:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial z_{T,k}} = \sum_{j=1}^{M_{\text{out}}} \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial \hat{y}_j}} \color{black}{\cdot} \color{#95E1D3}{\frac{\partial \hat{y}_j}{\partial h_{T,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial h_{T,k}}{\partial z_{T,k}}}
$$
</div>

<div>
where $\color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial \hat{y}_j} = 2(\hat{y}_{j} - y_{j})}$, $\color{#95E1D3}{\frac{\partial \hat{y}_k}{\partial h_{T,k}} = W_{hy,kj}}$ (from $\hat{\mathbf{y}} = \mathbf{W}_{hy} \mathbf{h}_T + \mathbf{b}_y$), and $\color{#4ECDC4}{\frac{\partial h_{T,k}}{\partial z_{T,k}} = \sigma'(z_{T,k})}$.
</div>

</div>

<div class="fragment" data-fragment-index="2">

In vector form for a single sample, this gives us the **error term**:

<div class="formula" style="margin-top: 20px;">
$$
\boldsymbol{\delta}_T = \left[\mathbf{W}_{hy}^\top (\hat{\mathbf{y}} - \mathbf{y})\right] \odot \sigma'(\mathbf{z}_T)
$$
</div>

where $\odot$ is element-wise multiplication.

</div>

</div>

---

## BPTT: Final Time Step

<div style="font-size: 0.80em;">

**Step 2**: Compute gradients w.r.t. weights and biases at final time step $T$

<div>
Given $\boldsymbol{\delta}_T = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_T}$ and forward pass $z_{T,k} = \sum_{i} W_{xh,ki} x_{T,i} + \sum_{l} W_{hh,kl} h_{T-1,l} + b_k$:
</div>

<div class="fragment" data-fragment-index="1">

**Input-to-hidden weight gradients**: Apply chain rule to $W_{xh,ki}$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial W_{xh,ki}} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{T,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{T,k}}{\partial W_{xh,ki}}} \color{black}{=} \color{#FF6B6B}{\delta_{T,k}} \color{black}{\cdot} \color{#4ECDC4}{x_{T,i}}
$$
</div>

In matrix form: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \boldsymbol{\delta}_T \mathbf{x}_T^\top$ (contribution from time $T$)

</div>

<div class="fragment appear-vanish" data-fragment-index="2">

**Bias gradients**: Apply chain rule to $b_k$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial b_k} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{T,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{T,k}}{\partial b_k}} \color{black}{=} \color{#FF6B6B}{\delta_{T,k}} \color{black}{\cdot} \color{#4ECDC4}{1} \color{black}{=} \delta_{T,k}
$$
</div>

</div>

<div class="fragment appear-vanish" data-fragment-index="3">

**Hidden-to-hidden weight gradients**: Apply chain rule to $W_{hh,kl}$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial W_{hh,kl}} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{T,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{T,k}}{\partial W_{hh,kl}}} \color{black}{=} \color{#FF6B6B}{\delta_{T,k}} \color{black}{\cdot} \color{#4ECDC4}{h_{T-1,l}}
$$
</div>
<div>
In matrix form: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \boldsymbol{\delta}_T \mathbf{h}_{T-1}^\top$ (contribution from time $T$)
</div>
</div>


</div>

---

## BPTT: Hidden Time Steps

<div style="font-size: 0.80em;">

**Step 3**: Propagate error backwards from time $t$ to time $t-1$

<div class="fragment" data-fragment-index="1">
To compute $\frac{\partial \mathcal{L}}{\partial z_{t-1,l}}$, we use the chain rule through time step $t$, as $\mathcal{L}$ depends on $z_{t-1,l}$ via all hidden units at time $t$:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial z_{t-1,l}} = \sum_{k=1}^{M_{\text{hidden}}} \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{t,k}}} \color{black}{\cdot} \color{#95E1D3}{\frac{\partial z_{t,k}}{\partial h_{t-1,l}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial h_{t-1,l}}{\partial z_{t-1,l}}}
$$
</div>

where $\color{#FF6B6B}{\delta_{t,k}}$ = error at next time step, $\color{#95E1D3}{\frac{\partial z_{t,k}}{\partial h_{t-1,l}} = W_{hh,kl}}$ = recurrent weight connecting time steps and $\color{#4ECDC4}{\frac{\partial h_{t-1,l}}{\partial z_{t-1,l}} = \sigma'(z_{t-1,l})}$

</div>

<div class="fragment" data-fragment-index="2">

This gives us the **error term** for time $t-1$:

<div class="formula" style="margin-top: 20px;">
$$
\delta_{t-1,l} = \left(\sum_{k=1}^{M_{\text{hidden}}} W_{hh,kl} \delta_{t,k}\right) \sigma'(z_{t-1,l})
$$
</div>
<div>
In vector form: $\boldsymbol{\delta}_{t-1} = \left[\mathbf{W}_{hh}^\top \boldsymbol{\delta}_t\right] \odot \sigma'(\mathbf{z}_{t-1})$
</div>
</div>

</div>

---

## BPTT: Hidden Time Steps

<div style="font-size: 0.80em;">

**Step 4**: Compute gradients w.r.t. weights and biases at time step $t$ (same as final time step!)

<div>
Given $\boldsymbol{\delta}_t = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_t}$ and forward pass $z_{t,k} = \sum_{i} W_{xh,ki} x_{t,i} + \sum_{l} W_{hh,kl} h_{t-1,l} + b_k$:
</div>

<div class="fragment" data-fragment-index="1">

**Input-to-hidden weight gradients**: Apply chain rule to $W_{xh,ki}$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial W_{xh,ki}} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{t,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{t,k}}{\partial W_{xh,ki}}} \color{black}{=} \color{#FF6B6B}{\delta_{t,k}} \color{black}{\cdot} \color{#4ECDC4}{x_{t,i}}
$$
</div>

In matrix form: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \boldsymbol{\delta}_t \mathbf{x}_t^\top$ (contribution from time $t$)

</div>

<div class="fragment appear-vanish" data-fragment-index="2">

**Bias gradients**: Apply chain rule to $b_j$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial b_j} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{t,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{t,k}}{\partial b_j}} \color{black}{=} \color{#FF6B6B}{\delta_{t,k}} \color{black}{\cdot} \color{#4ECDC4}{1} \color{black}{=} \delta_{t,k}
$$
</div>

</div>

<div class="fragment" data-fragment-index="3">

**Hidden-to-hidden weight gradients**: Apply chain rule to $W_{hh,kl}$

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial W_{hh,kl}} = \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial z_{t,k}}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial z_{t,k}}{\partial W_{hh,kl}}} \color{black}{=} \color{#FF6B6B}{\delta_{t,k}} \color{black}{\cdot} \color{#4ECDC4}{h_{t-1,l}}
$$
</div>
<div>
In matrix form: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \boldsymbol{\delta}_t \mathbf{h}_{t-1}^\top$ (contribution from time $t$)
</div>
</div>

</div>

---

## BPTT: Total Gradient Computation

<div style="font-size: 0.80em;">

**Step 5**: Sum gradients across all time steps

<div class="fragment" data-fragment-index="1">
Since the <strong>same weights</strong> $\mathbf{W}_{hh}$, $\mathbf{W}_{xh}$, and $\mathbf{b}$ are used at every time step, the total gradient is the sum of contributions from all time steps:
<div class="formula" style="margin-top: 20px;">
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} &= \sum_{t=1}^{T} \boldsymbol{\delta}_t \mathbf{h}_{t-1}^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} &= \sum_{t=1}^{T} \boldsymbol{\delta}_t \mathbf{x}_t^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} &= \sum_{t=1}^{T} \boldsymbol{\delta}_t
\end{aligned}
$$
</div>

</div>

</div>

---

## BPTT: Algorithm Summary

<div style="font-size: 0.80em;">

<div style="display: flex; gap: 40px;">

<div style="flex: 1;">

<strong>Forward Pass</strong>:
<ol>
<li>Initialize: $\mathbf{h}_0 = \mathbf{0}$ (or learned)</li>
<li>For $t = 1, \ldots, T$:
<ul>
<li>$\mathbf{z}_t = \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}$</li>
<li>$\mathbf{h}_t = \sigma(\mathbf{z}_t)$</li>
</ul>
</li>
<li>Output: $\hat{\mathbf{y}} = \mathbf{W}_{hy} \mathbf{h}_T + \mathbf{b}_y$</li>
<li>Loss: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$</li>
</ol>

</div>

<div style="flex: 1;">

<strong>Backward Pass</strong>:
<ol>
<li>Output gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_T}$ (from output layer)</li>
<li>Final time: $\boldsymbol{\delta}_T = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \odot \sigma'(\mathbf{z}_T)$</li>
<li>For $t = T-1, \ldots, 1$:
<ul>
<li>$\boldsymbol{\delta}_t = [\mathbf{W}_{hh}^\top \boldsymbol{\delta}_{t+1}] \odot \sigma'(\mathbf{z}_t)$</li>
</ul>
</li>
<li>Accumulate gradients:
<ul>
<li>$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t \mathbf{h}_{t-1}^\top$</li>
<li>$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t \mathbf{x}_t^\top$</li>
<li>$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t$</li>
</ul>
</li>
</ol>

</div>

</div>

</div>

---

## BPTT: Algorithm Summary

<div style="font-size: 0.80em; margin-top: 40px;">

**Weight Update** (Gradient Descent):

<div class="formula">
$$
\begin{aligned}
\mathbf{W}_{hh} & \leftarrow \mathbf{W}_{hh} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} \\
\mathbf{W}_{xh} & \leftarrow \mathbf{W}_{xh} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} \\
\mathbf{b} & \leftarrow \mathbf{b} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}}
\end{aligned}
$$
</div>

where $\eta$ is the learning rate.

</div>

---

## The Vanishing Gradient Problem

<div style="font-size: 0.85em;">

**Problem**: Gradients become exponentially small as they propagate back through time

<div class="fragment appear-vanish" data-fragment-index="0">
To compute the gradient at time $t$, we apply the chain rule through all time steps from $t+1$ to $T$:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}
$$
</div>

<div>
where each term comes from the RNN forward pass ($\mathbf{h}_k = \sigma(\mathbf{W}_{xh}\mathbf{x}_k + \mathbf{W}_{hh}\mathbf{h}_{k-1} + \mathbf{b})$):
</div>
<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \text{diag}(\sigma'(\mathbf{z}_k)) \mathbf{W}_{hh}
$$
</div>

This gives us the product of gradients:

</div>

<div class="formula fragment" style="margin-top: 20px;" data-fragment-index="0">
$$
\prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^{T} \text{diag}(\sigma'(\mathbf{z}_k)) \mathbf{W}_{hh}
$$
</div>

<div class="fragment" data-fragment-index="1">

**When does it vanish?**

- If $\|\mathbf{W}_{hh}\| < 1$ and activation derivatives $\sigma'(\mathbf{z}_k) < 1$ (e.g., sigmoid/tanh)
- After $T-t$ time steps, gradient magnitude: $\|\text{gradient}\| \approx (\|\mathbf{W}_{hh}\| \cdot \max_k \sigma'(\mathbf{z}_k))^{T-t}$
- For sigmoid: $\sigma'(z) \leq 0.25$, for tanh: $\tanh'(z) \leq 1$

</div>
</div>

<div class="fragment image-overlay highlight" data-fragment-index="3" style="text-align: left; top: 55%; width: 78%;">
<strong>Consequences</strong>:
<ul>
<li>Network cannot learn long-term dependencies (typically > 10-20 time steps)</li>
<li>Early layers receive negligible gradient updates</li>
</div>

</div>

---

## The Exploding Gradient Problem

<div style="font-size: 0.75em;">

**Problem**: Gradients become exponentially large as they propagate back through time

<div class="formula" style="margin-top: 20px;">
$$
\prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^{T} \text{diag}(\sigma'(\mathbf{z}_k)) \mathbf{W}_{hh}
$$
</div>

<div class="fragment" data-fragment-index="1">

**When does it explode?**

- If $\|\mathbf{W}_{hh}\| > 1$ and the product of derivatives grows unbounded
- Gradient magnitude: $\|\text{gradient}\| \approx (\|\mathbf{W}_{hh}\| \cdot \max_k \sigma'(\mathbf{z}_k))^{T-t}$
- Even with bounded activation derivatives, large $\|\mathbf{W}_{hh}\|$ can cause explosion

</div>

<div class="fragment" data-fragment-index="3">

**Common Solution: Gradient Clipping** - Limit gradient magnitude before parameter update:

<div class="formula" style="margin-top: 20px;">
$$
\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\
\theta \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \theta
\end{cases}
$$
</div>
where $\mathbf{g}$ is the gradient vector and $\theta$ is the clipping threshold.
</div>

</div>


<div class="fragment appear-vanish image-overlay highlight" data-fragment-index="2" style="text-align: left; width: 78%;">
<strong>Consequences</strong>:
<ul>
<li>Parameter updates become extremely large</li>
<li>Network weights oscillate wildly</li>
<li>Training becomes unstable, leading to NaN values</li>
<li>Model fails to converge</li>
</ul>
</div>

---

## Long Short-Term Memory (LSTM) Networks

<div style="font-size: 0.9em;">

- LSTMs are a type of recurrent neural network designed to mitigate the vanishing gradient problem and capture long-term dependencies in sequential data
- They use a memory cell with gating mechanisms (input, forget, output gates) to control information flow and maintain long-term dependencies

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/rnn_to_lstm.png" alt="Different Space Regions" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

</div>

---

## LSTM: Forget Gate

<div style="font-size: 0.85em;">
<strong>Forget Gate</strong>: Controls how much information from the previous cell state $\mathbf{c}_{t-1}$ is retained in current cell state $\mathbf{c}_t$.
<div class="formula" style="margin-top: 20px;">
$$\mathbf{f}_t = \sigma(\mathbf{W}_{xf} \mathbf{x}_t + \mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{b}_f)$$
</div>
where $\mathbf{f}_t$ is the forget gate vector (values between 0 and 1).
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/rnn_to_lstm.png" alt="Different Space Regions" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## LSTM: Input Gate

<div style="font-size: 0.85em;">
<strong>Input Gate</strong>: Controls how much new information from the current input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$ is added to the cell state $\mathbf{c}_t$.
<div class="formula" style="margin-top: 20px;">
$$\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_{xi} \mathbf{x}_t + \mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{b}_i) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_{xc} \mathbf{x}_t + \mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{b}_c)
\end{aligned}$$
</div>
where $\mathbf{i}_t$ is the input gate vector (values between 0 and 1) and $\tilde{\mathbf{c}}_t$ is the candidate cell state.
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/rnn_to_lstm.png" alt="Different Space Regions" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## LSTM: Cell State Update

<div style="font-size: 0.85em;">
<strong>Cell State Update</strong>: Combines the previous cell state $\mathbf{c}_{t-1}$ (modulated by forget gate) and the candidate cell state $\tilde{\mathbf{c}}_t$ (modulated by input gate) to form the new cell state $\mathbf{c}_t$.
<div class="formula" style="margin-top: 20px;">
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
</div>
where $\odot$ denotes element-wise multiplication.
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/rnn_to_lstm.png" alt="Different Space Regions" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## LSTM: Output Gate and Hidden State

<div style="font-size: 0.85em;">
<strong>Output Gate and Hidden State</strong>: Controls how much of the cell state $\mathbf{c}_t$ is exposed to the hidden state $\mathbf{h}_t$.
<div class="formula" style="margin-top: 20px;">
$$\begin{aligned}
\mathbf{o}_t &= \sigma(\mathbf{W}_{xo} \mathbf{x}_t + \mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{b}_o) \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}$$
</div>
where $\mathbf{o}_t$ is the output gate vector (values between 0 and 1).
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/rnn_to_lstm.png" alt="Different Space Regions" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## LSTM: Summary of Equations

<div style="font-size: 0.85em;">
The complete set of LSTM equations at time step $t$:
<div class="formula" style="margin-top: 20px;">
$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_{xf} \mathbf{x}_t + \mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_{xi} \mathbf{x}_t + \mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{b}_i) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_{xc} \mathbf{x}_t + \mathbf{W}_{hc} \mathbf{h}_{t-1} + \mathbf{b}_c) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_{xo} \mathbf{x}_t + \mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{b}_o) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}$$
</div>

</div>

<div class="fragment image-overlay" style="text-align: center; top: 62%; width: 78%;">
    <strong>PyTorch Documentation</strong>: <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html" target="_blank">torch.nn.LSTM</a>
</div>

---

## Gated Recurrent Unit (GRU)

<div style="font-size: 0.9em;">

- GRUs are a simplified variant of LSTMs that combine the forget and input gates into a single "update gate"
- They use fewer parameters than LSTMs while maintaining comparable performance
- GRUs have two gates: reset gate and update gate, making them computationally more efficient

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/gru_architecture.png" alt="GRU Architecture" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

</div>

---

## GRU: Forward Propagation - Reset Gate

<div style="font-size: 0.85em;">
<strong>Reset Gate</strong>: Controls how much of the previous hidden state $\mathbf{h}_{t-1}$ should be forgotten when computing the candidate hidden state.
<div class="formula" style="margin-top: 20px;">
$$\mathbf{r}_t = \sigma(\mathbf{W}_{xr} \mathbf{x}_t + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_r)$$
</div>
where $\mathbf{r}_t$ is the reset gate vector (values between 0 and 1).
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/gru_architecture.png" alt="GRU Architecture" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## GRU: Forward Propagation - Update Gate

<div style="font-size: 0.85em;">
<strong>Update Gate</strong>: Controls how much of the previous hidden state $\mathbf{h}_{t-1}$ to keep and how much of the candidate hidden state $\tilde{\mathbf{h}}_t$ to add.
<div class="formula" style="margin-top: 20px;">
$$\mathbf{z}_t = \sigma(\mathbf{W}_{xz} \mathbf{x}_t + \mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{b}_z)$$
</div>
where $\mathbf{z}_t$ is the update gate vector (values between 0 and 1).
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/gru_architecture.png" alt="GRU Architecture" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## GRU: Forward Propagation - Candidate Hidden State

<div style="font-size: 0.85em;">
<strong>Candidate Hidden State</strong>: Computes new information that could be added to the hidden state, using the reset gate to selectively forget parts of $\mathbf{h}_{t-1}$.
<div class="formula" style="margin-top: 20px;">
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h)$$
</div>
where $\tilde{\mathbf{h}}_t$ is the candidate hidden state and $\mathbf{r}_t \odot \mathbf{h}_{t-1}$ applies the reset gate to the previous hidden state.
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/gru_architecture.png" alt="GRU Architecture" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## GRU: Forward Propagation - Hidden State Update

<div style="font-size: 0.85em;">
<strong>Hidden State Update</strong>: Combines the previous hidden state $\mathbf{h}_{t-1}$ and candidate hidden state $\tilde{\mathbf{h}}_t$ using the update gate $\mathbf{z}_t$.
<div class="formula" style="margin-top: 20px;">
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$
</div>
where:
- $(1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1}$ keeps parts of the old hidden state
- $\mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ adds parts of the new candidate hidden state
</div>

<div style="text-align: center;">
    <img src="assets/images/05-recurrent_layers/gru_architecture.png" alt="GRU Architecture" style="width: 87%; margin-top: 40px;">
    <div class="reference" style="text-align: center; margin-top: 10px;">Source: https://github.com/acids-ircam/creative_ml</div>
</div>

---

## GRU: Summary of Equations

<div style="font-size: 0.85em;">
The complete set of GRU equations at time step $t$:
<div class="formula" style="margin-top: 20px;">
$$\begin{aligned}
\mathbf{r}_t &= \sigma(\mathbf{W}_{xr} \mathbf{x}_t + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_r) \\
\mathbf{z}_t &= \sigma(\mathbf{W}_{xz} \mathbf{x}_t + \mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{b}_z) \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\end{aligned}$$
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="0" style="text-align: center; top: 42%; width: 78%;">
    <strong>PyTorch Documentation</strong>: <a href="https://pytorch.org/docs/stable/generated/torch.nn.GRU.html" target="_blank">torch.nn.GRU</a>
</div>

<div class="fragment" data-fragment-index="1" style="margin-top: 40px;">

**Key differences from LSTM**:
- No separate cell state — hidden state serves both roles
- Fewer parameters: 3 weight matrices per gate vs. 4 in LSTM
- Update gate implicitly combines forget and input gates: $(1 - \mathbf{z}_t)$ forgets, $\mathbf{z}_t$ adds new info

</div>

</div>

---

## Gates Beyond LSTM and GRU

- Gating mechanisms can be integrated into other architectures, such as convolutional neural networks (CNNs) and transformer models
- Gates help control information flow, improve gradient propagation, and enhance model performance across various tasks
- Examples include attention gates in transformers and gated convolutional layers in CNNs

---

# Python Implementation

