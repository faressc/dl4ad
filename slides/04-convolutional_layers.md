# Convolutional Layers

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
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1943:0,1957:0,1965:0,1979:1,2012:1">
        {{TIMELINE:timeline_early_nn_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Methods for efficient learning and gradient-based optimization</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1967:0,1970:0,1986:0,1992:0,2009:1,2010:0,2012:0">
        {{TIMELINE:timeline_early_nn_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Datasets</div>
        <div class="timeline-text">Tools, platforms, and milestones that enabled practical deep learning</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="2002:0">
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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2016:1">
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

<div class="image-overlay fragment highlight" style="width: 78%; text-align: left;">

Can linear layers retain temporal information?

- Yes, but limited: Linear layers learn position-specific patterns (e.g., "value at t=5 relates to value at t=8")
- No translation invariance—same pattern at different positions must be learned separately
- Fixed input length, parameters scale poorly with sequence length (O(n²))

</div>

---

## Recap: Finite Impulse Response (FIR) Filters

- Digital filters implemented by convolving the input signal with a finite impulse response

<div style="font-size: 0.85em;">

**Time domain:**

<div class="formula">
$$
y[n] = \sum_{k=0}^{M-1} h[k] x[n-k]
$$
</div>

where $y[n]$ is the output, $x[n]$ is the input, $h[k]$ is the impulse response, and $M$ is the filter order.

**Frequency domain (via DFT):**

<div class="formula">
$$
\begin{aligned}
H(e^{j\omega}) & = \sum_{k=0}^{M-1} h[k] e^{-j\omega k} \\
Y(e^{j\omega}) & = H(e^{j\omega}) X(e^{j\omega})
\end{aligned}
$$
</div>

</div>

<div class="image-overlay fragment highlight" style="width: 78%;">

FIR filters are inherently stable, meaning they do not produce unbounded output for a bounded input.

</div>

---

## FIR Filters Animation

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/04-convolutional_layers/1080p60/FIRConvolution1D.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Convolutional Layers

<div style="font-size: 0.85em;">

- Convolutional layers apply learnable FIR filters to input data, enabling the model to capture local patterns
- They are translation invariant, meaning the same filter is applied across the entire input
- Convolutional layers can handle variable-length inputs and have fewer parameters compared to fully connected layers
- Commonly used in many audio tasks

**1D Convolution:**

<div class="formula">
$$
y[n] = \sum_{k=0}^{M-1} w[k] x[n-k] + b
$$
</div>

with learnable weights $w[k]$ of size $M$ and bias $b$. The output length is given by:

<div class="formula">
$$
L_{out} = L_{in} - M + 1
$$
</div>

---

## 2D Convolutional Layers

<div style="font-size: 0.85em;">

- 2D convolutional layers extend the concept of 1D convolutions to two-dimensional data, such as images
- They apply learnable 2D filters (kernels) to capture spatial patterns in the input
- Commonly used in computer vision tasks for feature extraction and pattern recognition

**2D Convolution:**

<div class="formula">
$$
y[m,n] = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} w[i,j] x[m-i,n-j] + b
$$
</div>

where $w[i,j]$ is the 2D filter of size $M \times N$ and $b$ is the bias.

The output has dimensions:

<div class="formula">
$$
H_{out} = H_{in} - M + 1, \quad W_{out} = W_{in} - N + 1
$$
</div>

</div>

</div>

<div class="fragment appear-vanish image-overlay" style="width: 45%; text-align: center;" data-fragment-index="0">
    <img src="assets/images/04-convolutional_layers/2d_convolution.gif" style="width: 100%;" alt="2D Convolution Illustration">
</div>

<div class="fragment appear-vanish image-overlay" style="width: 45%; text-align: center;" data-fragment-index="1">
    <img src="assets/images/04-convolutional_layers/no_padding_no_strides.gif" style="width: 100%;" alt="2D Convolution Illustration">
    <div class="reference">Source: <a href="https://github.com/vdumoulin/conv_arithmetic/tree/master">https://github.com/vdumoulin/conv_arithmetic/tree/master</a></div>
</div>

<div class="fragment appear-vanish image-overlay" style="width: 45%; text-align: center;" data-fragment-index="2">
    <img src="assets/images/04-convolutional_layers/vision_kernels.png" style="width: 100%;" alt="Vision Kernels">
</div>

---

## Transposed Convolutional Layers

<div style="font-size: 0.85em;">

- Transposed convolutional layers, also known as deconvolutional layers, are used for upsampling feature maps
- They reverse the spatial transformation of convolutional layers, increasing the spatial dimensions of the input
- Commonly used in generative models and image segmentation tasks

**1D Transposed Convolution:**
<div class="formula">
$$
y[n] = \sum_{k=0}^{M-1} w[k] \cdot x'\left[n+k\right] + b
$$
</div>

where $x'[i]$ is the input with $p = M - 1$ zeros at each edge. The output length is given by:

<div class="formula">
$$
L_{out} = L_{in} + M - 1
$$
</div>

<div class="fragment appear-vanish image-overlay" style="width: 45%; text-align: center;">
    <img src="assets/images/04-convolutional_layers/no_padding_no_strides_transposed.gif" style="width: 100%;" alt="2D Convolution Illustration">
    <div class="reference">Source: <a href="https://github.com/vdumoulin/conv_arithmetic/tree/master">https://github.com/vdumoulin/conv_arithmetic/tree/master</a></div>
</div>

</div>

---

## Options in Convolutional Layers

<div style="display: flex; gap: 40px; align-items: flex-start;">

<div style="flex: 1;" class="fragment" data-fragment-index="0">

<span style="font-size: 0.85em"><strong>Padding</strong>: Adding $p$ extra points at the edges of the input to control the spatial dimensions of the output</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/same_padding_no_strides.gif" style="width: 100%;" alt="Padding Illustration">
</div>

</div>

<div style="flex: 1;" class="fragment" data-fragment-index="1">

<span style="font-size: 0.85em"><strong>Strides</strong>: The step size $s$ with which the filter moves across the input, affecting the output size</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/padding_strides.gif" style="width: 100%;" alt="Strides Illustration">
</div>

</div>

<div style="flex: 1;" class="fragment" data-fragment-index="2">

<span style="font-size: 0.85em"><strong>Dilations</strong>: Spacing out the filter elements by a factor of $d$ to increase the receptive field without increasing parameters</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/dilation.gif" style="width: 100%;" alt="Dilation Illustration">
</div>

</div>

</div>

<div class="reference" style="text-align: center; margin-top: 20px;">Source: <a href="https://github.com/vdumoulin/conv_arithmetic/tree/master">https://github.com/vdumoulin/conv_arithmetic/tree/master</a></div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="3" style="text-align: center; width: 80%;">

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=1, padding=0, dilation=1, groups=1,
                      bias=True, padding_mode='zeros', device=None,
                      dtype=None)
```

</div>

---

## Multi-Head Convolutional Layers

- Multi-head convolutional layers consist of multiple parallel convolutional kernels (heads) that process the input simultaneously
- Each head learns different features from the input, allowing the model to capture a diverse set of patterns
- Each kernel has its own set of weights and biases, and gets convolved with all input channels

**Multi-Head 1D Convolution:**

<div class="formula">
$$
y_{j}[n] = \sum_{i=0}^{C_{in}-1} \sum_{k=0}^{M-1} w_{i,j}[k] x_{i}[n-k] + b_{j}
$$
</div>

where $C_{in}$ is the number of input channels, $y_{j}[n]$ is the output of head $j$, $x_{i}[n]$ is the input from channel $i$, $w_{i,j}[k]$ are the weights for head $j$ and channel $i$, and $b_{j}$ is the bias for head $j$.

<div class="fragment appear-vanish image-overlay" style="width: 60%; text-align: center;">

Grouped convolutions are a special case of multi-head convolutions where each head processes a distinct subset of input channels.

</div>

---

## Options in Transposed Convolutional Layers

<div style="display: flex; gap: 40px; align-items: flex-start;">

<div style="flex: 1;" class="fragment" data-fragment-index="0">

<span style="font-size: 0.85em"><strong>Padding</strong>: Reduces the inherent padding of transposed convolution to $p' = M - 1 - p$</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/same_padding_no_strides.gif" style="width: 100%;" alt="Padding Illustration">
</div>

</div>

<div style="flex: 1;" class="fragment" data-fragment-index="1">

<span style="font-size: 0.85em"><strong>Strides</strong>: Inserts $(s-1)$ zeros between input elements</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/padding_strides_transposed.gif" style="width: 100%;" alt="Strides Illustration">
</div>

</div>

<div style="flex: 1;" class="fragment" data-fragment-index="2">

<span style="font-size: 0.85em"><strong>Dilations</strong>: Spacing out the filter elements by a factor of $d$ to increase the receptive field without increasing parameters</span>

<div style="margin-top: 20px;">
    <img src="assets/images/04-convolutional_layers/dilation.gif" style="width: 100%;" alt="Dilation Illustration">
</div>

</div>

</div>

<div class="reference" style="text-align: center; margin-top: 20px;">Source: <a href="https://github.com/vdumoulin/conv_arithmetic/tree/master">https://github.com/vdumoulin/conv_arithmetic/tree/master</a></div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="3" style="text-align: center; width: 80%;">

```python
class torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                               stride=1, padding=0, output_padding=0,
                               groups=1, bias=True, dilation=1, 
                               padding_mode='zeros', device=None,
                               dtype=None)
```

</div>

---

## Ambiguities in Transposed Convolutions

- Transposed convolutions can lead to ambiguities in output size due to the interplay of padding, stride, and kernel size
- The `output_padding` parameter is used to resolve these ambiguities by specifying additional size to add to the output shape

<div style="font-size: 0.85em; width: 70%; top: 120%" class="fragment image-overlay" data-fragment-index="0">
<table style="width: 100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th><strong>Standard Convolution</strong></th>
            <th><strong>Transposed Convolution</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Input size: 6×6, Kernel: 3×3, Stride: 2, Padding: 1<br>Output size: 3×3</td>
            <td>Input size: 3×3, Kernel: 3×3, Stride: 2, Padding: 1<br>Output size: 5×5 or 6×6?</td>
        </tr>
        <tr>
            <td style="text-align: center;"><img src="assets/images/04-convolutional_layers/padding_strides_odd.gif" style="width: 70%;" alt="Standard Convolution"></td>
            <td style="text-align: center;"><img src="assets/images/04-convolutional_layers/padding_strides_odd_transposed.gif" style="width: 70%;" alt="Transposed Convolution"></td>
        </tr>
    </tbody>
</table>

<div class="reference" style="margin-top: 10px;">Source: <a href="https://github.com/vdumoulin/conv_arithmetic/tree/master">https://github.com/vdumoulin/conv_arithmetic/tree/master</a></div>

</div>

---

## AlexNet Example

- AlexNet is a pioneering convolutional neural network architecture that achieved significant success in image classification tasks
- Consists of: convolutional layers followed by fully connected layers, utilizing ReLU activations and max pooling for downsampling
- AlexNet marked a breakthrough in deep learning

<div style="text-align: center; margin-top: 40px;">
    <img src="assets/images/01-history/alexnet.png" alt="Deep Learning Era Timeline" style="width: 1200px; height: auto;">
    <div class="reference" style="margin-top: 10px; text-align: center;">
        Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
    </div>
</div>

---

## Pooling Layers

- Pooling layers reduce the spatial dimensions of the input, helping to decrease computational load and control overfitting
- Common types of pooling include max pooling and average pooling

**1D Max Pooling:**

<div class="formula">
$$
y[n] = \max\limits_{0 \leq k < M} x[n \cdot s + k]
$$
</div>

where $M$ is the pool size and $s$ is the stride. The output length is given by:

<div class="formula">
$$
L_{out} = \left\lfloor \frac{L_{in} - M}{s} \right\rfloor + 1
$$
</div>

<div class="fragment appear-vanish image-overlay" style="width: 60%; text-align: center;">

<img src="assets/images/04-convolutional_layers/2d_maxpool.gif" style="width: 100%;" alt="2D Max Pooling Illustration">

<div style="margin-top: 40px;">2D Max Pooling with pool size 2x2 and stride 2.</div>

</div>

---

## WaveNet Example

- WaveNet is a deep generative model for raw audio waveforms that utilizes dilated causal convolutions to capture long-range temporal dependencies
- It employs multiple layers of dilated convolutions with increasing dilation factors, allowing the receptive field to grow exponentially with depth

<div style="text-align: center; margin-top: 40px;" class="fragment appear-vanish" data-fragment-index="0">
    <img src="assets/images/01-history/wavenet_before.png" alt="WaveNet Before" style="width: 1300px; height: auto;">
    <div class="reference" style="margin-top: 10px; text-align: center;">
        Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio (No. arXiv:1609.03499). https://doi.org/10.48550/arXiv.1609.03499
    </div>
</div>

<div style="text-align: center; margin-top: 40px;" class="fragment appear-vanish" data-fragment-index="1">
    <img src="assets/images/01-history/wavenet_after.png" alt="WaveNet Before" style="width: 1300px; height: auto;">
    <div class="reference" style="margin-top: 10px; text-align: center;">
        Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio (No. arXiv:1609.03499). https://doi.org/10.48550/arXiv.1609.03499
    </div>
</div>

---

## Differentiating Convolutional Layers

<div class="highlight" style="margin-top: 150px; text-align: center;">
How do we compute gradients for convolutional layers during backpropagation?
</div>

<div class="fragment">
Need to compute gradients of the loss $\mathcal{L}$ with respect to:

<div class="formula" style="margin-top: 40px;">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial b} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{x}}
$$
</div>

where $\mathbf{x}$ is the input (needed for backpropagating to previous layers).

</div>

---

## Forward Pass: 1D Convolution

<div style="font-size: 0.85em;">

Recall the forward pass for a 1D convolutional layer:

<div class="formula" style="margin-top: 40px;">
$$
y[n] = \sum_{k=0}^{M-1} w[k] x[n-k] + b
$$
</div>

<div class="fragment" data-fragment-index="1">

Or in vector form, for all output positions:

<div class="formula" style="margin-top: 20px;">
$$
\mathbf{y} = \mathbf{w} \ast \mathbf{x} + b
$$
</div>

where $\ast$ denotes the convolution operation.

</div>

<div class="fragment" data-fragment-index="2" style="margin-top: 40px;">

**Key observation**: Each output $y[n]$ depends on a **local window** of inputs $x[n], x[n-1], \ldots, x[n-M+1]$.

</div>

</div>

---

## Backward Pass: Gradient w.r.t. Bias

<div style="font-size: 0.80em;">

**Step 1**: Compute gradient w.r.t. bias $b$

<div class="fragment" data-fragment-index="1">

Assume we have the gradient from the next layer: $\frac{\partial \mathcal{L}}{\partial y[n]}$ for each output position $n$.

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{n} \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial y[n]}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial y[n]}{\partial b}}
$$
</div>

</div>

<div class="fragment" data-fragment-index="2">

Since $y[n] = \sum_{k=0}^{M-1} w[k] x[n-k] + b$, we have $\color{#4ECDC4}{\frac{\partial y[n]}{\partial b} = 1}$.

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{n} \frac{\partial \mathcal{L}}{\partial y[n]}
$$
</div>

</div>

<div class="fragment image-overlay highlight" data-fragment-index="3" style="width: 80%; text-align: left;">

**Key Insight**: The bias gradient is simply the **sum** of all output gradients, since the bias contributes equally to every output position.

</div>

</div>

---

## Backward Pass: Gradient w.r.t. Weights

<div style="font-size: 0.75em;">

**Step 2**: Compute gradient w.r.t. weight $w[k]$ for each kernel position $k$

<div class="fragment" data-fragment-index="1">

Apply the chain rule:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial w[k]} = \sum_{n} \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial y[n]}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial y[n]}{\partial w[k]}}
$$
</div>

</div>

<div class="fragment" data-fragment-index="2">

Since $y[n] = \sum_{k'=0}^{M-1} w[k'] x[n-k'] + b$, we have:

<div class="formula" style="margin-top: 20px;">
$$
\color{#4ECDC4}{\frac{\partial y[n]}{\partial w[k]} = x[n-k]}
$$
</div>

</div>

<div class="fragment" data-fragment-index="3">

Therefore:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial w[k]} = \sum_{n} \frac{\partial \mathcal{L}}{\partial y[n]} \cdot x[n-k]
$$
</div>

</div>

<div class="fragment image-overlay highlight" data-fragment-index="4" style="width: 80%; text-align: left;">

**Key Insight**: The gradient w.r.t. $w[k]$ is a **convolution** between the output gradient and the input!

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \ast \mathbf{x}$$

</div>

</div>

---

## Backward Pass: Gradient w.r.t. Input

<div style="font-size: 0.75em;">

**Step 3**: Compute gradient w.r.t. input $x[m]$ for each input position $m$

<div class="fragment" data-fragment-index="1">

Apply the chain rule. Each $x[m]$ contributes to multiple outputs:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial x[m]} = \sum_{n} \color{#FF6B6B}{\frac{\partial \mathcal{L}}{\partial y[n]}} \color{black}{\cdot} \color{#4ECDC4}{\frac{\partial y[n]}{\partial x[m]}}
$$
</div>

</div>

<div class="fragment" data-fragment-index="2">

Since $y[n] = \sum_{k=0}^{M-1} w[k] x[n-k] + b$, we have $\color{#4ECDC4}{\frac{\partial y[n]}{\partial x[m]} = w[k]}$ when $m = n - k$ (i.e., $k = n - m$), and $0$ otherwise.

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial x[m]} = \sum_{n} \frac{\partial \mathcal{L}}{\partial y[n]} \cdot w[n-m]
$$
</div>

</div>

<div class="fragment" data-fragment-index="3">

Reindexing with $k = n - m$:

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial x[m]} = \sum_{k} \frac{\partial \mathcal{L}}{\partial y[m+k]} \cdot w[k]
$$
</div>

</div>

<div class="fragment image-overlay highlight" data-fragment-index="4" style="width: 80%; text-align: left;">

**Key Insight**: The gradient w.r.t. input is a **convolution** with the **flipped kernel**!

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \ast \text{flip}(\mathbf{w})$$

</div>

</div>

---

## Backpropagation: Algorithm Summary

<div style="font-size: 0.70em;">

<div style="display: flex; gap: 40px;">

<div style="flex: 1;">

**Forward Pass**:

<div class="formula" style="margin-top: 20px;">
$$
y[n] = \sum_{k=0}^{M-1} w[k] x[n-k] + b
$$
</div>

Or in vector form:

<div class="formula" style="margin-top: 20px;">
$$
\mathbf{y} = \mathbf{w} \ast \mathbf{x} + b
$$
</div>

</div>

<div style="flex: 1;">

**Backward Pass**:

Given $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$, compute:

<div class="formula" style="margin-top: 20px;">
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b} & = \sum_{n} \frac{\partial \mathcal{L}}{\partial y[n]} \\
\\
\frac{\partial \mathcal{L}}{\partial w[k]} & = \sum_{n} \frac{\partial \mathcal{L}}{\partial y[n]} \cdot x[n-k] \\
\\
\frac{\partial \mathcal{L}}{\partial x[m]} & = \sum_{k} \frac{\partial \mathcal{L}}{\partial y[m+k]} \cdot w[k]
\end{aligned}
$$
</div>

</div>

</div>

</div>

<div class="fragment" data-fragment-index="1" style="font-size: 0.75em; margin-top: 40px;">

**In compact notation**:

<div class="formula">
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b} & = \mathbf{1}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \\
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} & = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \ast \mathbf{x} \\
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} & = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \ast \text{flip}(\mathbf{w})
\end{aligned}
$$
</div>

</div>

<div class="fragment image-overlay highlight" data-fragment-index="2" style="width: 80%; text-align: left;">

**Key Takeaway**: Backpropagation through convolution is also a convolution!
- Weight gradient: convolve output gradient with input
- Input gradient: convolve output gradient with flipped kernel

</div>

---

## Multi-Channel Convolution: Forward Pass

<div style="font-size: 0.80em;">

For multi-channel inputs and outputs (recall multi-head convolutions):

<div class="formula" style="margin-top: 40px;">
$$
y_j[n] = \sum_{i=0}^{C_{in}-1} \sum_{k=0}^{M-1} w_{i,j}[k] x_i[n-k] + b_j
$$
</div>

where:
- $C_{in}$ is the number of input channels
- $C_{out}$ is the number of output channels
- $x_i[n]$ is the input from channel $i$
- $y_j[n]$ is the output for channel $j$
- $w_{i,j}[k]$ are the weights connecting input channel $i$ to output channel $j$
- $b_j$ is the bias for output channel $j$

<div class="fragment" style="margin-top: 40px;">

In tensor notation:

<div class="formula" style="margin-top: 20px;">
$$
\mathbf{Y} = \mathbf{W} \ast \mathbf{X} + \mathbf{b}
$$
</div>

where $\mathbf{X} \in \mathbb{R}^{C_{in} \times L_{in}}$, $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times M}$, $\mathbf{Y} \in \mathbb{R}^{C_{out} \times L_{out}}$.

</div>

</div>

---

## Multi-Channel Convolution: Backward Pass

<div style="font-size: 0.75em;">

Given $\frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{C_{out} \times L_{out}}$, compute gradients:

<div class="fragment" data-fragment-index="1">

**Bias gradient** (for each output channel $j$):

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial b_j} = \sum_{n} \frac{\partial \mathcal{L}}{\partial y_j[n]}
$$
</div>

</div>

<div class="fragment" data-fragment-index="2">

**Weight gradient** (for each input-output channel pair $(i,j)$ and kernel position $k$):

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial w_{i,j}[k]} = \sum_{n} \frac{\partial \mathcal{L}}{\partial y_j[n]} \cdot x_i[n-k]
$$
</div>

Or in compact form: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{:,i,j}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_j} \ast \mathbf{x}_i$

</div>

<div class="fragment" data-fragment-index="3">

**Input gradient** (for each input channel $i$ and position $m$):

<div class="formula" style="margin-top: 20px;">
$$
\frac{\partial \mathcal{L}}{\partial x_i[m]} = \sum_{j=0}^{C_{out}-1} \sum_{k} \frac{\partial \mathcal{L}}{\partial y_j[m+k]} \cdot w_{i,j}[k]
$$
</div>

Or in compact form: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} = \sum_{j=0}^{C_{out}-1} \frac{\partial \mathcal{L}}{\partial \mathbf{y}_j} \ast \text{flip}(\mathbf{w}_{i,j})$

</div>

</div>

---

## Convolutional vs. Fully Connected Gradients

<div style="font-size: 0.85em;">

**Fully Connected Layer**:
- Weight gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \boldsymbol{\delta} \mathbf{h}^\top$ (outer product)
- Each weight connects to **all** inputs and outputs
- Gradient computation: $O(M \times N)$ for $M$ outputs and $N$ inputs

<div class="fragment" data-fragment-index="1" style="margin-top: 40px;">

**Convolutional Layer**:
- Weight gradient: $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \ast \mathbf{x}$ (convolution)
- Each weight connects to **local** windows only
- Gradient computation: $O(L \times M)$ for length $L$ and kernel size $M$

</div>

</div>

<div class="fragment image-overlay highlight" data-fragment-index="2" style="width: 80%; text-align: left;">

**Key Advantage**: Convolutional layers have **significantly fewer parameters** and **cheaper gradient computation** compared to fully connected layers!

- Parameters: $M \times C_{in} \times C_{out}$ vs. $L_{in} \times L_{out}$
- Shared weights across positions enable translation invariance
- Backpropagation remains efficient through convolution operations

</div>

---

