# Perceptrons

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
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1943:1,1957:1,1965:1">
        {{TIMELINE:timeline_early_nn_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Methods for efficient learning and gradient-based optimization</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1967:0,1970:0,1986:0,1992:1,2010:1">
        {{TIMELINE:timeline_early_nn_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Datasets</div>
        <div class="timeline-text">Tools, platforms, and milestones that enabled practical deep learning</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1998:1,2002:1">
        {{TIMELINE:timeline_early_nn_software}}
    </div>
</div>

<div class="fragment" data-fragment-index="1"></div>

---

## The Deep Learning Era

<!-- Layers & Architectures Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Deep architectures</div>
        <div class="timeline-text">Deep architectures and generative models transforming AI capabilities</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
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

## McCulloch-Pitts Neuron (1943)

<div style="text-align: center; margin-top: 80px;">
    <img src="assets/images/03-perceptrons/perceptron-neuroscience.png" alt="Neuroscience Inspirations" style="width: 1200px;">
    <div class="reference">https://github.com/acids-ircam/creative_ml</div>
</div>

---

## McCulloch-Pitts Neuron (1943)

<div style="display: flex; align-items: center; gap: 40px; margin-top: 80px;">
    <div style="flex: 1;">
        <img src="assets/images/03-perceptrons/perceptron-neuroscience.png" alt="Neuroscience Inspirations" style="width: 80%;">
        <div class="reference" style="text-align: center;">https://github.com/acids-ircam/creative_ml</div>
    </div>
    <div style="flex: 1;">
        Neuron is the weighted $w_i$ sum of its inputs $x_i$ and fires if the sum exceeds a threshold $T$:
        <div class="formula" style="margin-top: 60px;">
            $$
                y = 
                \begin{cases} 
                    1 & \text{if } \sum_{i=1}^{N} w_i x_i \geq T \\ 
                    0 & \text{otherwise}
                \end{cases}
            $$
        </div>
    </div>
</div>
<div class="formula fragment" data-fragment-index="1", style="font-size: 0.8em; margin-top: 80px; text-align: center;">
$$
    y = 
    \begin{cases} 
        1 & \text{if } \sum_{i=1}^{N} w_i x_i \geq T \\ 
        0 & \text{otherwise}
    \end{cases}
    \quad\to\quad
    y = \phi\left( \sum_{i=1}^{N} w_i x_i - T \right)\text{ where } \phi(z) = 
    \begin{cases} 
        1 & \text{if } z \geq 0 \\
        0 & \text{otherwise}
    \end{cases}
$$
</div>

---

## Frank Rosenblatt's Perceptron (1958)

<div style="text-align: center; margin-top: 80px;">
    <img src="assets/images/01-history/perceptron.svg" alt="Frank Rosenblatt" style="width: 1400px;">
    <p style="font-size: 0.9em; color: var(--fs-text-color-muted); margin-top: 5px;">The Perceptron Model</p>
</div>

<div class="image-overlay fragment">
What are the learnable parameters in this model?
</div>

---

## Frank Rosenblatt's Perceptron (1958)

<div style="text-align: center; margin-top: 80px;">
    <img src="assets/images/03-perceptrons/perceptron_learnable_params.svg" alt="Frank Rosenblatt" style="width: 1400px;">
    <p style="font-size: 0.9em; color: var(--fs-text-color-muted); margin-top: 5px;">The Perceptron Model</p>
</div>

---

## Frank Rosenblatt's Perceptron (1958)

<div style="display: flex; align-items: center; gap: 40px; margin-top: 80px;">
    <div style="flex: 1;">
        <img src="assets/images/01-history/perceptron.svg" alt="Frank Rosenblatt" style="width: 80%;">
    </div>
    <div style="flex: 1;">
        The perceptron introduces a bias term $b$ to shift the activation threshold:
        <div class="formula" style="margin-top: 60px;">
            $$
                y = \phi\left( \sum_{i=1}^{N} w_i x_i + b \right) \\
                \text{where } \phi(z) = 
                \begin{cases} 
                    1 & \text{if } z \geq 0 \\
                    0 & \text{otherwise}
                \end{cases}
            $$
        </div>
    </div>
</div>
<div class="fragment appear-vanish" data-fragment-index="1", style="display:flex; font-size: 0.8em; margin-top: 80px; text-align: left; align-items: center; gap: 20px;">
    <div style="flex: 1;">
        In vector notation, this can be expressed as:
    </div>
    <div class="formula" style="flex: 1;">
    $$
        y = \phi\left( \mathbf{w}^\top \mathbf{x} + b \right)
    $$
    </div>
</div>

<div class="fragment appear-vanish" data-fragment-index="2", style="display:flex; font-size: 0.8em; margin-top: 80px; text-align: left; align-items: center; gap: 20px;">
    <div style="flex: 1;">
        Or if $\mathbf{x}$ includes a bias input $x_{0} = 1$, we can fold $b$ into the weights:
    </div>
    <div class="formula" style="flex: 1;">
    $$
        y = \phi\left( \mathbf{w}^\top \mathbf{x} \right) \text{ where }  \mathbf{x} = [1, x_1, x_2, \ldots, x_N]^\top \text{ and } \mathbf{w} = [b, w_1, w_2, \ldots, w_N]^\top
    $$
</div>

---

## Recall: Simple Linear Regression

- **Function**: $f_{\boldsymbol{\theta}}(x): \mathbb{R} \to \mathbb{R}$ defined as:

<div class="formula">
$$
f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x
$$
</div>

- **Parameter space**: $\Theta = \mathbb{R}^2$ with parameters $\boldsymbol{\theta} = (\theta_0, \theta_1)$
- **Dataset**: $D = \lbrace(x_i, y_i)\rbrace$ for $i = 1, \ldots, N$
- **Input space**: $\mathcal{X} = \mathbb{R}$
- **Output space**: $\mathcal{Y} = \mathbb{R}$
- **Loss function**: Mean Squared Error (MSE):

<div class="formula">
$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - f_{\boldsymbol{\theta}}(x_i))^2
$$
</div>

---

## Example: Simple Linear Regression

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/LinearRegressionSimple.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Comparison to Linear Regression

<div style="font-size: 0.9em;">

<div class="formula">
$$
\begin{aligned}
\text{Perceptron: } & \quad f_{\mathbf{w}}(\mathbf{x}) = \phi\left( \mathbf{w}^\top \mathbf{x} \right), \quad \phi(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}\\
\text{Linear Regression: } & \quad f_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x} \text{, with } \mathbf{x} = [1, x_1, x_2, \ldots, x_N]^\top
\end{aligned}
$$
</div>

**Key Differences**:

- **Output space**: Perceptron outputs binary labels $\mathcal{Y} = \{0, 1\}$; Linear regression outputs continuous values $\mathcal{Y} = \mathbb{R}$
- **Function space**: Both belong to $\mathcal{F}_1^{(n)}$ **before** activation — perceptron adds non-linearity via $\phi$

</div>

---

## Example: Binary Classification

- **Function**: $f_{\boldsymbol{\theta}}(x): \mathbb{R}^2 \to \mathbb{R}$ defined as:

<div class="formula">
$$
f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 \text{, with } \hat{y} = \text{sign}(f_{\boldsymbol{\theta}}(x))
$$
</div>

- **Parameter space**: $\Theta = \mathbb{R}^3$ with parameters $\boldsymbol{\theta} = (\theta_0, \theta_1, \theta_2)$
- **Dataset**: $D = \lbrace(x_i, y_i)\rbrace$ for $i = 1, \ldots, N$
- **Input space**: $\mathcal{X} = \mathbb{R}^2$
- **Output space**: $\mathcal{Y} = \lbrace -1, +1 \rbrace$ (binary labels)
- **Loss function**: Mean hinge loss:

<div class="formula">
$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i f_{\boldsymbol{\theta}}(x_i))
$$
</div>

---

## Binary Classification

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/BinaryClassificationSimple.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Comparison to Binary Classification

<div style="font-size: 0.9em;">

Both the perceptron and linear binary classifiers perform **binary classification** using linear decision boundaries:

<div class="formula" style="margin-top: 40px;">
$$
\begin{aligned}
\text{Perceptron: } & \quad f_{\mathbf{w}}(\mathbf{x}) = \phi\left( \mathbf{w}^\top \mathbf{x} \right), \quad \phi(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}\\
\text{Linear Classifier: } & \quad f_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x} \text{, with } \hat{y} = \text{sign}(f_{\boldsymbol{\theta}}(\mathbf{x})) = \begin{cases} +1 & \text{if } f_{\boldsymbol{\theta}}(\mathbf{x}) \geq 0 \\ -1 & \text{otherwise} \end{cases}
\end{aligned}
$$
</div>

**Key Similarities**:

- **Decision boundary**: Both use a linear hyperplane to separate classes
- **Function space**: Both belong to $\mathcal{F}_1^{(n)}$ before applying the output function

</div>

<div class="fragment highlight image-overlay">
If we change the perceptron activation to a sign function, both models become equivalent!

→ This means we can use the same training algorithm for both models!
</div>

---

## Differentiable Activation Functions

To enable gradient flow through the activation as well, we can use differentiable alternatives such as:

<div style="margin-top: 40px; font-size: 0.85em;">

<table>
<thead>
<tr>
<th>Activation</th>
<th>Function</th>
<th>Derivative</th>
</tr>
</thead>
<tbody>
<tr class="fragment" data-fragment-index="1">
<td><strong>Sigmoid</strong></td>
<td>$\sigma(z) = \frac{1}{1 + e^{-z}}$</td>
<td>$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$</td>
</tr>
<tr class="fragment" data-fragment-index="3">
<td><strong>Tanh</strong></td>
<td>$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$</td>
<td>$\frac{d\tanh}{dz} = 1 - \tanh^2(z)$</td>
</tr>
<tr class="fragment" data-fragment-index="5">
<td><strong>ReLU</strong></td>
<td>$\text{ReLU}(z) = \max(0, z)$</td>
<td>$\frac{d\text{ReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$</td>
</tr>
<tr class="fragment" data-fragment-index="7">
<td><strong>Leaky ReLU</strong></td>
<td>$\text{LeakyReLU}(z) = \max(\alpha z, z)$</td>
<td>$\frac{d\text{LeakyReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{otherwise} \end{cases}$</td>
</tr>
</tbody>
</table>

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="2" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/SigmoidActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/TanhActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="6" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/ReLUActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="8" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/LeakyReLUActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment" data-fragment-index="9"></div>

<div class="fragment image-overlay highlight" data-fragment-index="10">
Our gradients can now flow through the activation function!<br>
→ We can use loss functions that depend on the final output of the perceptron!<br>
→ We can chain multiple perceptrons together to form multi-layer perceptrons (MLPs)!
</div>

---

## Example: Binary Classification with MSE

- **Function**: $f_{\boldsymbol{\theta}}(x): \mathbb{R}^2 \to (-1, 1)$ defined as:

<div class="formula">
$$
f_{\boldsymbol{\theta}}(x) = \tanh(w_0 + w_1 x_1 + w_2 x_2)
$$
</div>

- **Parameter space**: $\Theta = \mathbb{R}^3$ with parameters $\boldsymbol{\theta} = (w_0, w_1, w_2)$
- **Dataset**: $D = \lbrace(x_i, y_i)\rbrace$ for $i = 1, \ldots, N$
- **Input space**: $\mathcal{X} = \mathbb{R}^2$
- **Output space**: $\mathcal{Y} = (-1, 1)$
- **Loss function**: Mean squared error (MSE):

<div class="formula">
$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \left(y_i - \hat{y}_i \right)^2
$$
</div>

---

## Example: Binary Classification with Sigmoid

- **Function**: $f_{\boldsymbol{\theta}}(x): \mathbb{R}^2 \to (0, 1)$ defined as:

<div class="formula">
$$
f_{\boldsymbol{\theta}}(x) = \sigma(w_0 + w_1 x_1 + w_2 x_2) \text{, with } \sigma(z) = \frac{1}{1 + e^{-z}}
$$
</div>

- **Parameter space**: $\Theta = \mathbb{R}^3$ with parameters $\boldsymbol{\theta} = (w_0, w_1, w_2)$
- **Dataset**: $D = \lbrace(x_i, y_i)\rbrace$ for $i = 1, \ldots, N$
- **Input space**: $\mathcal{X} = \mathbb{R}^2$
- **Output space**: $\mathcal{Y} = (0, 1)$ (probabilistic outputs)
- **Loss function**: Binary cross-entropy loss:

<div class="formula">
$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
</div>

<div class="fragment appear-vanish image-overlay" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/CrossEntropyLossVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Multilayer Perceptrons

<div style="display: flex; gap: 100px; justify-content: center; margin-top: 40px;">
    <img src="assets/images/03-perceptrons/single_neuron.svg" alt="Layer 1" style="width: 28%;">
    <img src="assets/images/03-perceptrons/single_layer.svg" alt="Layer 2" style="width: 28%;">
    <img src="assets/images/03-perceptrons/multi_layer.svg" alt="Layer 3" style="width: 28%;">
</div>

<div style="display: grid; grid-template-columns: 1fr 15fr 1fr 15fr; gap: 20px; align-items: center; margin-top: 60px;">
    <img src="assets/images/03-perceptrons/input_legend.svg" alt="Detail 1" style="width: 100%;">
    <div style="font-size: 0.9em;">
        <strong>Input Values</strong>: $$\mathbf{x} = [x_1, x_2, \ldots, x_N]^\top$$ represent the features fed into the perceptron.
    </div>
    <img src="assets/images/03-perceptrons/bias_legend.svg" alt="Detail 2" style="width: 100%;">
    <div style="font-size: 0.9em;">
        <strong>Bias Term</strong>: $$1$$ is added to the input vector to allow shifting the activation threshold.
    </div>
    <img src="assets/images/03-perceptrons/output_legend.svg" alt="Detail 3" style="width: 100%;">
    <div style="font-size: 0.9em;">
        <strong>Output Values</strong>: $$\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_K]^\top$$ represent the predicted outputs of the perceptron.
    </div>
    <img src="assets/images/03-perceptrons/hidden_legend.svg" alt="Detail 3" style="width: 100%;">
    <div style="font-size: 0.9em;">
        <strong>Hidden Units</strong>: $$\mathbf{h}^{(l)} = [h_1^{(l)}, h_2^{(l)}, \ldots, h_{M^{(l)}}^{(l)}]^\top$$ represent intermediate computations within the $l$-th layer.
    </div>
</div>

<div class="reference" style="text-align: center; margin-top: 20px;">Generated with https://alexlenail.me/NN-SVG/</div>

---

## Forward Propagation

<div style="display: flex; gap: 60px; align-items: flex-start; margin-top: 40px; font-size: 0.85em;">

<div style="flex: 1;">

**Hidden Layer Computation:**

<div class="formula">
$$
\begin{aligned}
\mathbf{z}^{(l)} & = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\text{ or} \\
\mathbf{z}^{(l)} & = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)}\\
\mathbf{h}^{(l)} & = \sigma(\mathbf{z}^{(l)})
\end{aligned}
$$
</div>

where:
- $\mathbf{W}^{(l)} \in \mathbb{R}^{M \times M'}$ is the weight matrix
- $\mathbf{b}^{(l)} \in \mathbb{R}^{M}$ is the bias vector
- $\sigma(\cdot)$ is the activation function
- $\mathbf{h}^{(0)} = \mathbf{x}$ (input layer)

</div>

<div style="flex: 1;">

**Output Layer Computation:**

<div class="formula">
$$
\begin{aligned}
\mathbf{z}^{(L)} & = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)} \text{ or} \\
\mathbf{z}^{(L)} & = \mathbf{W}^{(L)} \mathbf{h}^{(L)}\\
\hat{\mathbf{y}} & = \sigma_{L}(\mathbf{z}^{(L)})
\end{aligned}
$$
</div>

where:
- $\mathbf{W}^{(L)} \in \mathbb{R}^{K \times M}$ is the output weight matrix
- $\mathbf{b}^{(L)} \in \mathbb{R}^{K}$ is the output bias vector
- $\sigma_{L}(\cdot)$ is the output activation
- $L$ is the index of the last hidden layer

</div>

</div>

<div class="fragment image-overlay" style="width: 80%;">
In element-wise form, each neuron computes:
<div class="formula" style="margin-top: 20px;">
$$
\begin{aligned}
z_j^{(l)} & = \sum_{i=1}^{M'} W_{ji}^{(l)} h_i^{(l-1)} + b_j^{(l)} \\
h_j^{(l)} & = \sigma(z_j^{(l)})
\end{aligned}
$$
</div>
where:

- $i$ indexes neurons in the previous layer
- $j$ indexes neurons in the current layer

</div>

---

## Backpropagation: The Chain Rule

<div style="text-align: center; margin-top: 40px;">
    <img src="assets/images/03-perceptrons/simple_mlp_backprop.svg" alt="Simple MLP for Backpropagation" style="width: 800px;">
</div>

<div style="font-size: 0.8em; margin-top: 40px;">

**The Challenge**: How do we train a multilayer perceptron with potentially many layers?

<div class="fragment" data-fragment-index="1">

**Goal**: Compute gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}$ for **each layer** $l$ to update parameters using gradient descent.

</div>

<div class="fragment" data-fragment-index="2">

**The Problem**: The loss $\mathcal{L}$ depends on the output $\hat{\mathbf{y}}$, which depends on the last layer's weights $\mathbf{W}^{(L)}$, which depends on the previous layer's output $\mathbf{h}^{(L-1)}$, which depends on $\mathbf{W}^{(L-1)}$, and so on...

</div>

<div class="fragment" data-fragment-index="3">

**The Solution**: **Backpropagation** - an efficient algorithm that uses the **chain rule** to compute gradients by propagating errors backwards through the network.

</div>

</div>

<div class="fragment" data-fragment-index="4" style="font-size: 0.75em; margin-top: 40px; padding: 20px; background: rgba(100, 100, 255, 0.1); border-left: 4px solid #6666ff;">

**Chain Rule Reminder**: If $y = f(g(x))$, then $\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

For neural networks: $\mathcal{L} = f(\mathbf{h}^{(L)}(\mathbf{h}^{(L-1)}(...\mathbf{h}^{(1)}(\mathbf{x}))))$

We apply the chain rule **recursively** to compute how the loss changes with respect to parameters in **any layer**.

</div>

---

## Backpropagation: Output Layer

<div style="font-size: 0.75em; margin-top: 30px;">

For a **regression task** with MSE loss: $\mathcal{L} = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$

**Step 1**: Compute gradient with respect to output layer pre-activation $\mathbf{z}^{(L)}$

<div class="fragment" data-fragment-index="1">

We need to apply the **chain rule** since $\mathcal{L}$ depends on $\mathbf{z}^{(L)}$ through $\hat{\mathbf{y}}$:

<div class="formula">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \odot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}^{(L)}}
$$
</div>

where $\odot$ denotes element-wise multiplication (Hadamard product).

</div>

<div class="fragment" data-fragment-index="2">

**Computing each term**:
- Loss gradient: $\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} = \frac{\partial}{\partial \hat{\mathbf{y}}} \left[\frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2\right] = -(\mathbf{y} - \hat{\mathbf{y}})$ (prediction error)
- Activation derivative: $\frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}^{(L)}} = \sigma'(\mathbf{z}^{(L)})$ (how sensitive the activation is)

</div>

<div class="fragment" data-fragment-index="3">

Combining these gives us the **error term** $\boldsymbol{\delta}^{(L)}$:

<div class="formula">
$$
\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} = -(\mathbf{y} - \hat{\mathbf{y}}) \odot \sigma'(\mathbf{z}^{(L)})
$$
</div>

$\boldsymbol{\delta}^{(L)}$ tells us **how much each output neuron's pre-activation should change** to reduce the loss.

</div>

</div>

---

## Backpropagation: Output Layer (cont.)

<div style="font-size: 0.75em; margin-top: 30px;">

**Step 2**: Compute gradients with respect to weights and biases

Now that we have $\boldsymbol{\delta}^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}}$, we can find how the loss changes with respect to the parameters.

<div class="fragment" data-fragment-index="1">

**Recall the forward pass**: $\mathbf{z}^{(L)} = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$

Applying the chain rule again:

<div class="formula">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(L)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{W}^{(L)}} = \boldsymbol{\delta}^{(L)} (\mathbf{h}^{(L-1)})^\top
$$
</div>

**Interpretation**: The gradient for weight $W_{ji}^{(L)}$ is proportional to:
- How much neuron $j$ needs to change ($\delta_j^{(L)}$)
- How active the input from neuron $i$ was ($h_i^{(L-1)}$)

</div>

<div class="fragment" data-fragment-index="2">

For the bias term:

<div class="formula">
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(L)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{b}^{(L)}} = \boldsymbol{\delta}^{(L)}
$$
</div>

**Interpretation**: Bias gradient equals the error term directly (since $\frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{b}^{(L)}} = \mathbf{I}$)

</div>

</div>

---

## Backpropagation: Hidden Layers

<div style="font-size: 0.8em; margin-top: 40px;">

**Step 3**: Propagate error backwards to hidden layer $l$:

<div class="formula">
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} & = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} \odot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{z}^{(l)}} \\
& = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l+1)}} \frac{\partial \mathbf{z}^{(l+1)}}{\partial \mathbf{h}^{(l)}} \odot \sigma'(\mathbf{z}^{(l)}) \\
& = \left[(\mathbf{W}^{(l+1)})^\top \boldsymbol{\delta}^{(l+1)}\right] \odot \sigma'(\mathbf{z}^{(l)}) \\
& = \boldsymbol{\delta}^{(l)}
\end{aligned}
$$
</div>

</div>

<div class="fragment" data-fragment-index="1" style="font-size: 0.8em; margin-top: 40px;">

**Step 4**: Compute gradients with respect to weights and biases:

<div class="formula">
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} & = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} & = \boldsymbol{\delta}^{(l)}
\end{aligned}
$$
</div>

</div>

---

## Backpropagation: Algorithm Summary

<div style="font-size: 0.75em; margin-top: 40px;">

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

<div class="fragment" data-fragment-index="1" style="font-size: 0.75em; margin-top: 40px;">

**Weight Update** (Gradient Descent):

<div class="formula">
$$
\begin{aligned}
\mathbf{W}^{(l)} & \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \\
\mathbf{b}^{(l)} & \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
\end{aligned}
$$
</div>

where $\eta$ is the learning rate.

</div>

---

## Backpropagation: Element-wise View

<div style="font-size: 0.75em; margin-top: 40px;">

For a single neuron $j$ in layer $l$, the gradient with respect to its weight $W_{ji}^{(l)}$ is:

<div class="formula">
$$
\frac{\partial \mathcal{L}}{\partial W_{ji}^{(l)}} = \delta_j^{(l)} h_i^{(l-1)}
$$
</div>

where the error term $\delta_j^{(l)}$ is computed as:

<div class="formula">
$$
\delta_j^{(l)} = \begin{cases}
\left(\frac{\partial \mathcal{L}}{\partial \hat{y}_j}\right) \sigma'(z_j^{(L)}) & \text{if } l = L \text{ (output layer)} \\
\\
\left(\sum_{k=1}^{M^{(l+1)}} W_{kj}^{(l+1)} \delta_k^{(l+1)}\right) \sigma'(z_j^{(l)}) & \text{if } l < L \text{ (hidden layer)}
\end{cases}
$$
</div>

</div>

<div class="fragment image-overlay highlight" style="width: 80%;">
**Key Insight**: Each neuron's error $\delta_j^{(l)}$ depends on:
1. The weighted sum of errors from neurons in the next layer
2. The derivative of its own activation function

This recursive structure enables efficient gradient computation through the chain rule!
</div>

---

## Backpropagation & Weight Updates


