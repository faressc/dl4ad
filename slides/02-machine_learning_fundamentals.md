# Machine Learning<br>Fundamentals

---

## Mathematical Foundations

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Calculus & Linear Algebra</div>
        <div class="timeline-text">Basis for optimization algorithms and machine learning model operations</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1676:0,1805:0,1847:0,1951:0">
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
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;">
        {{TIMELINE:timeline_early_nn_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Methods for efficient learning and gradient-based optimization</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;" data-timeline-fragments-select="1967:0,1970:0,1986:0">
        {{TIMELINE:timeline_early_nn_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Datasets</div>
        <div class="timeline-text">Tools, platforms, and milestones that enabled practical deep learning</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;">
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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
        {{TIMELINE:timeline_deep_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>

---

## Machine Learning

<div style="font-size: 0.9em;">

**Definition**: Learning a function that maps inputs to outputs based on labeled training examples.

**Goal**: Minimize the difference between predicted and actual outputs.

**Mathematical Formulation**:

- Given a dataset $D = \lbrace(\mathbf{x}_i, \mathbf{y}_i)\rbrace$ for $i = 1, \ldots, N$, where $\mathbf{x}_i \in \mathcal{X}$ are input features and $\mathbf{y}_i \in \mathcal{Y}$ are corresponding labels and $N$ is the number of samples.

- Find a function $$f_{\boldsymbol{\theta}}: \mathcal{X} \to \mathcal{Y}$$ parameterized by $$\boldsymbol{\theta}$$ that minimizes the empirical risk:

<div class="formula">
$$
R(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)
$$
</div>

where $$\mathcal{L}$$ is a loss function (e.g., Mean Squared Error for regression, Hinge Loss for classification).

</div>

---

## Defining the Function Space

<div style="font-size: 0.8em;">

<ul>
    <li>$f_{\boldsymbol{\theta}} \in \mathcal{F}_{\Theta}$: A specific function parameterized by $\boldsymbol{\theta}$ belongs to the function space $\mathcal{F}_{\Theta}$.</li>
    <li>$\boldsymbol{\theta} \in \Theta$: The parameters $\boldsymbol{\theta}$ come from the parameter space $\Theta$ (e.g., $\Theta = \mathbb{R}^d$ for $d$ parameters).</li>
    <li>$\mathcal{F}_{\Theta} = \lbrace f_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta\rbrace$: The family of all functions obtained by varying $\boldsymbol{\theta}$ over $\Theta$.</li>
</ul>

**Examples of Function Spaces**:

<ul>
    <li>$\mathcal{F}$: All possible functions $\mathcal{X} \to \mathcal{Y}$ (infinite, intractable)</li>
    <li>$\mathcal{F}_1^{(1)}$: Linear functions in 1 variable, $\Theta = \mathbb{R}^2$, $f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x$</li>
    <!-- <li>$\mathcal{F}_d^{(1)}$: Polynomial functions of degree $d$ in 1 variable, $\Theta = \mathbb{R}^{d+1}$, $f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x + \ldots + \theta_d x^d$</li> -->
    <li>$\mathcal{F}_1^{(n)}$: Linear functions in $n$ variables, $\Theta = \mathbb{R}^{n+1}$, $f_{\boldsymbol{\theta}}(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \ldots + \theta_n x_n$</li>
    <li>$\mathcal{F}_{\text{logistic}}$: Logistic functions, $\Theta = \mathbb{R}^3$, $f_{\boldsymbol{\theta}}(x) = \frac{\theta_0}{1 + e^{-\theta_1(x - \theta_2)}}$ (S-shaped curves for growth/saturation)</li>
    <li>$\mathcal{F}_d^{(n)}$ Polynomial functions of degree $d$ in $n$ input variables, $\Theta = \mathbb{R}^{\binom{n+d}{d}}$
</ul>

**Important**: For a fixed dimension $n$, we have $\mathcal{F}_1^{(n)} \subset \mathcal{F}_2^{(n)} \subset \ldots \subset \mathcal{F}_d^{(n)} \subset \mathcal{F}$ — more complex models have larger function spaces and can represent more patterns, but require more data to learn effectively.

</div>

---

## Finding the Optimal Parameters

<div style="font-size: 0.9em;">

**Objective**: Find the optimal parameters $$\boldsymbol{\theta}^*$$ in the parameter space $\Theta$ from a function space $\mathcal{F}_{\Theta}$ that minimize the empirical risk:

<div class="formula">
$$
\boldsymbol{\theta}^* = \arg\min\limits_{\boldsymbol{\theta} \in \Theta} R(\boldsymbol{\theta}) = \arg\min\limits_{\boldsymbol{\theta} \in \Theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)
$$
</div>

The loss function $$\mathcal{L}$$ quantifies the difference between the predicted output $$f_{\boldsymbol{\theta}}(\mathbf{x}_i) = \hat{\mathbf{y}}_i$$ and the true label $$\mathbf{y}_i$$.

**Examples of loss functions**:

<ul>
    <li><strong>Mean Absolute Error (L1 Loss)</strong>: $\mathcal{L}_{\text{MAE}} = \lVert \hat{\mathbf{y}} - \mathbf{y}_i \rVert_1$</li>
    <li><strong>Mean Squared Error (L2 Loss)</strong>: $\mathcal{L}_{\text{MSE}} = \lVert \hat{\mathbf{y}} - \mathbf{y}_i \rVert_2^2$</li>
    <li><strong>Hinge Loss</strong>: $\mathcal{L}_{\text{Hinge}} = \max(0, 1 - y_i \hat{y}_i)$ for binary labels $y_i \in \{-1, 1\}$</li>
</ul>

</div>

---

## Residual Errors

<div style="font-size: 0.85em;">

- We assume the true dataset is distributed according to the function $f^*$ plus noise $\epsilon$

<div class="formula">
$$
y_i = f^*(\mathbf{x}_i) + \epsilon_i\text{, }
$$
</div>

where $\epsilon_i$ represents inherent noise or randomness in the data generation process. E.g., normal distribution:

<div class="formula">
$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{\epsilon_i^2}{2\sigma^2}}
$$
</div>

- Residual errors measure the difference between predictions and observations:

<div class="formula">
$$
\begin{aligned}
r_i & = y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)\\
r_i & = \underbrace{[f^*(\mathbf{x}_i) - f_{\boldsymbol{\theta}}(\mathbf{x}_i)]}_{\text{approximation error}} + \underbrace{\epsilon_i}_{\text{irreducible noise}}
\end{aligned}
$$
</div>

</div>

<div class="image-overlay fragment highlight" style="width: 70%">
Even with the optimal parameters $\boldsymbol{\theta}^*$ and infinite training data, certain loss functions will not reach zero due to: (1) irreducible noise $\epsilon_i$ (always present), and (2) approximation error when $f^* \notin \mathcal{F}_{\Theta}$ (model class limitation).
</div>
---

## Over- and Underfitting

<div style="font-size: 0.9em;">

**Balancing Model Complexity**:

- **Overfitting**: Even when $f^* \in \mathcal{F}_{\Theta}$, using overly complex models (e.g., high-degree polynomials) can lead to fitting noise rather than the underlying pattern, resulting in poor generalization to new data.

- **Underfitting**: When $f^* \notin \mathcal{F}_{\Theta}$, the model class is too restrictive to capture the true data-generating process, leading to high approximation error on both training and new data.

The goal is to select a function space $\mathcal{F}_{\Theta}$ that balances expressiveness with generalization capability.

</div>

---

## Over- and Underfitting

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/QuadraticRegressionOverUnderfit.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## How to test this?

---

## The Key Components

<div style="font-size: 0.9em;">

<div class="grid" style="display: grid; grid-template-columns: auto 1fr; gap: 1em; align-items: start;">

<div style="text-align: center; font-size: 1.5em; font-weight: bold; color: var(--fs-highlight-background)">0</div>
<div>
<strong>Dataset</strong> ($D$): A collection of input-output pairs $D = \lbrace(\mathbf{x}_i, \mathbf{y}_i)\rbrace_{i=1}^{N}$. The dataset must be representative of the true underlying function $f^*: \mathcal{X} \to \mathcal{Y}$.
</div>

<div style="text-align: center; font-size: 1.5em; font-weight: bold; color: var(--fs-highlight-background)">1</div>
<div>
<strong>Function or Model</strong> ($f_{\boldsymbol{\theta}}$): A parameterized function that maps inputs $\mathbf{x}_i$ to predicted outputs $\hat{\mathbf{y}}_i = f_{\boldsymbol{\theta}}(\mathbf{x}_i)$. The choice of function defines the function space $\mathcal{F}_{\Theta}$.
</div>

<div style="text-align: center; font-size: 1.5em; font-weight: bold; color: var(--fs-highlight-background)">2</div>
<div>
<strong>Parameters</strong> ($\boldsymbol{\theta}$): The set of parameters that define the specific function within the function space. These parameters are adjusted during training to minimize the empirical risk.
</div>

<div style="text-align: center; font-size: 1.5em; font-weight: bold; color: var(--fs-highlight-background)">3</div>
<div>
<strong>Loss Function</strong> ($\mathcal{L}$): A function that quantifies the difference between the predicted outputs $\hat{\mathbf{y}}_i$ and the true labels $\mathbf{y}_i$. The choice of loss function depends on the task (e.g., regression vs. classification).
</div>

<div style="text-align: center; font-size: 1.5em; font-weight: bold; color: var(--fs-highlight-background)">4</div>
<div>
<strong>Optimization Algorithm</strong>: A method for adjusting the parameters $\boldsymbol{\theta}$ to minimize the empirical risk $R(\boldsymbol{\theta})$. Common algorithms include Gradient Descent and its variants.
</div>

</div>

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

## Simple Linear Regression Formulation

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

---

## Binary Classification

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/BinaryClassificationSimple.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>
