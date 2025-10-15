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

<div class="fragment" data-fragment-index="1"></div>

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

<div style="font-size: 0.9em;">

<ul>
    <li>$f_{\boldsymbol{\theta}} \in \mathcal{F}_{\Theta}$: A specific function parameterized by $\boldsymbol{\theta}$ belongs to the function space $\mathcal{F}_{\Theta}$.</li>
    <li>$\boldsymbol{\theta} \in \Theta$: The parameters $\boldsymbol{\theta}$ come from the parameter space $\Theta$ (e.g., $\Theta = \mathbb{R}^d$ for $d$ parameters).</li>
    <li>$\mathcal{F}_{\Theta} = \lbrace f_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta\rbrace$: The family of all functions obtained by varying $\boldsymbol{\theta}$ over $\Theta$.</li>
</ul>

**Examples of Function Spaces**:

<ul>
    <li>$\mathcal{F}$: All possible functions $\mathcal{X} \to \mathcal{Y}$ (infinite, intractable)</li>
    <li>$\mathcal{F}_1$: Linear functions, $\Theta = \mathbb{R}^2$, e.g., $f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x$</li>
    <li>$\mathcal{F}_d$: Polynomial functions of degree $d$, $\Theta = \mathbb{R}^{d+1}$, e.g., $f_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x + \ldots + \theta_d x^d$</li>
    <li>$\mathcal{F}_2^{(2)}$: Quadratic functions, $\Theta = \mathbb{R}^6$, e.g., $f_{\boldsymbol{\theta}}(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2 + \theta_5 x_1 x_2$</li>
</ul>

**Important**: $\mathcal{F}_1 \subset \mathcal{F}_2 \subset \mathcal{F}_d \subset \mathcal{F}$ — more complex models have larger function spaces and can represent more patterns, but require more data to learn effectively.

</div>

---

## Finding the Optimal Parameters

<div style="font-size: 0.9em;">

After defining the function space and loss function, we can try to find the optimal parameters $$\boldsymbol{\theta}^*$$ that minimize the empirical risk.

**Objective**: Find the optimal parameters $$\boldsymbol{\theta}^*$$ that minimize the empirical risk:

<div class="formula">
$$
\boldsymbol{\theta}^* = \arg\min\limits_{\boldsymbol{\theta} \in \Theta} R(\boldsymbol{\theta}) = \arg\min\limits_{\boldsymbol{\theta} \in \Theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)
$$
</div>

The loss function $$\mathcal{L}$$ quantifies the difference between the predicted output $$f_{\boldsymbol{\theta}}(\mathbf{x}_i) = \hat{\mathbf{y}}_i$$ and the true label $$\mathbf{y}_i$$. **Examples of loss functions**:

<ul>
    <li><strong>Mean Absolute Error (MAE) (L1 Loss)</strong>: $\mathcal{L}_{\text{MAE}} = \lVert \hat{\mathbf{y}} - \mathbf{y}_i \rVert_1$</li>
    <li><strong>Mean Squared Error (MSE) (L2 Loss)</strong>: $\mathcal{L}_{\text{MSE}} = \lVert \hat{\mathbf{y}} - \mathbf{y}_i \rVert_2^2$</li>
    <li><strong>Hinge Loss</strong>: $\mathcal{L}_{\text{Hinge}} = \max(0, 1 - y_i \hat{y}_i)$ for binary labels $y_i \in \{-1, 1\}$</li>
</ul>

</div>

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

## Example: Linear Regression

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/LinearRegressionSimple.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Binary Classification

<div style="text-align: center;">
    <video width="70%" data-autoplay loop muted controls>
        <source src="assets/videos/02-machine_learning_fundamentals/1080p60/BinaryClassificationSimple.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>
