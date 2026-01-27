# Latent Models
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
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1763:0,1812:0,1815:0,1922:0">
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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2015:0,2016:0,2017:0,2021:0">
        {{TIMELINE:timeline_deep_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Advanced learning techniques and representation learning breakthroughs</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2013:0,2014:0,2015:0,2016:0">
        {{TIMELINE:timeline_deep_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2017:0,2018:0,2020:0,2022:0,2023:0">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>


---

## Recap: Probability Fundamentals

<div style="font-size: 0.75em;">

**Foundation:** Random variables ($X$, $Y$) with distributions (PMF/PDF), characterized by expectation, joint/marginal/conditional probabilities

**Bayes' Theorem:** $p_{Y|X}(y|x) = \frac{p_{X|Y}(x|y) \cdot p_Y(y)}{p_X(x)}$ — connects posterior, likelihood, and prior

**Decision Rules (Classification):**
- Bayesian: $\arg\min_{\hat{y}} \sum_{y} \mathcal{L}(y, \hat{y}) \cdot p_{Y|X}(y|x)$ (minimize expected loss)
- MAP: $\arg\max_{\hat{y}} p_{X|Y}(x|\hat{y}) p_Y(\hat{y})$ (0-1 loss → maximize posterior)
- ML: $\arg\max_{\hat{y}} p_{X|Y}(x|\hat{y})$ (uniform prior → maximize likelihood)

**Parameter Estimation (Training):**
- Bayesian: $p_{\Theta|\mathcal{D}}(\boldsymbol{\theta}|\mathcal{D}) = \frac{p_{\mathcal{D}|\Theta}(\mathcal{D}|\boldsymbol{\theta}) \cdot p_\Theta(\boldsymbol{\theta})}{p_{\mathcal{D}}(\mathcal{D})}$ (full posterior distribution)
- MAP: $\arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p_{Y|X,\Theta}(y_i|\mathbf{x}_i, \boldsymbol{\theta}) \cdot p_\Theta(\boldsymbol{\theta})$ (mode of posterior = **regularization**)
- MLE: $\arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p_{Y|X,\Theta}(y_i|\mathbf{x}_i, \boldsymbol{\theta})$ (uniform prior)

**Key:** Same probabilistic framework applies to both **what to predict** (classification) and **how to learn** (training)

</div>

---

## Supervised Learning

<div style="font-size: 0.75em;">

<img src="assets/images/09-latent_models/supervised_learning.svg" alt="Supervised Learning Diagram" style="float: right; width: 30%; margin-left: 20px; margin-bottom: 20px;">

Our previous machine learning models were primarily focused on supervised learning tasks.

**Dataset Structure:**

In supervised learning, we have access to a labeled dataset:

<div class="formula" style="width: 60%; margin-left: 0;">
  $$
\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n
  $$
</div>

where each input $\mathbf{x}_i$ is paired with a corresponding output $\mathbf{y}_i$.

**Probabilistic Formulation:**

We can frame supervised learning probabilistically by assuming the data is generated from some conditional distribution $p_{Y|X,\Theta}(y|\mathbf{x}, \boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$. Using Bayes' theorem, the posterior distribution over parameters is:

<div class="formula">
  $$
  p_{\Theta|X,Y}(\boldsymbol{\theta}|\mathbf{X}, \mathbf{Y}) = \frac{p_{X,Y|\Theta}(\mathbf{X}, \mathbf{Y}|\boldsymbol{\theta}) \cdot p_\Theta(\boldsymbol{\theta})}{p_{X,Y}(\mathbf{X}, \mathbf{Y})}
  $$
</div>

**Parameter Estimation:**

In practice, we typically estimate a single "best" parameter value rather than computing the full posterior. We can use techniques like Maximum Likelihood Estimation (MLE) or Maximum A Posteriori (MAP):

<div class="formula">
  $$
\theta_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p_{Y|X,\Theta}(\mathbf{y}_i|\mathbf{x}_i, \boldsymbol{\theta}) \cdot p_\Theta(\boldsymbol{\theta})
  $$
</div>

</div>

---

## Unsupervised Learning

<div style="font-size: 0.75em;">

<img src="assets/images/09-latent_models/unsupervised_learning_no_latents.svg" alt="Unsupervised Learning Diagram" style="float: right; width: 10%; margin-left: 20px; margin-bottom: 20px; margin-right: 200px;">

In unsupervised learning, we work with unlabeled data and aim to discover hidden structure.

**Dataset Structure:**

We only have access to observations without corresponding labels:

<div class="formula" style="width: 60%; margin-left: 0;">
  $$
\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n
  $$
</div>

**Probabilistic Formulation:**

We model the data distribution directly as $p_{X|\Theta}(\mathbf{x}|\boldsymbol{\theta})$. The goal is to find parameters that explain the observed data:

<div class="formula">
  $$
  p_{\Theta|X}(\boldsymbol{\theta}|\mathbf{X}) = \frac{p_{X|\Theta}(\mathbf{X}|\boldsymbol{\theta}) \cdot p_\Theta(\boldsymbol{\theta})}{p_X(\mathbf{X})}
  $$
</div>

**Parameter Estimation:**

Using Maximum Likelihood Estimation, we find parameters that maximize the probability of observing the data:

<div class="formula">
  $$
\boldsymbol{\theta}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p_{X|\Theta}(\mathbf{x}_i|\boldsymbol{\theta})
  $$
</div>

</div>

---

## Latent Variables

<div style="font-size: 0.75em;">

<img src="assets/images/09-latent_models/unsupervised_learning.svg" alt="Latent Variable Model" style="float: right; width: 30%; margin-left: 20px; margin-bottom: 20px;">

So far, we have focused on datasets with fully observed variables. However, in many real-world scenarios, some variables are **unobserved** or **hidden**.

**Motivation:**

- Data often has underlying structure not directly measurable
- Examples: topics in documents, speaker identity in audio, object pose in images

**Latent Variable Model:**

We introduce latent variables $\mathbf{z}$ to capture hidden factors:

<div class="formula" style="width: 60%; margin-left: 0;">
  $$
p_{X|\Theta}(\mathbf{x}|\boldsymbol{\theta}) = \int p_{X|Z,\Theta}(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \cdot p_{Z|\Theta}(\mathbf{z}|\boldsymbol{\theta}) \, d\mathbf{z}
  $$
</div>

**Key Insight:**

The observed data $\mathbf{x}$ is generated by first sampling a latent variable $\mathbf{z}$ from a prior $p_{Z|\Theta}(\mathbf{z}|\boldsymbol{\theta})$, then generating $\mathbf{x}$ conditioned on $\mathbf{z}$.

**Challenge:** The integral (marginalization over $\mathbf{z}$) is often intractable!

</div>

---

