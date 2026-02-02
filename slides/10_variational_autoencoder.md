# Variational Autoencoder

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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2013:1,2015:0,2016:0,2017:0,2021:0">
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

## Recap: Latent Models

<div style="font-size: 0.75em;">

**Latent Variable Models:** Introduce hidden $\mathbf{z}$ to model complex distributions; marginal likelihood: $p(\mathbf{x}|\boldsymbol{\theta}) = \int p(\mathbf{x}, \mathbf{z}|\boldsymbol{\theta}) \, d\mathbf{z}$

**GMM (Discrete Latent):** $p(\mathbf{x}|\boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ — tractable sum, but **log-of-sum** prevents closed-form MLE

**EM Algorithm:** Iteratively optimize when direct MLE is intractable
- **E-Step:** Compute responsibilities $\gamma_{ik} = p(z_i=k|\mathbf{x}_i, \boldsymbol{\theta}^{(t)})$ (soft cluster assignments)
- **M-Step:** Update $\boldsymbol{\theta}$ via weighted MLE: $\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}$, etc.

**Variational View:**
- **ELBO:** $\log p(\mathbf{x}|\boldsymbol{\theta}) = \text{ELBO}(q, \boldsymbol{\theta}) + D_{\text{KL}}(q(z|\mathbf{x}) \,\|\, p(z|\mathbf{x}, \boldsymbol{\theta}))$
- **E-Step** = minimize KL → set $q = p(z|\mathbf{x}, \boldsymbol{\theta})$ (tighten bound)
- **M-Step** = maximize Q-function $\mathbb{E}_q[\log p(\mathbf{x}, z|\boldsymbol{\theta})]$ (raise bound)

**Key:** EM converges because log-likelihood is monotonically non-decreasing; K-means is EM with hard assignments

</div>

---

## From GMM to Deep Latent Models

<div style="font-size: 0.7em;">

**GMM worked because:**

| Component | GMM Choice | Why it's tractable |
|:----------|:-----------|:-------------------|
| Latent $z$ | Discrete: $z \in \{1, ..., K\}$ | Sum over $K$ values instead of integral |
| Prior $p(z)$ | Categorical: $\pi_k$ | Simple mixing weights |
| Decoder $p(x\|z)$ | Gaussian: $\mathcal{N}(x\|\mu_k, \Sigma_k)$ | Closed-form posterior |

<div class="fragment appear" data-fragment-index="1">

**What if we want more expressive models?**

- **Continuous latent space:** $\mathbf{z} \in \mathbb{R}^d$ — can represent smooth, continuous factors of variation
- **Neural network decoder:** $p_\theta(\mathbf{x}|\mathbf{z})$ — can model complex, nonlinear relationships

This is the **deep latent variable model** — but what breaks?

</div>

</div>

---

## The Intractable Posterior Problem

<div style="font-size: 0.7em;">

**Recall the E-step goal:** Compute the posterior $p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$

Using Bayes' theorem:

<div class="formula">
$$
p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) = \frac{p_\theta(\mathbf{x}|\mathbf{z}) \cdot p(\mathbf{z})}{p(\mathbf{x}|\boldsymbol{\theta})} = \frac{p_\theta(\mathbf{x}|\mathbf{z}) \cdot p(\mathbf{z})}{\int p_\theta(\mathbf{x}|\mathbf{z}') \cdot p(\mathbf{z}') \, d\mathbf{z}'}
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**The denominator is the problem!**

| Model | Decoder $p(\mathbf{x}\|\mathbf{z})$ | Marginal $p(\mathbf{x})$ | Posterior $p(\mathbf{z}\|\mathbf{x})$ |
|:------|:-----------------------------------|:------------------------|:------------------------------------|
| GMM | Gaussian | Finite sum | **Tractable** |
| Deep LVM | Neural Network | Intractable integral | **Intractable** |

</div>

<div class="fragment appear" data-fragment-index="2">

**Why neural networks break tractability:**

The integral $\int p_\theta(\mathbf{x}|\mathbf{z}) \cdot p(\mathbf{z}) \, d\mathbf{z}$ has no closed form when $p_\theta(\mathbf{x}|\mathbf{z})$ involves nonlinear transformations — we cannot analytically integrate over all possible latent codes!

</div>

</div>

---

## EM Breaks Down

<div style="font-size: 0.7em;">

**Recall the EM framework:**

<div class="formula">
$$
\log p(\mathbf{x}|\boldsymbol{\theta}) = \text{ELBO}(q, \boldsymbol{\theta}) + D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) \right)
$$
</div>

| Step | GMM | Deep Latent Model |
|:-----|:----|:------------------|
| **E-step** | Set $q = p(\mathbf{z}\|\mathbf{x}, \boldsymbol{\theta})$ exactly | **Cannot compute** $p(\mathbf{z}\|\mathbf{x}, \boldsymbol{\theta})$ |
| **M-step** | Closed-form weighted MLE | Gradient descent on NN parameters |
| **Bound** | Tight (KL = 0 after E-step) | **Always a gap** |

<div class="fragment appear" data-fragment-index="1">

**The fundamental problem:**

In GMM, we could set $q(\mathbf{z}|\mathbf{x}) = p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$ exactly, making the ELBO tight.

With neural network decoders, we **cannot compute the true posterior** — so we cannot perform the E-step!

</div>

<div class="fragment appear highlight" data-fragment-index="2">

**We need an approximation strategy...**

</div>

</div>

---

## Traditional Variational Inference

<div style="font-size: 0.7em;">

**Goal:** Approximate the intractable posterior $p(\mathbf{z}|\mathbf{x})$ with a tractable distribution $q(\mathbf{z})$

<div class="fragment appear" data-fragment-index="1">

**For each datapoint $\mathbf{x}_i$:** Optimize separate variational parameters $\boldsymbol{\lambda}_i = (\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i)$

<div class="formula">
$$
q_i(\mathbf{z}) = \mathcal{N}(\mathbf{z} \,|\, \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}^2_i)) \quad \text{maximize } \mathcal{L}(\boldsymbol{\lambda}_i) \text{ until convergence}
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

**Algorithm (Coordinate Ascent VI):**

1. For each datapoint $\mathbf{x}_i$: initialize $\boldsymbol{\lambda}_i$ randomly
2. Iterate until convergence: update each $\lambda_{i,j}$ to maximize ELBO
3. Store optimal $\boldsymbol{\lambda}_i^*$ for $\mathbf{x}_i$

</div>

<div class="fragment appear" data-fragment-index="3">

**Problems:**

- **Slow:** $N$ datapoints = $N$ separate optimization problems
- **No generalization:** New $\mathbf{x}_{\text{new}}$ requires optimization from scratch
- **Memory:** Must store parameters for every datapoint

</div>

</div>

---

## The VAE Solution: Amortized Variational Inference

<div style="font-size: 0.7em;">

**Key insight:** If we can't compute $p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$, let's **learn to approximate it!**

<div class="fragment appear" data-fragment-index="1">

**Introduce an encoder network** $q_\phi(\mathbf{z}|\mathbf{x})$ that approximates the intractable posterior:

<div class="formula">
$$
q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}\left(\mathbf{z} \,|\, \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))\right)
$$
</div>

- $\boldsymbol{\mu}_\phi(\mathbf{x})$: neural network outputting the mean
- $\boldsymbol{\sigma}_\phi(\mathbf{x})$: neural network outputting the standard deviation

</div>

<div class="fragment appear" data-fragment-index="2">

**"Amortized" = share inference cost across all datapoints:**

| Traditional VI | Amortized VI (VAE) |
|:---------------|:-------------------|
| Optimize $q(\mathbf{z})$ separately for each $\mathbf{x}$ | Single encoder $q_\phi(\mathbf{z}\|\mathbf{x})$ for all $\mathbf{x}$ |
| Slow: per-datapoint optimization | Fast: one forward pass |
| Exact for each point | Approximate, but generalizes |

</div>

</div>

---

