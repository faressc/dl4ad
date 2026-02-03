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

<ul>
<li><strong>E-Step:</strong> Compute responsibilities $\gamma_{ik} = p(z_i=k|\mathbf{x}_i, \boldsymbol{\theta}^{(t)})$ (soft cluster assignments)</li>
<li><strong>M-Step:</strong> Update $\boldsymbol{\theta}$ via weighted MLE: $\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}$, etc.</li>
</ul>

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
| Decoder $p(\mathbf{x}\|z)$ | Gaussian: $\mathcal{N}(\mathbf{x}\|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Closed-form posterior |

<div class="fragment appear" data-fragment-index="1">

**What if we want more expressive models?**

<ul>
<li><strong>Continuous latent space:</strong> $\mathbf{z} \in \mathbb{R}^d$ — can represent smooth, continuous factors of variation</li>
<li><strong>Neural network decoder:</strong> $p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}), \sigma^2 \mathbf{I})$ — mean is neural network output, variance is fixed</li>
</ul>

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
p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) = \frac{p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \cdot p(\mathbf{z})}{p(\mathbf{x}|\boldsymbol{\theta})} = \frac{p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \cdot p(\mathbf{z})}{\int p(\mathbf{x}|\mathbf{z}', \boldsymbol{\theta}) \cdot p(\mathbf{z}') \, d\mathbf{z}'}
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

<div>
GMM has fixed $\boldsymbol{\mu}_k$ and discrete $z$ (finite sum). The deep latent variable model has $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}) = \text{NeuralNet}(\mathbf{z})$ — a complex function over continuous latent space. The marginal $\int p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \cdot p(\mathbf{z}) \, d\mathbf{z}$ has no closed form!
</div>

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

</div>

<div class="fragment appear highlight image-overlay" data-fragment-index="2">

**We need an approximation strategy...**

</div>

---

## Learn the Posterior Approximation

<div style="font-size: 0.7em;">

**Key insight:** If we can't compute $p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta})$, let's **learn to approximate it!**

<div class="fragment appear" data-fragment-index="1">

**Introduce an encoder network** $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ that approximates the intractable posterior:

<div class="formula">
$$
q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \mathcal{N}\left(\mathbf{z} \,|\, \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_{\boldsymbol{\phi}}(\mathbf{x}))\right)
$$
</div>

- $\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x})$: neural network outputting the mean
- $\boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x})$: neural network outputting the standard deviation

</div>

<div class="fragment appear" data-fragment-index="2">

**Why this works:**

- A single encoder handles **all datapoints** — one forward pass per $\mathbf{x}$
- The encoder learns to map $\mathbf{x} \mapsto (\boldsymbol{\mu}, \boldsymbol{\sigma})$ that approximate the true posterior
- Generalizes to unseen data (unlike per-datapoint optimization)

</div>

<div class="fragment appear" data-fragment-index="3">

**This is called "amortized inference"** — the cost of learning the posterior is amortized across the entire dataset by sharing encoder parameters $\boldsymbol{\phi}$.

</div>

</div>

---

## VAE vs GMM: The Setup

<div style="font-size: 0.65em;">

**Recall the ELBO decomposition** (same as GMM!):

<div class="formula">
$$
\log p(\mathbf{x}|\boldsymbol{\theta}) = \text{ELBO}(q, \boldsymbol{\theta}) + D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z}|\mathbf{x}, \boldsymbol{\theta}) \right)
$$
</div>

<div class="fragment appear" data-fragment-index="1">

| Component | GMM | VAE |
|:----------|:----|:----|
| **Latent** $\mathbf{z}$ | Discrete: $z \in \{1, ..., K\}$ | Continuous: $\mathbf{z} \in \mathbb{R}^d$ |
| **Prior** $p(\mathbf{z})$ | Categorical: $\pi_k$ | Standard Gaussian: $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| **Decoder** $p(\mathbf{x}\|\mathbf{z}, \boldsymbol{\theta})$ | Gaussian: $\mathcal{N}(\mathbf{x}\|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Neural network with Gaussian output |
| **Posterior approx.** $q$ | Exact: $q = p(z\|\mathbf{x}, \boldsymbol{\theta})$ | Learned encoder: $q(\mathbf{z}\|\mathbf{x}, \boldsymbol{\phi})$ |
| **Parameters** | $\boldsymbol{\theta} = \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k\}$ | $\boldsymbol{\theta}$ (decoder NN), $\boldsymbol{\phi}$ (encoder NN) |

</div>

<div class="fragment appear" data-fragment-index="2">

**Key difference:** In VAE, we optimize **both** $\boldsymbol{\theta}$ and $\boldsymbol{\phi}$ jointly, since we cannot compute the true posterior!


</div>

</div>

---

## Recap: Deriving the ELBO

<div style="font-size: 0.65em;">

**The fundamental challenge:** We want to maximize $\log p(\mathbf{x}|\boldsymbol{\theta})$, but the log-of-sum is intractable.

We introduce a variational distribution $q(z|\mathbf{x})$ and use Jensen's inequality:

<div class="formula">
  $$
\begin{aligned}
\log p_{X|\Theta}(\mathbf{x}|\boldsymbol{\theta}) &= \log \left( \sum_{k=1}^K p(\mathbf{x}, z=k|\boldsymbol{\theta}) \right) \\
&= \log \left( \sum_{k=1}^K q(z=k|\mathbf{x}) \cdot \frac{p(\mathbf{x}, z=k|\boldsymbol{\theta})}{q(z=k|\mathbf{x})} \right) \\
&= \log \left( \mathbb{E}_{z \sim q(z|\mathbf{x})} \left[ \frac{p(\mathbf{x}, z|\boldsymbol{\theta})}{q(z|\mathbf{x})} \right] \right) \\
&\geq \mathbb{E}_{z \sim q(z|\mathbf{x})} \left[ \log \frac{p(\mathbf{x}, z|\boldsymbol{\theta})}{q(z|\mathbf{x})} \right] = \text{ELBO}(q, \boldsymbol{\theta})
\end{aligned}
  $$
</div>

</div>

---

## The VAE ELBO

<div style="font-size: 0.65em;">

Starting from the general ELBO definition:

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log \frac{p(\mathbf{x}, \mathbf{z} | \boldsymbol{\theta})}{q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \right]
$$
</div>

<div class="fragment appear" data-fragment-index="1">

Using the chain rule $p(\mathbf{x}, \mathbf{z} | \boldsymbol{\theta}) = p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \cdot p(\mathbf{z})$:

<div class="formula">
$$
\begin{aligned}
\text{ELBO} &= \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) + \log p(\mathbf{z}) - \log q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \right] \\
&= \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right] + \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log \frac{p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \right]
\end{aligned}
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

Recognizing the KL divergence, we get the **VAE objective**:

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \underbrace{\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right)}_{\text{Regularization term}}
$$
</div>

</div>

</div>

---

## Understanding the VAE Objective

<div style="font-size: 0.7em;">

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \underbrace{\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right)}_{\text{Regularization}}
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Reconstruction term:** How well can the decoder reconstruct $\mathbf{x}$ from samples $\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$?

- Encourages the latent code to **preserve information** about $\mathbf{x}$
- Like the expected complete-data log-likelihood in EM's M-step

</div>

<div class="fragment appear" data-fragment-index="2">

**Regularization term:** How close is the encoder's output to the prior?

- Encourages the latent space to be **well-structured** (match $\mathcal{N}(\mathbf{0}, \mathbf{I})$)
- Prevents the encoder from "cheating" by encoding each $\mathbf{x}$ as a delta function
- No direct analogue in GMM — posterior is exact there!

</div>

<div class="fragment appear" data-fragment-index="3">

**Trade-off:** Reconstruction wants $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ to be specific to each $\mathbf{x}$; regularization wants $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ close to the prior. The VAE balances these!

</div>

</div>

---

## Comparison: GMM Q-Function vs VAE ELBO

<div style="font-size: 0.65em;">

**GMM (E-step sets $q = p(z|\mathbf{x}, \boldsymbol{\theta})$ exactly):**

<div class="formula">
$$
Q(\boldsymbol{\theta}; \boldsymbol{\theta}^{(t)}) = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} \log p(\mathbf{x}_i, z_i=k | \boldsymbol{\theta}) = \sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{ik} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**VAE (optimize $q$ and $\boldsymbol{\theta}$ jointly):**

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}) = \sum_{i=1}^{n} \left[ \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}_i, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}_i|\mathbf{z}, \boldsymbol{\theta}) \right] - D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}_i, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right) \right]
$$
</div>

</div>

</div>

---

## Monte Carlo Estimation

<div style="font-size: 0.7em;">

**Problem:** How do we compute expectations when integrals have no closed form?

<div class="formula">
$$
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] = \int p(\mathbf{x}) f(\mathbf{x}) \, d\mathbf{x} \quad \text{(often intractable)}
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Monte Carlo estimation:** Approximate the expectation using samples!

<div class="formula">
$$
\mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] \approx \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{x}^{(l)}), \quad \text{where } \mathbf{x}^{(l)} \sim p(\mathbf{x})
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

**Why this works:** By the Law of Large Numbers, the sample mean converges to the true expectation:

<div class="formula">
$$
\frac{1}{L} \sum_{l=1}^{L} f(\mathbf{x}^{(l)}) \xrightarrow{L \to \infty} \mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})]
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="3">

**Key properties:**
- **Unbiased:** $\mathbb{E}\left[\frac{1}{L}\sum_l f(\mathbf{x}^{(l)})\right] = \mathbb{E}_{p}[f(\mathbf{x})]$
- **Variance:** $\text{Var} \propto \frac{1}{L}$ — more samples = lower variance
- **Works for any** $f$ as long as we can sample from $p(\mathbf{x})$

</div>

</div>

---

## The Optimization Challenge

<div style="font-size: 0.7em;">

**Goal:** Maximize the ELBO with respect to both $\boldsymbol{\theta}$ and $\boldsymbol{\phi}$

<div class="formula">
$$
\boldsymbol{\phi}^*, \boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\phi}, \boldsymbol{\theta}} \sum_{i=1}^{n} \text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}_i)
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Problem: The reconstruction term involves an expectation**

<div class="formula">
$$
\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right] = \int q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \, d\mathbf{z}
$$
</div>

This integral has no closed form when $p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta})$ is a neural network!

</div>

<div class="fragment appear" data-fragment-index="2">

**Solution:** Monte Carlo estimation — sample $\mathbf{z}^{(l)} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$:

<div class="formula">
$$
\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right] \approx \frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta})
$$
</div>

In practice, $L = 1$ works well during training!

</div>

</div>

---

## Gradient w.r.t. Decoder Parameters $\boldsymbol{\theta}$

<div style="font-size: 0.7em;">

**Good news:** The gradient w.r.t. $\boldsymbol{\theta}$ is straightforward!

<div class="formula">
$$
\nabla_{\boldsymbol{\theta}} \frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta}) = \frac{1}{L} \sum_{l=1}^{L} \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta})
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Why is this easy?**

- The samples $\mathbf{z}^{(l)}$ come from the **encoder** (parameters $\boldsymbol{\phi}$)
- From the decoder's perspective, $\mathbf{z}^{(l)}$ is just a **fixed input** — like any other input to a neural network
- No sampling w.r.t. $\boldsymbol{\theta}$ means standard backpropagation works!

</div>

<div class="fragment appear" data-fragment-index="2">

**This is just like training any neural network:**

$$\mathbf{z}^{(l)} \xrightarrow{\text{Decoder}_{\boldsymbol{\theta}}} \hat{\mathbf{x}} \xrightarrow{\text{loss}} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta})$$

Backprop through the decoder as usual!

</div>

</div>

---

## Gradient w.r.t. Encoder Parameters $\boldsymbol{\phi}$

<div style="font-size: 0.7em;">

**Problem:** We need gradients w.r.t. $\boldsymbol{\phi}$, but we sample from $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$!

<div class="formula">
$$
\nabla_{\boldsymbol{\phi}} \frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta}), \quad \text{where } \mathbf{z}^{(l)} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**The issue:** The samples $\mathbf{z}^{(l)}$ depend on $\boldsymbol{\phi}$ through stochastic sampling!

- Sampling $\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})$ is a **stochastic operation**
- Gradients don't flow through random sampling!
- We cannot backpropagate through the sampling step

</div>

<div class="fragment appear" data-fragment-index="2">

**Compare the two gradients:**

| Parameter | Gradient | Difficulty |
|:----------|:---------|:-----------|
| $\boldsymbol{\theta}$ (decoder) | $\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}\|\mathbf{z}, \boldsymbol{\theta})$ | Standard backprop — $\mathbf{z}$ is just an input |
| $\boldsymbol{\phi}$ (encoder) | $\nabla_{\boldsymbol{\phi}} \log p(\mathbf{x}\|\mathbf{z}, \boldsymbol{\theta})$ | **Problematic** — $\mathbf{z}$ depends on $\boldsymbol{\phi}$ via sampling |

</div>

</div>

---

## The Reparameterization Trick

<div style="font-size: 0.7em;">

**Key insight:** Rewrite the sampling process to separate stochasticity from parameters!

<div class="fragment appear" data-fragment-index="1">

**Before (non-differentiable):**

<div class="formula">
$$\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_{\boldsymbol{\phi}}(\mathbf{x})))$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

**After (differentiable):**

<div class="formula">
$$
\mathbf{z} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
</div>

<ul>
<li>$\boldsymbol{\epsilon}$ is sampled from a <strong>fixed</strong> distribution (independent of $\boldsymbol{\phi}$)</li>
<li>$\mathbf{z}$ is now a <strong>deterministic function</strong> of $\boldsymbol{\phi}$ (given $\boldsymbol{\epsilon}$)</li>
<li>Gradients flow through $\boldsymbol{\mu}_{\boldsymbol{\phi}}$ and $\boldsymbol{\sigma}_{\boldsymbol{\phi}}$ via standard backpropagation!</li>
</ul>

</div>

<div class="fragment appear image-overlay" style="width: 80%" data-fragment-index="3">

**The expectation becomes:**

<div class="formula">
$$
\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ f(\mathbf{z}) \right] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ f(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\epsilon}) \right]
$$
</div>

<div>
Now $\nabla_{\boldsymbol{\phi}}$ can go inside the expectation!
</div>


</div>

</div>

---

## Reparameterization: The Math

<div style="font-size: 0.65em;">

**With reparameterization, we can compute gradients of the MC estimate:**

<div class="formula">
$$
\nabla_{\boldsymbol{\phi}} \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}^{(l)}) = \frac{1}{L} \sum_{l=1}^{L} \nabla_{\boldsymbol{\phi}} f(\boldsymbol{\mu}_{\boldsymbol{\phi}} + \boldsymbol{\sigma}_{\boldsymbol{\phi}} \odot \boldsymbol{\epsilon}^{(l)})
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Applying the chain rule:**

<div class="formula">
$$
\nabla_{\boldsymbol{\phi}} f(\mathbf{z}) = \nabla_\mathbf{z} f(\mathbf{z}) \cdot \nabla_{\boldsymbol{\phi}} \mathbf{z} = \nabla_\mathbf{z} f(\mathbf{z}) \cdot \left( \nabla_{\boldsymbol{\phi}} \boldsymbol{\mu}_{\boldsymbol{\phi}} + \boldsymbol{\epsilon} \odot \nabla_{\boldsymbol{\phi}} \boldsymbol{\sigma}_{\boldsymbol{\phi}} \right)
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

**In practice (with $L$ samples):**

<div class="formula">
$$
\nabla_{\boldsymbol{\phi}} \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}^{(l)}) = \frac{1}{L} \sum_{l=1}^{L} \nabla_\mathbf{z} f(\mathbf{z}^{(l)}) \cdot \left( \nabla_{\boldsymbol{\phi}} \boldsymbol{\mu}_{\boldsymbol{\phi}} + \boldsymbol{\epsilon}^{(l)} \odot \nabla_{\boldsymbol{\phi}} \boldsymbol{\sigma}_{\boldsymbol{\phi}} \right)
$$
</div>

</div>

</div>

---

## Applying to the VAE Reconstruction Term

<div style="font-size: 0.65em;">

**Now let's substitute** $f(\mathbf{z}) = \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta})$ — the decoder log-likelihood:

<div class="formula">
$$
\nabla_{\boldsymbol{\phi}} \frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta}) = \frac{1}{L} \sum_{l=1}^{L} \nabla_{\boldsymbol{\phi}} \log p(\mathbf{x}|\boldsymbol{\mu}_{\boldsymbol{\phi}} + \boldsymbol{\sigma}_{\boldsymbol{\phi}} \odot \boldsymbol{\epsilon}^{(l)}, \boldsymbol{\theta})
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Expanding with the chain rule:**

<div class="formula">
$$
= \frac{1}{L} \sum_{l=1}^{L} \underbrace{\nabla_\mathbf{z} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta})}_{\text{decoder gradient}} \cdot \left( \nabla_{\boldsymbol{\phi}} \boldsymbol{\mu}_{\boldsymbol{\phi}} + \boldsymbol{\epsilon}^{(l)} \odot \nabla_{\boldsymbol{\phi}} \boldsymbol{\sigma}_{\boldsymbol{\phi}} \right)
$$
</div>

</div>

<div class="fragment appear" data-fragment-index="2">

**Key insight:** The gradient flows from decoder → through $\mathbf{z}$ → to encoder parameters $\boldsymbol{\phi}$

- $\nabla_\mathbf{z} \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta})$: how changing $\mathbf{z}$ affects reconstruction
- $\nabla_{\boldsymbol{\phi}} \boldsymbol{\mu}_{\boldsymbol{\phi}}$: how encoder parameters affect the mean
- $\boldsymbol{\epsilon}^{(l)} \odot \nabla_{\boldsymbol{\phi}} \boldsymbol{\sigma}_{\boldsymbol{\phi}}$: how encoder parameters affect variance (scaled by noise)

</div>

<div class="fragment appear" data-fragment-index="3">

**In practice ($L=1$):** Sample one $\boldsymbol{\epsilon}$, compute $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, backprop through decoder and encoder!

</div>

</div>

---

## Recap: The Full VAE Objective

<div style="font-size: 0.7em;">

**We're optimizing the ELBO:**

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \underbrace{\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right)}_{\text{Regularization term}}
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**What we've solved — the reconstruction term:**

| Challenge | Solution |
|:----------|:---------|
| Intractable expectation | Monte Carlo: $\frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}\|\mathbf{z}^{(l)}, \boldsymbol{\theta})$ |
| Gradient w.r.t. $\boldsymbol{\theta}$ | Standard backprop (z is just an input) |
| Gradient w.r.t. $\boldsymbol{\phi}$ | Reparameterization trick |

</div>

<div class="fragment appear" data-fragment-index="2">

**What's left — the KL term:**

How do we compute $D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right)$?

</div>

</div>

---

## The KL Term: Closed Form

<div style="font-size: 0.65em;">

**Good news:** The KL divergence between two Gaussians has a closed form!

For $q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

<div class="fragment appear" data-fragment-index="1">

<div class="formula">
$$
\begin{aligned}
D_{\text{KL}}(q \| p) &= \mathbb{E}_q[\log q(\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{z})] \\[0.5em]
&= \mathbb{E}_q\left[-\frac{1}{2}\sum_{j=1}^d \left(\log(2\pi\sigma_j^2) + \frac{(z_j - \mu_j)^2}{\sigma_j^2}\right)\right] - \mathbb{E}_q\left[-\frac{1}{2}\sum_{j=1}^d \left(\log(2\pi) + z_j^2\right)\right] \\[0.5em]
&= -\frac{1}{2}\sum_j \left(\log \sigma_j^2 + 1\right) + \frac{1}{2}\sum_j \mathbb{E}_q[z_j^2] \\[0.5em]
&= -\frac{1}{2}\sum_j \left(\log \sigma_j^2 + 1\right) + \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2\right) \\[0.5em]
&= \frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)
\end{aligned}
$$
</div>

where $j \in \{1, \ldots, d\}$ indexes each dimension of the latent vector $\mathbf{z} \in \mathbb{R}^d$.

</div>

<div class="fragment appear" data-fragment-index="2">

**No Monte Carlo needed for this term!** Gradients w.r.t. $\boldsymbol{\phi}$ are straightforward.

</div>

</div>

---

## The Complete VAE Loss

<div style="font-size: 0.65em;">

**Putting it all together:** For a single datapoint $\mathbf{x}$:

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \underbrace{\frac{1}{L} \sum_{l=1}^{L} \log p(\mathbf{x}|\mathbf{z}^{(l)}, \boldsymbol{\theta})}_{\text{Monte Carlo estimate}} - \underbrace{\frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)}_{\text{Closed-form KL}}
$$
</div>

<div>
where $\mathbf{z}^{(l)} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\epsilon}^{(l)}$, $\boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
</div>

<div class="fragment appear" data-fragment-index="1">

**But what is** $\log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta})$**?** We need to specify the decoder's output distribution!

**Common choice:** Gaussian with fixed variance $\sigma^2$

<div class="formula">
$$
p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) = \mathcal{N}(\mathbf{x} \,|\, \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}), \sigma^2 \mathbf{I})
$$
</div>

The neural network $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})$ outputs the **mean** of this Gaussian — the reconstructed $\hat{\mathbf{x}}$.

</div>

</div>

---

## Decoder Likelihood: From Gaussian to MSE

<div style="font-size: 0.65em;">

**Decoder output distribution:** $p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}), \sigma^2 \mathbf{I})$

<div class="fragment appear" data-fragment-index="1">

**Taking the log of the Gaussian PDF:**

<div class="formula">
$$
\begin{aligned}
\log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) &= \log \left( \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left( -\frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})\|^2 \right) \right) \\[0.5em]
&= -\frac{D}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})\|^2 \\[0.5em]
&= -\frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})\|^2 + \text{const}
\end{aligned}
$$
</div>

where $D$ is the data dimensionality (e.g., number of pixels).

</div>

<div class="fragment appear" data-fragment-index="2">

**Key insight:** Since $\sigma^2$ is a fixed constant:
<div class="formula">
$$\max_{\boldsymbol{\theta}} \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \quad \Longleftrightarrow \quad \min_{\boldsymbol{\theta}} \|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})\|^2$$
</div>

Maximizing Gaussian log-likelihood is equivalent to minimizing **mean squared error (MSE)**!

</div>

</div>

---

## The Practical VAE Loss

<div style="font-size: 0.65em;">

**Substituting the Gaussian decoder into the ELBO:**

<div class="formula">
$$
\text{ELBO} \propto -\frac{1}{2\sigma^2} \|\mathbf{x} - \hat{\mathbf{x}}\|^2 - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \| p(\mathbf{z}))
$$
</div>

<div class="fragment appear" data-fragment-index="1">

**Converting to a loss (negate and drop constants):**

<div class="formula">
$$
\mathcal{L}_{\text{VAE}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{Reconstruction loss (MSE)}} + \underbrace{\beta \cdot D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \| p(\mathbf{z}))}_{\text{KL regularization}}
$$
</div>

where $\hat{\mathbf{x}} = \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z})$ is the decoder output.

</div>

<div class="fragment appear" data-fragment-index="2" style="font-size: 0.8em;">

| $\beta$ | Effect |
| :--------- | :-------- |
| $\beta = 1$ | Standard VAE (original formulation) |
| $\beta < 1$ | Better reconstructions, less regularized latent space |
| $\beta > 1$ | **$\beta$-VAE**: stronger regularization, more disentangled latents |

</div>

<div class="fragment appear" data-fragment-index="3">

**Note:** The relationship $\beta = 2\sigma^2$ shows that $\beta$ implicitly controls the assumed decoder variance — larger $\beta$ corresponds to assuming a noisier decoder!

</div>

</div>

---

## VAE Training Algorithm

<div style="font-size: 0.65em; height: 1000px;">

```
Initialize: encoder parameters φ, decoder parameters θ

For each epoch:
    For each minibatch {x₁, ..., xₘ}:
        
        # Forward pass (encoder)
        For each xᵢ:
            (μᵢ, σᵢ) = Encoder_φ(xᵢ)
        
        # Reparameterization (sample latent codes)
        For each xᵢ:
            εᵢ ~ N(0, I)
            zᵢ = μᵢ + σᵢ ⊙ εᵢ
        
        # Forward pass (decoder)
        For each zᵢ:
            x̂ᵢ = Decoder_θ(zᵢ)
        
        # Compute loss
        L_recon = (1/m) Σᵢ ||xᵢ - x̂ᵢ||²
        L_KL = (1/m) Σᵢ Σⱼ (σᵢⱼ² + μᵢⱼ² - 1 - log σᵢⱼ²) / 2
        L = L_recon + β · L_KL
        
        # Backward pass & update
        Compute ∇_θ L, ∇_φ L via backpropagation
        Update θ, φ using optimizer (e.g., Adam)
```

</div>

---

## GMM vs VAE: Optimization Comparison

<div style="font-size: 0.6em;">

<table>
<thead>
<tr>
<th align="left">Aspect</th>
<th align="left">GMM (EM)</th>
<th align="left">VAE</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>E-step / Encoder</strong></td>
<td align="left">Compute $\gamma_{ik} = p(z_i=k|\mathbf{x}_i, \boldsymbol{\theta})$ exactly</td>
<td align="left">Forward pass: $(\boldsymbol{\mu}, \boldsymbol{\sigma}) = \text{Encoder}_{\boldsymbol{\phi}}(\mathbf{x})$</td>
</tr>
<tr>
<td align="left"><strong>Posterior</strong></td>
<td align="left">Exact (tractable)</td>
<td align="left">Approximate (learned)</td>
</tr>
<tr>
<td align="left"><strong>Sampling</strong></td>
<td align="left">Weighted sum over $K$ components</td>
<td align="left">Monte Carlo: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$</td>
</tr>
<tr>
<td align="left"><strong>M-step / Decoder</strong></td>
<td align="left">Closed-form: $\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}$</td>
<td align="left">Gradient descent on NN</td>
</tr>
<tr>
<td align="left"><strong>Joint optimization</strong></td>
<td align="left">Alternating (E then M)</td>
<td align="left">Simultaneous (SGD on $\boldsymbol{\theta}, \boldsymbol{\phi}$)</td>
</tr>
<tr>
<td align="left"><strong>Convergence</strong></td>
<td align="left">Monotonic increase in likelihood</td>
<td align="left">ELBO increases (with noise from SGD)</td>
</tr>
<tr>
<td align="left"><strong>KL gap</strong></td>
<td align="left">Zero (ELBO is tight)</td>
<td align="left">Non-zero (approximation gap)</td>
</tr>
</tbody>
</table>

<div class="fragment appear" data-fragment-index="1">

**Key insight:** VAE trades exactness for expressiveness:

- GMM: Exact inference, limited model (Gaussian components)
- VAE: Approximate inference, powerful model (neural networks)

</div>

</div>

---

## Summary: Optimizing the VAE

<div style="font-size: 0.65em;">

**The VAE objective** (maximize ELBO):

<div class="formula">
$$
\text{ELBO}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})} \left[ \log p(\mathbf{x}|\mathbf{z}, \boldsymbol{\theta}) \right] - D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) \,\|\, p(\mathbf{z}) \right)
$$
</div>

**Three key ingredients:**

| Challenge | Solution |
|:----------|:---------|
| Intractable posterior $p(\mathbf{z}\|\mathbf{x})$ | Learn encoder $q(\mathbf{z}\|\mathbf{x}, \boldsymbol{\phi})$ |
| Intractable expectation | Monte Carlo sampling ($L=1$ suffices) |
| Non-differentiable sampling | Reparameterization trick |

</div>

---

# Questions?
