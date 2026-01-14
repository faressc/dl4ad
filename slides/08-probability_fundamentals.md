# Probability Fundamentals

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
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1951;" data-timeline-fragments-select="1763:1,1812:1,1815:0,1830:1,1922:1">
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

## Probability Theory in Deep Learning

*Probability theory provides a framework for modeling uncertainty in data, predictions, and model parameters*


**Supervised Learning:**

<div style="font-size: 0.85em;">

- Loss functions such as cross-entropy and mean squared error are derived from probabilistic principles
- Softmax outputs represent conditional class probabilities

</div>

**Unsupervised Learning:**

<div style="font-size: 0.85em;">

- Variational Autoencoders (VAEs) learn latent representations to approximate the probability distributions of input data
- Generative Adversarial Networks (GANs) implicitly model the data distribution through adversarial training
- Diffusion models learn to reverse a gradual noising process to generate samples from the data distribution

</div>

---

## Key Probability Concepts

- **Experiment**: A procedure that yields one of several possible outcomes
- **Sample Space** ($\Omega$): The set of all possible elementary outcomes of an experiment
- **Outcome** ($\omega \in \Omega$): A single possible result of an experiment
- **$\sigma$-Algebra** ($\mathcal{F}$): A collection of subsets of $\Omega$ that contains $\Omega$ and is closed under complementation and countable unions
- **Event** ($A \in \mathcal{F}$): A measurable subset of the sample space (an element of the σ-algebra)
- **Probability Measure** ($P: \mathcal{F} \to [0,1]$): A function assigning probabilities to events

---

## Probability Axioms

1. **Non-negativity**: For any event $A \in \mathcal{F}$, $P(A) \geq 0$
2. **Normalization**: $P(\Omega) = 1$
3. **Additivity**: For any countable sequence of mutually exclusive/disjoint events $A_1, A_2, \ldots \in \mathcal{F}$:

<div class="formula">
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$
</div>

with $\bigcup_{i=1}^{n} = A_1 \cup A_2 \cup \ldots \cup A_n$ and $A_i \cap A_j = \emptyset$ for $i \neq j$.

---

## Random Variables

A **random variable** $X: \Omega \to \mathbb{R}$ is a measurable function that assigns a numerical value to each outcome

<div style="display: flex; gap: 2em;">
<div style="flex: 1;">

**Discrete Random Variable**

Takes on a finite or countably infinite set of values

*Examples:*
- Number of heads in 10 coin flips
- Class label in classification (0, 1, 2, ...)
- Word token index in a vocabulary

</div>
<div style="flex: 1;">

**Continuous Random Variable**

Takes on any value within an interval or the entire real line

*Examples:*
- Audio sample amplitude
- Pixel intensity (0.0 to 1.0)
- Latent vector components in VAEs

</div>
</div>

---

## Probability Mass Function (PMF)

For a **discrete** random variable $X$, the PMF $p_X: \mathbb{R} \to [0,1]$ gives the probability of each value:

$$p_X(x) = P(\omega \in \Omega : X(\omega) = x)$$

**Properties:**

1. $p_X(x) \geq 0$ for all $x$
2. $\sum_{x} p_X(x) = 1$ (sums over all possible values)
3. $P(X \in A) = \sum_{x \in A} p_X(x)$

<div class="fragment image-overlay">

**Example:** Fair die roll

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|
| $p_X(x)$ | $\frac{1}{6}$ | $\frac{1}{6}$ | $\frac{1}{6}$ | $\frac{1}{6}$ | $\frac{1}{6}$ | $\frac{1}{6}$ |
</div>

---

## Probability Density Function (PDF)

For a **continuous** random variable $X$, the PDF $f_X: \mathbb{R} \to [0, \infty)$ describes the density of probability:

$$P(\{\omega \in \Omega : a \leq X(\omega) \leq b\}) = \int_a^b f_X(x) \, dx$$

**Properties:**

1. $f_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$
3. $P(\{\omega \in \Omega : X(\omega) = x\}) = 0$ for any specific value (probability is in intervals, not points)

<div class="fragment image-overlay">

**Example:** Standard Normal Distribution

$$f_X(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

</div>

---

## PMF vs PDF

<div style="font-size: 0.85em;">

| Property | PMF (Discrete) | PDF (Continuous) |
|----------|----------------|------------------|
| Notation | $p_X(x)$ | $f_X(x)$ |
| Value meaning | Actual probability | Probability density |
| Summing/Integrating | $\sum_x p_X(x) = 1$ | $\int_{-\infty}^{\infty} f_X(x) dx = 1$ |
| Point probability | $P(X=x) = p_X(x)$ | $P(X=x) = 0$ |
| Interval probability | $\sum_{x \in [a,b]} p_X(x)$ | $\int_a^b f_X(x) dx$ |
| Range of values | $[0, 1]$ | $[0, \infty)$ |

</div>

**Key insight:** For continuous variables, the PDF can exceed 1 (it's a density, not a probability), but the integral over any region is always $\leq 1$.


---

