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

## Event Relationships

<div style="font-size: 0.85em;">

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Condition</th>
      <th>Formula</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Disjoint/Mutually Exclusive</td>
      <td>Cannot both occur</td>
      <td>$A \cap B = \emptyset$</td>
      <td>$P(A \cap B) = 0$</td>
    </tr>
    <tr>
      <td>Overlapping</td>
      <td>Can both occur</td>
      <td>$A \cap B \neq \emptyset$</td>
      <td>$P(A \cap B) > 0$</td>
    </tr>
    <tr>
      <td>Independent</td>
      <td>Don't affect each other</td>
      <td>$P(A \cap B) = P(A)P(B)$</td>
      <td>$P(A|B) = P(A)$</td>
    </tr>
    <tr>
      <td>Dependent</td>
      <td>Affect each other</td>
      <td>$P(A \cap B) \neq P(A)P(B)$</td>
      <td>$P(A|B) \neq P(A)$</td>
    </tr>
    <tr>
      <td>Conditional Independence</td>
      <td>Independent given C</td>
      <td>$P(A \cap B|C) = P(A|C)P(B|C)$</td>
      <td>$P(A|B,C) = P(A|C)$</td>
    </tr>
  </tbody>
</table>

</div>

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

<div class="formula">

$$p_X(x) = P(\omega \in \Omega : X(\omega) = x)$$

</div>

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

For a **continuous** random variable $X$, the PDF $p_X: \mathbb{R} \to [0, \infty)$ describes the density of probability:

<div class="formula">

$$P(\{\omega \in \Omega : a \leq X(\omega) \leq b\}) = \int_a^b p_X(x) \, dx$$

</div>

**Properties:**

1. $p_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} p_X(x) \, dx = 1$
3. $P(\{\omega \in \Omega : X(\omega) = x\}) = 0$ for any specific value (probability is in intervals, not points)

<div class="fragment image-overlay">

**Example:** Standard Normal Distribution

<div class="formula">

$$p_X(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

</div>

</div>

---

## PMF vs PDF

<div style="font-size: 0.85em;">

| Property | PMF (Discrete) | PDF (Continuous) |
|----------|----------------|------------------|
| Notation | $p_X(x)$ | $p_X(x)$ |
| Value meaning | Actual probability | Probability density |
| Summing/Integrating | $\sum_x p_X(x) = 1$ | $\int_{-\infty}^{\infty} p_X(x) dx = 1$ |
| Point probability | $P(X=x) = p_X(x)$ | $P(X=x) = 0$ |
| Interval probability | $\sum_{x \in [a,b]} p_X(x)$ | $\int_a^b p_X(x) dx$ |
| Range of values | $[0, 1]$ | $[0, \infty)$ |

</div>

**Key insight:** For continuous variables, the PDF can exceed 1 (it's a density, not a probability), but the integral over any region is always $\leq 1$.

---

<!-- ## Cumulative Distribution Function (CDF)

The **Cumulative Distribution Function** (CDF) $F_X: \mathbb{R} \to [0,1]$ gives the probability that a random variable $X$ takes on a value less than or equal to $x$:

<div class="formula">

$$F_X(x) = P(X \leq x)$$

</div>

**For Discrete Random Variables:**

<div class="formula">

$$F_X(x) = \sum_{t \leq x} p_X(t)$$

</div>

**For Continuous Random Variables:**

<div class="formula">

$$F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$$

</div>

**Properties:**

1. Non-decreasing: If $a < b$, then $F_X(a) \leq F_X(b)$
2. Limits: $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$
3. Right-continuous: $F_X(x) = \lim_{t \to x^+} F_X(t)$
4. Relationship to PMF/PDF:
   - Discrete: $p_X(x) = F_X(x) - F_X(x^-)$
   - Continuous: $f_X(x) = \frac{d}{dx} F_X(x)$  -->

## Types of Distributions

<div style="display: flex; gap: 2em;">

<div style="flex: 1;">

**Discrete Distributions**

- **Bernoulli Distribution**: Models a single binary outcome (success/failure)
- **Binomial Distribution**: Models the number of successes in a fixed number of independent Bernoulli trials
- **Poisson Distribution**: Models the number of events occurring in a fixed interval of time/space

</div>

<div style="flex: 1;">

**Continuous Distributions**

- **Normal Distribution**: Models a continuous variable with a bell-shaped curve
- **Exponential Distribution**: Models the time between events in a Poisson process
- **Uniform Distribution**: Models a variable with equal probability over an interval

</div>

---

## Joint Distributions

For continuous multiple random variables $X$ and $Y$, the **joint distribution** describes the probability of their combined outcomes.

<div class="formula">
$$
\begin{aligned}
P(a \leq X \leq b, c \leq Y \leq d) &= \int_{a}^{b} \int_{c}^{d} p_{X,Y}(x,y) \, dy \, dx \\
P(A) &= \iint_{(x,y) \in A} p_{X,Y}(x,y) \, dy \, dx, \quad A \subseteq \mathbb{R}^2 \\
\text{where } p_{X,Y}(x,y) &\geq 0 \\
\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p_{X,Y}(x,y) \, dy \, dx &= 1
\end{aligned}
$$
</div>

For $n$ random variables, the joint PDF is $p_{X_1, X_2, \ldots, X_n}(x_1, x_2, \ldots, x_n)$.

---

## Marginal Distributions

The **marginal distribution** of a random variable is obtained by integrating the joint distribution over the other variable(s).

<div class="formula">
$$
\begin{aligned}
p_X(x) &= \int_{-\infty}^{\infty} p_{X,Y}(x,y) \, dy \\
p_Y(y) &= \int_{-\infty}^{\infty} p_{X,Y}(x,y) \, dx
\end{aligned}
$$
</div>

This process is called **marginalization**. It is also applicable higher dimensions: for $n$ variables, integrate over $n-1$ variables to get the marginal of one variable.

---

## Conditional Distributions

The **conditional distribution** of a random variable given another is derived from the joint distribution.

<div class="formula">
$$
\begin{aligned}
p_{X|Y}(x|y) &= \frac{p_{X,Y}(x,y)}{p_Y(y)} \quad \text{or} \quad p_{Y|X}(y|x) &= \frac{p_{X,Y}(x,y)}{p_X(x)}
\end{aligned}
$$
</div>

<div style="font-size: 0.85em;">

**Interpretation of $p_{X|Y}(x|y)$:**

<div style="display: flex; gap: 2em;">
<div style="flex: 1;">

**As a function of $x$ (with $y$ fixed)**

This is the conditional probability distribution of $X$ given $Y=y$. It describes the probability distribution over $X$ when we know $Y=y$.

Must integrate/sum to 1: $\int p_{X|Y}(x|y) \, dx = 1$

</div>
<div style="flex: 1;">

**As a function of $y$ (with $x$ fixed)**

This is the **likelihood** of $y$ given the observed value $X=x$. It measures the relative support for different values of $y$.

Does **NOT** sum to 1 over $y$.

</div>
</div>

</div>

---

## Independence of Random Variables

Two random variables $X$ and $Y$ are **independent** if knowing the value of one provides no information about the other:

<div class="formula">
$$
\begin{aligned}
p_{X,Y}(x,y) &= p_X(x) \cdot p_Y(y) \\
p_{X|Y}(x|y) &= p_X(x) \\
p_{Y|X}(y|x) &= p_Y(y)
\end{aligned}
$$
</div>

---

## Bayes' Theorem

Bayes' theorem relates the conditional and marginal distributions of random variables.

<div class="formula">
$$
p_{X|Y}(x|y) = \frac{p_{Y|X}(y|x) p_X(x)}{p_Y(y)}, \quad p_Y(y) > 0
$$
</div>

This theorem is fundamental in Bayesian inference, allowing us to update our beliefs about $X$ after observing $Y$.

---
