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
        Or if $\mathbf{x}$ includes a bias input $x_0 = 1$, we can fold $b$ into the weights:
    </div>
</div>

---