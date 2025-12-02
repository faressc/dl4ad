# Attention and Transformers

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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2015:1,2016:0,2017:1">
        {{TIMELINE:timeline_deep_architectures}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Advanced learning techniques and representation learning breakthroughs</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2014:1,2015:0,2016:0">
        {{TIMELINE:timeline_deep_training}}
    </div>
</div>

<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2017:0,2018:1,2020:1,2022:1,2023:1">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>

---

## Recurrent Layers: Recap

<div style="font-size: 0.8em;">

**Vanilla RNN**:

<ul>
<li>Maintains hidden state across time steps to capture temporal dependencies</li>
<li>Suffers from vanishing/exploding gradients for long sequences</li>
<li>Formula: $\mathbf{h}_t = \sigma\left(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}\right)$</li>
</ul>

<div class="fragment" data-fragment-index="1">

**LSTM (Long Short-Term Memory)**:

- Uses gating mechanisms (forget, input, output gates) and separate cell state
- Better at capturing long-term dependencies, mitigates vanishing gradients
- More parameters and computational cost than vanilla RNN

</div>

<div class="fragment" data-fragment-index="2">

**GRU (Gated Recurrent Unit)**:

- Simplified variant with reset and update gates (no separate cell state)
- Fewer parameters and computational cost than LSTM while maintaining comparable performance

</div>

<div class="fragment image-overlay highlight" data-fragment-index="3" style="text-align: left; width: 70%;">

Limitations of Recurrent Layers:
<ul>
<li>Recurrent layer do only have direct connections to the previous time step, limiting long-range dependency capture</li>
<li>Sequential processing hinders parallelization, leading to long training times</li>
</ul>

</div>

---

## Attention Mechanism

- Introduced to address limitations of recurrent layers by allowing direct connections between all time steps
- Computes a weighted sum of all input representations, where weights are determined by a compatibility function (attention scores)
- Enables modeling of long-range dependencies and parallel processing

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center;">
    <img src="assets/images/01-history/attention.png" alt="Attention Mechanism" style="max-width: 90%; max-height: 90%; object-fit: contain;">
    <div class="reference" data-fragment-index="1" style="margin: 10px; text-align: center;">
        Bahdanau, D., Cho, K., & Bengio, Y. (2016). Neural Machine Translation by Jointly Learning to Align and Translate (No. arXiv:1409.0473). arXiv. https://doi.org/10.48550/arXiv.1409.0473
    </div>
</div>

---

# Python Implementation
