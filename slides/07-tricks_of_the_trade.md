# Tricks of the Trade

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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2015:0,2016:0,2017:0,2021:0">
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
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;" data-timeline-fragments-select="2017:0,2018:0,2020:0,2022:0,2023:0">
        {{TIMELINE:timeline_deep_software}}
    </div>
</div>

---

## Motivation for this Lecture

- Many fancy frameworks give the illusion that neural network training can magicly solve data science problems, with a few lines of code
- Just like other libraries or modules, that abstract away complexity

```python
>>> your_data = # plug your awesome dataset here
>>> model = SuperCrossValidator(SuperDuper.fit, your_data, ResNet50, SGDOptimizer)
# conquer world here
```

```python
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
```

<div class="reference" style="text-align: center;">
    Source: <a href="https://karpathy.github.io/2019/04/25/recipe/">The Recipe for Training Neural Networks</a> by Andrej Karpathy
</div>

<div class="highlight image-overlay fragment" style="width: 80%">
    Unfortunately, there is no magic network, normalization, or optimizer that fits all problems!
    It all depends on the data and the task at hand
</div>

---

## Motivation for this Lecture

- Neural network training fails silently most of the time
- In code if you plug an integer where a string is expected, you get an error
- You can easily unit test small parts of your code
- But how do you know if your neural network is learning correctly?
- Your model could be syntactically correct, but still there can be logical bugs
- And often even with the bugs the model trains surprisingly well, but the performance is suboptimal

<div class="highlight image-overlay fragment" style="width: 80%; text-align: left;">

- Lecture covers practical tips to debug and optimize neural network training
- Don't rush—understand mechanics and apply tricks systematically
- Start with simple baseline, add complexity incrementally

</div>

---

# How do we start?

---

## Become one with the Data

- Use a feature representation that makes sense for your data (Use the knowledge from MIRMLA course)
- Understand the data you are working with
- Visualize samples from the dataset
- Check for class imbalance
- Visualize distributions of features and pay special attention to outliers
- Finally, normalize or standardize features if necessary
- Check for data leakage between train and validation sets

---

## Set up a Simple Baseline Model

<div style="font-size: 0.8em;">

- Fix a random seed for reproducibility
- Start with a very simple "toy" model architecture
- Compute a simple human-understandable baseline metrics (e.g., accuracy, confusion matrix) on the train and validation set (use k-fold cross-validation for small datasets)
- Verify the loss function and metrics at initialization (e.g., random predictions should yield expected loss)
- Initialize weights properly (e.g. if you are regressing some values with mean 100, initialize the last layer bias to 100)
- Use a small subset (as little as 2 samples) of the train set to verify that the model can overfit it (i.e., loss goes to zero)
- Analyze and visualize model predictions at different layer stages (e.g., attention maps, embeddings, feature maps)
- Increase the complexity of the model gradually and monitor the performance on train and validation sets
- Visualize and analyze predictions on a fixed (unshuffled) set of samples from the validation set after every epoch
- Check the weights and neuron, as well as their gradients - compute statistics for the different layers (e.g., make sure they are not vanishing or exploding)

</div>

---

## Overfit

<div style="font-size: 0.8em;">

- Look into the related literature for similar problems and datasets and find an architecture that works well
- Do not use data augmentation or regularization at this stage
- The Adam optimizer is a good default choice for most problems with a learning rate of 1e-3
- Make sure your model can overfit on a small subset of the training data (e.g., 100 samples)
- Gradually increase the model complexity one step at a time until you can overfit on the full training set
- Be careful not to overcomplicate the model too early
- Beware of learning rate schedules if they are dependent on the number of epochs
- When training deep models, check for vanishing or exploding gradients and apply residual connections if necessary
- When having unstable activation scales consider using normalization layers

</div>

---

## Regularize

<div style="font-size: 0.8em;">

- Once you can overfit the training set, try to improve the generalization performance
- The best regularization method is to get more data
- If that is not possible, try data augmentation techniques suitable for your data modality (only on the training set)
- Decrease the model complexity if possible
- Pay attention to spuriously correlated features in the data and try to remove features that do not generalize well
- Add dropout, but pay attention with dropout and batch normalization together
- Try weight decay (L2 regularization) on the weights of the model
- Introduce early stopping based on the validation performance
- Transfer learning from a pretrained model can help regularization as well as it can be considered as inductive bias towards solutions that generalize well

</div>

---

## Tune

<div style="font-size: 0.8em;">

- Once you have a working model with good generalization performance, try to tune the hyperparameters
- Have a good version control system in place to track experiments - e.g., [dvc](https://dvc.org/)
- Have a systematic way to log and visualize training and validation metrics - e.g., [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://wandb.ai/) (Commercial)
- Optimize computation efficiency i.e., use mixed precision training
- Use random search or Bayesian optimization instead of grid search - i.e. with [optuna](https://optuna.org/)
- Focus on tuning the learning rate first, as it has the largest impact on performance, try using a learning rate finder, consider using warmup strategies
- Then tune the batch size, model architecture, and regularization parameters
- Consider using learning rate schedules, adaptive optimizers or different input representations
- Monitor the training and validation performance closely to avoid overfitting during hyperparameter tuning
- Use ensembles of models or mixtures of experts to boost performance further
- Finally, let the model train for a longer time to see if the performance improves further and use model checkpointing to save the best performing model

</div>

---

# Tricks of the Trade

---

## Choice of Activation Functions

<div style="font-size: 0.70em;">

<table>
<thead>
<tr>
<th>Activation</th>
<th>Function</th>
<th>Typical Use Case</th>
<th>Network Type</th>
</tr>
</thead>
<tbody>
<tr class="fragment" data-fragment-index="1">
<td><strong>ReLU</strong></td>
<td>$\text{ReLU}(z) = \max(0, z)$</td>
<td>Hidden layers (default choice)</td>
<td>CNNs, MLPs, ResNets</td>
</tr>
<tr class="fragment" data-fragment-index="3">
<td><strong>Leaky ReLU / PReLU</strong></td>
<td>$\text{LeakyReLU}(z) = \max(\alpha z, z)$</td>
<td>Hidden layers (when dying ReLU is an issue)</td>
<td>Deep CNNs, GANs</td>
</tr>
<tr class="fragment" data-fragment-index="5">
<td><strong>GELU</strong></td>
<td>$\text{GELU}(z) = z \cdot \Phi(z)$</td>
<td>Hidden layers in modern architectures</td>
<td>Transformers, BERT, GPT</td>
</tr>
<tr class="fragment" data-fragment-index="7">
<td><strong>Swish / SiLU</strong></td>
<td>$\text{Swish}(z) = \frac{z}{1 + e^{-z}}$</td>
<td>Hidden layers in deep networks</td>
<td>EfficientNet, modern CNNs</td>
</tr>
<tr class="fragment" data-fragment-index="9">
<td><strong>Tanh</strong></td>
<td>$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$</td>
<td>Hidden layers, gates</td>
<td>RNNs, LSTMs, GRUs</td>
</tr>
<tr class="fragment" data-fragment-index="11">
<td><strong>Sigmoid</strong></td>
<td>$\sigma(z) = \frac{1}{1 + e^{-z}}$</td>
<td>Output layer (binary classification), gates</td>
<td>Binary classifiers, LSTM gates</td>
</tr>
<tr class="fragment" data-fragment-index="13">
<td><strong>Softmax</strong></td>
<td>$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$</td>
<td>Output layer (multi-class classification)</td>
<td>Multi-class classifiers</td>
</tr>
<tr class="fragment" data-fragment-index="15">
<td><strong>Linear</strong></td>
<td>$f(z) = z$</td>
<td>Output layer (regression)</td>
<td>Regression models</td>
</tr>
</tbody>
</table>

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="2" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/ReLUActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/LeakyReLUActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="6" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/GELUActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="8" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/SwishActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="10" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/TanhActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="12" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/03-perceptrons/1080p60/SigmoidActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="14" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/SoftmaxActivationVisualization.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

---

## Choice of Initialization Schemes

<div style="font-size: 0.70em;">

<table>
<thead>
<tr>
<th>Initialization</th>
<th>Method</th>
<th>Typical Use Case</th>
<th>Network Type</th>
</tr>
</thead>
<tbody>
<tr class="fragment" data-fragment-index="1">
<td><strong>Xavier / Glorot</strong></td>
<td>$\mathbf{W} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$</td>
<td>Hidden layers with tanh/sigmoid activations</td>
<td>MLPs, shallow networks</td>
</tr>
<tr class="fragment" data-fragment-index="2">
<td><strong>He (Kaiming)</strong></td>
<td>$\mathbf{W} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$</td>
<td>Hidden layers with ReLU activations</td>
<td>CNNs, ResNets, deep networks</td>
</tr>
<tr class="fragment" data-fragment-index="3">
<td><strong>LeCun</strong></td>
<td>$\mathbf{W} \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)$</td>
<td>Hidden layers with SELU activations</td>
<td>Self-normalizing networks - Networks designed to maintain mean and variance without normalization</td>
</tr>
<tr class="fragment" data-fragment-index="4">
<td><strong>Orthogonal</strong></td>
<td>$\mathbf{W}$ = orthogonal matrix</td>
<td>Recurrent connections</td>
<td>RNNs, LSTMs, GRUs</td>
</tr>
<tr class="fragment" data-fragment-index="5">
<td><strong>Zero</strong></td>
<td>$\mathbf{W} = 0$</td>
<td>Bias terms only</td>
<td>All networks (biases)</td>
</tr>
<tr class="fragment" data-fragment-index="6">
<td><strong>Constant</strong></td>
<td>$\mathbf{W} = c$</td>
<td>Specific layer requirements</td>
<td>Output layers (regression)</td>
</tr>
</tbody>
</table>

</div>

<div class="highlight image-overlay fragment" data-fragment-index="8" style="width: 80%; text-align: left;">

**Key Principle:** Match initialization to activation function to maintain stable gradient flow
- Use He for ReLU and variants
- Use Xavier for tanh/sigmoid
- Use Orthogonal for recurrent connections

</div>

---

## Choice of Optimizers

<div style="font-size: 0.70em;">

<table>
<colgroup>
<col style="width: 15%;">
<col style="width: 35%;">
<col style="width: 30%;">
<col style="width: 20%;">
</colgroup>
<thead>
<tr>
<th>Optimizer</th>
<th>Update Rule</th>
<th>Typical Use Case</th>
<th>Network Type</th>
</tr>
</thead>
<tbody>
<tr class="fragment" data-fragment-index="1">
<tr class="fragment" data-fragment-index="2">
<td><strong>Mini-batch SGD + Momentum</strong></td>
<td>$\mathbf{m}_{t} = \beta \mathbf{m}_{t-1} + \nabla \mathcal{L}$ <br> $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_{t}$</td>
<td>Computer vision, training from scratch - noisier updates can better find global minima</td>
<td>CNNs, ResNets, image classification</td>
</tr>
<tr class="fragment" data-fragment-index="3">
<td><strong>Mini-batch SGD + RMSprop</strong></td>
<td>$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)(\nabla \mathcal{L})^2$ <br> $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \nabla \mathcal{L}$</td>
<td>Recurrent networks, non-stationary objectives</td>
<td>RNNs, online learning</td>
</tr>
<tr class="fragment" data-fragment-index="4">
<td><strong>Adam (RMSprop + Momentum)</strong></td>
<td>$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla \mathcal{L}$ <br> $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla \mathcal{L})^2$ <br> $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \mathbf{m}_t$</td>
<td>Default choice for most problems</td>
<td>Transformers, GANs, general purpose</td>
</tr>
<tr class="fragment" data-fragment-index="5">
<td><strong>AdamW</strong></td>
<td>Adam + decoupled weight decay</td>
<td>Modern deep learning, large models</td>
<td>BERT, GPT, ViT, large-scale models</td>
</tr>
</tbody>
</table>

</div>

<div class="highlight image-overlay fragment" data-fragment-index="9" style="width: 80%; text-align: left;">

**Key Principle:** Match optimizer to your problem characteristics
- **Adam/AdamW**: Default choice for most modern architectures (LR ~ 1e-3 to 1e-4)
- **SGD + Momentum**: Best for CNNs when training from scratch (LR ~ 0.1 with schedule)
- **RMSprop**: Good for RNNs and non-stationary problems
- **AdamW**: Preferred over Adam for large models with weight decay

</div>

---

## Learning Rate Schedules

<div style="font-size: 0.90em;">

- LR schedules can significantly impact convergence and performance
- Use step-based schedules (not epoch-based) for flexibility across batch sizes

</div>

<div style="font-size: 0.70em;">

<table>
<thead>
<tr>
<th>Schedule</th>
<th>Formula</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr class="fragment" data-fragment-index="1">
<td><strong>Step Decay</strong></td>
<td>$\eta_t = \eta_0 \times \gamma^{\lfloor t / T \rfloor}$</td>
<td>Simple baseline, works well for CNNs</td>
</tr>
<tr class="fragment" data-fragment-index="3">
<td><strong>Linear Decay</strong></td>
<td>$\eta_t = \eta_0 - \frac{(\eta_0 - \eta_{min}) \cdot t}{T}$</td>
<td>Linear decay from initial to minimum LR</td>
</tr>
<tr class="fragment" data-fragment-index="5">
<td><strong>Exponential Decay</strong></td>
<td>$\eta_t = \eta_0 \times \gamma^t$</td>
<td>Smooth continuous decay</td>
</tr>
<tr class="fragment" data-fragment-index="7">
<td><strong>Cosine Annealing</strong></td>
<td>$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$</td>
<td>Transformers, modern architectures, smoother than step decay</td>
</tr>
<tr class="fragment" data-fragment-index="9">
<td><strong>One Cycle Policy</strong></td>
<td>Warmup then cosine annealing</td>
<td>Fast convergence, good generalization, allows big learning rates, limited training budget</td>
</tr>
<tr class="fragment" data-fragment-index="11">
<td><strong>Warm Restarts (SGDR)</strong></td>
<td>$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi T_{cur}}{T_i}\right)\right)$</td>
<td>Snapshot ensembling, escape local minima, exploration</td>
</tr>
</tbody>
</table>

</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="2" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/StepDecaySchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="4" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/LinearDecaySchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="6" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/ExponentialDecaySchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="8" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/CosineAnnealingSchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="10" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/OneCyclePolicySchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="fragment appear-vanish image-overlay" data-fragment-index="12" style="text-align: center; width: 1200px; height: auto;">
    <video width="100%" data-autoplay loop muted controls>
        <source src="assets/videos/07-tricks_of_the_trade/1080p60/WarmRestartsSchedule.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<div class="highlight image-overlay fragment" data-fragment-index="13" style="width: 80%; text-align: left;">
    Attention: Learning rate schedules interact with optimizers differently; Consider the momentum term in optimizers like SGD with momentum or Adam when designing schedules.
</div>

---

## Residual Connections

- When training deep models, check for vanishing or exploding gradients and apply residual connections if necessary
- Residual connections help gradients flow through deep networks by providing shortcut paths
- 

---

## Normalization Layers


---

## Regularization Techniques

- Dropout
- Weight Decay
- Data Augmentation
- Early Stopping

---

## Transfer Learning & Pretrained Models

---

# Python Implementation
