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

- Understand the data you are working with
- Visualize samples from the dataset
- Check for class imbalance
- Visualize distributions of features and pay special attention to outliers
- Finally, normalize or standardize features if necessary

---

## Set up a Simple Baseline Model

<div style="font-size: 0.8em;">

- Fix a random seed for reproducibility
- Start with a very simple "toy" model architecture
- Compute a simple human-understandable baseline metrics (e.g., accuracy) on the train and validation set
- Verify the loss function and metrics at initialization (e.g., random predictions should yield expected loss)
- Initialize weights properly (e.g. if you are regressing some values with mean 100, initialize the last layer bias to 100)
- Use a small subset (as little as 2 samples) of the train set to verify that the model can overfit it (i.e., loss goes to zero)
- Increase the complexity of the model gradually and monitor the performance on train and validation sets
- Visualize and analyze predictions on a fixed (unshuffled) set of samples from the validation set after every epoch
- Check the gradients and weights statistics (e.g., make sure they are not vanishing or exploding)

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
- Beware of learning rate schedules as they often depend on the total number of epochs
- When training deep models, check for vanishing or exploding gradients and apply residual connections if necessary
- When having unstable activation scales consider using normalization layers

</div>

---

## Regularize

<div style="font-size: 0.8em;">

- Once you can overfit the training set, try to improve the generalization performance
- The best regularization method is to get more data
- If that is not possible, try data augmentation techniques suitable for your data modality
- Decrease the model complexity if possible
- Pay attention to spuriously correlated features in the data and try to remove features that do not generalize well
- Add dropout, but pay attention with dropout and batch normalization together
- Try weight decay (L2 regularization) on the weights of the model
- Introduce early stopping based on the validation performance

</div>

---

## Tune

<div style="font-size: 0.8em;">

- Once you have a working model with good generalization performance, try to tune the hyperparameters
- Use random search or Bayesian optimization instead of grid search - i.e. with [optuna](https://optuna.org/)
- Focus on tuning the learning rate first, as it has the largest impact on performance
- Then tune the batch size, model architecture, and regularization parameters
- Consider using learning rate schedules or adaptive optimizers
- Monitor the training and validation performance closely to avoid overfitting during hyperparameter tuning
- Finally, let the model train for a longer time to see if the performance improves further

</div>

---



---

# Python Implementation
