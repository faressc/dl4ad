---

## Regularization Techniques

To prevent overfitting in multilayer perceptrons, we can use various regularization techniques:

<div style="font-size: 0.90em;">

<div class="fragment" data-fragment-index="1">

- **L1 or L2 Regularization (Weight Decay for SGD)**: Adds a penalty term to the loss function proportional to the magnitude of the weights.

</div>
<div class="formula fragment appear-vanish" data-fragment-index="1">
$$
\begin{aligned}
\mathcal{L}_{reg} & = \mathcal{L} + \mathcal{R} \\
\mathcal{R}_1 & = \lambda \sum_{l} \sum_{i,j} |W_{ij}^{(l)}| \quad \text{(L1 Regularization)} \\
\mathcal{R}_2 & = \lambda \sum_{l} \sum_{i,j} (W_{ij}^{(l)})^2 \quad \text{(L2 Regularization)}
\end{aligned}
$$
</div>

<div class="fragment" data-fragment-index="2">

- **Batch Normalization**: Normalizes the inputs of each layer to have zero mean and unit variance, improving training stability.

</div>
<div class="formula fragment appear-vanish" data-fragment-index="2">
$$
\begin{aligned}
\mu_B & = \frac{1}{m} \sum_{i=1}^{m} z_i \\
\sigma_B^2 & = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu_B)^2\\
\hat{z}_i & = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\end{aligned}
$$
</div>

<div class="fragment" data-fragment-index="3">

- **Dropout**: Randomly sets a fraction of the neurons to zero during training to prevent co-adaptation.

</div>

<div class="fragment appear-vanish" data-fragment-index="3" style="text-align: center;">

<img src="assets/images/03-perceptrons/dropout.webp" alt="Dropout Illustration" style="width: 30%; margin-top: 20px;">
<div class="reference" style="text-align: center; margin-top: 10px;">Source: https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5</div>

</div>

<div class="fragment" data-fragment-index="4">

- **Early Stopping**: Monitors validation loss during training and stops when it starts to increase.

</div>

</div>

---