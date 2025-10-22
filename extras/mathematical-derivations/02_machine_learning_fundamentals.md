# Mathematical Proofs for Machine Learning Fundamentals

## Bias-Variance Decomposition: Complete Mathematical Proof

### Problem Setup

We want to understand the expected prediction error of a model trained on different datasets and noise realizations.

**Given:**
- True relationship: $y = f^*(\mathbf{x}) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- Model prediction: $\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})$ learned from training data $D$
- Objective: Analyze the expected squared error $\mathbb{E}_{D,\epsilon} \left[ (y - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right]$

---

### Proof

#### Step 1: Substitute the True Relationship

$$
\mathbb{E}_{D,\epsilon} \left[ (y - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right] = \mathbb{E}_{D,\epsilon} \left[ (f^*(\mathbf{x}) + \epsilon - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right]
$$

#### Step 2: Expand the Square

Using $(a + b)^2 = a^2 + 2ab + b^2$:

$$
= \mathbb{E}_{D,\epsilon} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 + 2(f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))\epsilon + \epsilon^2 \right]
$$

#### Step 3: Apply Linearity of Expectation

$$
= \mathbb{E}_{D} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right] + 2\mathbb{E}_{D,\epsilon} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))\epsilon \right] + \mathbb{E}_{\epsilon} \left[ \epsilon^2 \right]
$$

#### Step 4: Simplify Using Properties of Noise

Since $\mathbb{E}[\epsilon] = 0$ and $\epsilon$ is independent of $D$:
- Middle term: $\mathbb{E}_{D,\epsilon} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))\epsilon \right] = \mathbb{E}_D[f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] \cdot \mathbb{E}_{\epsilon}[\epsilon] = 0$
- Last term: $\mathbb{E}[\epsilon^2] = \text{Var}(\epsilon) = \sigma^2$

Therefore:
$$
= \mathbb{E}_{D} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right] + \sigma^2
$$

#### Step 5: Decompose the First Term

Add and subtract $\mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})]$:

$$
\mathbb{E}_{D} \left[ (f^*(\mathbf{x}) - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right] = \mathbb{E}_{D} \left[ (f^*(\mathbf{x}) - \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] + \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right]
$$

#### Step 6: Define Components and Expand

Let:
- $a = f^*(\mathbf{x}) - \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})]$ (constant with respect to $D$)
- $b = \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x})$ (varies with $D$)

Then:
$$
\mathbb{E}_D[(a + b)^2] = \mathbb{E}_D[a^2 + 2ab + b^2]
$$

#### Step 7: Simplify Using Expectation Properties

**Observe:**

- $a$ is constant w.r.t. $D$ - thus $\mathbb{E}_D[a] = a$ and $\mathbb{E}_D[a^2] = a^2$

$$
= a^2 + 2a\mathbb{E}_D[b] + \mathbb{E}_D[b^2]
$$

**Key observations:**

- Since $b$ represents deviations from the mean: 
  $$\mathbb{E}_D[b] = \mathbb{E}_D[\mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] = \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] - \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})] = 0$$

Therefore, the cross term vanishes:
$$
= a^2 + \mathbb{E}_D[b^2]
$$

#### Step 8: Substitute Back

$$
= \left(f^*(\mathbf{x}) - \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})]\right)^2 + \mathbb{E}_D\left[\left(\hat{f}_{\boldsymbol{\theta}}(\mathbf{x}) - \mathbb{E}_D[\hat{f}_{\boldsymbol{\theta}}(\mathbf{x})]\right)^2\right]
$$

---

### Final Result

$$
\boxed{
\mathbb{E}_{D,\epsilon} \left[ (y - \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}))^2 \right] = \underbrace{\left( \mathbb{E}_D \left[ \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}) \right] - f^*(\mathbf{x}) \right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_D \left[ \left( \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}) - \mathbb{E}_D \left[ \hat{f}_{\boldsymbol{\theta}}(\mathbf{x}) \right] \right)^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}
}
$$

---

### Interpretation

| Component | Meaning | Cause |
|-----------|---------|-------|
| **Bias²** | Systematic error: how far is the average prediction from the truth? | Model too simple (underfitting) |
| **Variance** | Prediction instability: how much does the model change with different training data? | Model too complex (overfitting) |
| **Irreducible Error** | Random noise inherent in the data | Fundamental uncertainty in the problem |

**The Tradeoff:** Reducing bias typically increases variance, and vice versa. The optimal model minimizes their sum.