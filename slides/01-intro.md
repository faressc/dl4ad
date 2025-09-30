# Machine Learning for<br>Audio Data

Note:
    - Seminar Machine Learning for Audio Data

<!-- .slide: data-state="no-header" -->
---

## Lecturer

<div style="display: flex; align-items: center; gap: 40px;">
    <div style="flex: 0 0 380px;">
        <figure style="text-align: center;">
            <img src="assets/images/profile_picture.jpg" alt="Fares Schulz" style="width: 100%; max-width: 380px; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px;">
            <figcaption>
                <strong>Fares Schulz</strong><br>
                <a href="mailto:fares.schulz@tu-berlin.de" class="small">fares.schulz@tu-berlin.de</a>
            </figcaption>
        </figure>
    </div>
    <div style="flex: 1 0; max-width: 500px; display: flex; flex-direction: column; text-align: left;">
        <strong style="margin-bottom: 0.5em;">Research Interests</strong>
        <div class="small">
            <div style="display: flex; align-items: flex-start;"><span style="margin-right: 0.6em;">•</span><span>Neural Networks for Audio Effects and Synthesis</span></div>
            <div style="display: flex; align-items: flex-start;"><span style="margin-right: 0.6em;">•</span><span>Real-Time Audio Environments</span></div>
            <div style="display: flex; align-items: flex-start;"><span style="margin-right: 0.6em;">•</span><span>Mixed-Signal Audio Processing</span></div>
        </div>
        <strong style="margin: 1em 0 0.5em 0;">Research Associate<br><span style="font-size: 0.8em; font-weight: normal;">Audio Communication Group</span></strong>
        <div class="small">
            <div>Lead of:</div>
            <div style="display: flex; align-items: flex-start;"><span style="margin-right: 0.6em;">•</span><span>Research Team Computer Music and Neural Audio Systems</span></div>
            <div style="display: flex; align-items: flex-start;"><span style="margin-right: 0.6em;">•</span><span>Electronic Studio of TU Berlin</span></div>
        </div>
    </div>
</div>

---

## Course Topic

- Mathematical and algorithmic fundamentals of machine learning with focus on deep learning and neural networks
- Methods for data exploration, preprocessing and augmentation
- Training, evaluation, export and inference of deep learning models
- Setup and optimization of machine learning workflows, pipelines and lifecycles

<div class="highlight">
  <p>Emphasis on: Creative applications of neural networks in the<br>music production domain</p>
</div>

---

## Deep Learning

<div style="text-align: center;">
    <img src="assets/images/ai_vs_ml_vs_dl.png" alt="AI vs ML vs DL" style="max-width: 90%; height: auto; margin-top: 28px;">
    <figcaption><a href="https://www.edureka.co/blog/ai-vs-machine-learning-vs-deep-learning/" style="margin-top: 30px; display: inline-block;">Atul. (2025). <em>AI vs Machine Learning vs Deep Learning</em>. Edureka.</a></figcaption>
</div>

---

## Types of Learning

<div class="tiles-grid">
    <div class="tile fragment custom select" data-fragment-index="1">
        <h3>Supervised<br>Learning</h3>
        <div class="tile-description fragment custom select" data-fragment-index="1">
            Learn from input-output pairs with external labels, where the algorithm is trained on a dataset containing both features and their corresponding correct answers.
        </div>
        <div class="tile-applications">
            <div class="applications-label">Common Applications:</div>
            Classification and regression tasks
        </div>
    </div>
    <div class="tile fragment custom select" data-fragment-index="1">
        <h3>Unsupervised<br>Learning</h3>
        <div class="tile-description fragment custom select" data-fragment-index="1">
            Learn from unlabeled data to discover hidden patterns and structures without explicit guidance or target variables (includes self-supervised learning).
        </div>
        <div class="tile-applications">
            <div class="applications-label">Common Applications:</div>
            Clustering, dimensionality reduction, and generative models
        </div>
    </div>
    <div class="tile">
        <h3>Reinforcement<br>Learning</h3>
        <div class="tile-description">
            Learn through trial-and-error interaction with an environment, optimizing actions based on rewards and penalties received.
        </div>
        <div class="tile-applications">
            <div class="applications-label">Common Applications:</div>
            Robotics, gaming, and autonomous systems
        </div>
    </div>
</div>

---

## Course Structure

<div style="display: flex; gap: 40px; align-items: flex-start; justify-content: flex-start;">
    <div style="flex: 1 0;">
        <ul>
            <li>Lectures: Theoretical foundations</li>
            <li>Jupyter Notebooks: Practical implementations in Python</li>
            <li>Projects: Hands-on experience with deep learning in audio</li>
        </ul>
        <strong>Where?</strong>
        <ul>
            <li>Resources: Slides and notebooks available on the course repository</li>
            <li>Project selection: On ISIS</li>
        </ul>
    </div>
    <div style="flex: 0 0 450px;">
        <figure>
            <div style="text-align: center;">
                <!-- Light theme QR code -->
                <img src="assets/images/repo_qr_code-light.png" alt="Course Repo QR (Light)" style="width: 360px;" class="picture-light">
                <!-- Dark theme QR code -->
                <img src="assets/images/repo_qr_code-dark.png" alt="Course Repo QR (Dark)" style="width: 360px;" class="picture-dark">
                <figcaption><a href="https://github.com/faressc/dl4ad" class="small">github.com/faressc/dl4ad</a></figcaption>
            </div>
        </figure>
    </div>
</div>

Note:
    - New branch at the end of the semester

---

## Dates and Deadlines

<div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.6em;">

<div style="flex: 1; width: 40%;">

<table style="width: 100%;">
    <thead>
        <tr>
            <th>Date</th>
            <th>Topic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="date">16.10.2025</td>
            <td>Introduction</td>
        </tr>
        <tr>
            <td class="date">23.10.2025</td>
            <td>Machine Learning Fundamentals</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">26.10.2025</td>
            <td>Course Application Deadline</td>
        </tr>
        <tr>
            <td class="date">30.10.2025</td>
            <td>Neural Networks</td>
        </tr>
        <tr>
            <td class="date">06.11.2025</td>
            <td>Convolutional and Recurrent Neural Networks</td>
        </tr>
        <tr>
            <td class="date">13.11.2025</td>
            <td>Preprocessing / Training Deep Architectures</td>
        </tr>
        <tr>
            <td class="date">20.11.2025</td>
            <td>Autoencoders / Transformers</td>
        </tr>
        <tr>
            <td class="date">27.11.2025</td>
            <td>Bayesian Inference</td>
        </tr>
        <tr>
            <td class="date">04.12.2025</td>
            <td>Variational Inference</td>
        </tr>
    </tbody>
</table>

</div>

<div style="flex: 1; width: 40%;">

<table style="width: 100%;">
    <thead>
        <tr>
            <th>Date</th>
            <th>Topic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="date">11.12.2025</td>
            <td>Variational Autoencoder</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">18.12.2025</td>
            <td>Project Pitches (14:00 - 18:00)</td>
        </tr>
        <tr>
            <td class="date">08.01.2026</td>
            <td>Adversarial Training</td>
        </tr>
        <tr>
            <td class="date">15.01.2026</td>
            <td>Diffusion Models</td>
        </tr>
        <tr>
            <td class="date">22.01.2026</td>
            <td>Real-Time Inference</td>
        </tr>
        <tr>
            <td class="date">29.01.2026</td>
            <td>Tricks of the Trade</td>
        </tr>
        <tr>
            <td class="date">05.02.2026</td>
            <td>Office Hours</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">12.02.2026</td>
            <td>Project Presentations (14:00 - 18:00)</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">31.03.2026</td>
            <td>Final Project Submission</td>
        </tr>
    </tbody>
</table>

</div>

</div>

---

## Course Application

<div style="display: flex; flex-direction: column; gap: 20px; font-size: 0.9em;">
    <div>
        <strong>Application Deadline:</strong> 26.10.2025
    </div>
    <div>
        <strong>How to Apply:</strong> Send a confirmation email to <a href="mailto:fares.schulz@tu-berlin.de">fares.schulz@tu-berlin.de</a>
    </div>
    <div>
        <strong>What to Include:</strong>
        <ul>
            <li>Your full name</li>
            <li>Your matriculation number</li>
            <li>Course of Study</li>
            <li>University Email Address</li>
            <li>Evidence of completion of the modules "Signale und Systeme" or "Digitale Signalverarbeitung" (e.g. excerpt from certificate of grades)</li>
        </ul>
    </div>
    <div>
        <strong>Note:</strong> The course is limited to 16 participants (4 groups of 4). Selection will be based on a lottery. After receiving confirmation, enroll in MOSES until 6. November 2025.
    </div>
</div>

---

## Assessment – ML4AD

**Project Presentation** (1/3 Grade) – *Date:* 18.12.2025

<ul class="small">
    <li>10-minute presentation per group and 5-minute Q&A session</li>
</ul>

**Git repository** (1/3 Grade) – *Deadline:* 31.03.2026

<ul class="small">
    <li>README with clear, step-by-step instructions for running your code</li>
    <li>Environment setup file (<code>requirements.txt</code>, <code>env.yml</code>, or <code>pyproject.toml</code>)</li>
    <li>Well-commented source code</li>
</ul>

**Project Paper** (1/3 Grade) – *Deadline:* 31.03.2026

<ul class="small">
    <li>Maximum 4 pages</li>
    <li>Use the <a href="https://www.ieee.org/conferences/publishing/templates.html">IEEE conference template</a></li>
</ul>

---

## Module Information

- This course is part of the module *Machine Learning and Big Data Processing with Audio and Music* [11005](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?number=11005) (6 ECTS)
- The module consists of:
    - *Music Information Retrieval und Machine Learning für Audio* [13702](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13702)
    - *Machine Learning for Audio Data* [13575](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13575)
- AKT students can take this course as part of *Wahlpflichtbereich Vertiefung*

### Grading

- Project in Machine Learning for Audio Data (2/3 Grade)

- Lecture excercises in Music Information Retrieval und Machine Learning für Audio (1/3 Grade)

---

## Prerequisites

- Python Knowledge – [Python and Numpy Refresher](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)
- Linear Algebra – [Linear Algebra Review](https://see.stanford.edu/materials/aimlcs229/cs229-linalg.pdf) / [UDL Book Appendix B](https://github.com/udlbook/udlbook)
- Calculus – [Calculus Review](https://people.uncw.edu/hermanr/pde1/pdebook/CalcRev.pdf) / [UDL Book Appendix B](https://github.com/udlbook/udlbook)
- Probability Theory– [Probability Theory Review](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf) / [UDL Book Appendix C](https://github.com/udlbook/udlbook)
- Statistics – [Statistics Cheat Sheet](https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics) / [UDL Book Appendix C](https://github.com/udlbook/udlbook)

**Digital Signal Processing**

<div class="highlight">
    <p>Successful completion of the modules <em>Signale und Systeme</em> <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40700">40700</a>, <em>Digitale Signalverarbeitung</em> <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=10002">10002</a> or similar is required.</p>
</div>

Note: If you don't know if your course counts, drop me an email.

---

## Additional Resources

- [Creative Machine Learning](https://github.com/acids-ircam/creative_ml) – Prof. Philippe Esling
- [Understanding Deep Learning](https://github.com/udlbook/udlbook) – Simon J.D. Prince
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) – Andrew Ng
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) – Andrew Ng
- [Milestone Paper Overview](https://github.com/faressc/dl4ad/blob/main/extras/milestone_paper_overview.md)

**TU Berlin Modules**

- *Machine Learning I/II* ([40550](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40550)/[40551](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40551))
- *Deep Learning I/II* ([41071](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41071)/[41072](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41072))

---

## Machine Learning History

15. **1943** – First mathematical model of artificial neurons — McCulloch, W. & Pitts, W. (1943).  
18. **1957** – Perceptron, first trainable artificial neural network — Rosenblatt, F. (1957). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*.  
19. **1965** – First practical deep neural network learning algorithms — Ivakhnenko, A. & Lapa, V. (1965).  
20. **1967–68** – Stochastic gradient descent for neural network optimization — Amari, S. (1967-68).
21. **1970** – Automatic differentiation algorithm, theoretical basis for backpropagation — Linnainmaa, S. (1970). *The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors*.  
22. **1979** – Neocognitron, first hierarchical convolutional neural network — Fukushima, K. (1979).  
23. **1979** – k-means clustering algorithm for unsupervised partitioning of data into k clusters based on nearest centroids — Hartigan, J. A. & Wong, M. A. (1979). *Algorithm AS 136: A K-Means Clustering Algorithm. Building on MacQueen's 1967 concept*.  
24. **1986** – Backpropagation algorithm enabling efficient neural network training — Hinton, G., Rumelhart, D., & Williams, R. (1986). *Learning Representations by Back-Propagating Errors*.  
25. **1989–1998** – LeNet-5: successful application of CNNs to handwritten digit recognition — LeCun, Y. et al. (1989-1998).  
26. **1992** – Weight decay (L2 regularization) preventing overfitting in neural networks — Krogh, A. & Hertz, J. (1992). *A Simple Weight Decay Can Improve Generalization*.  
27. **1995–1999** – Support Vector Machines with kernel trick for non-linear classification — Vapnik, V. (1995-1999).  
28. **1997** – LSTM networks solving vanishing gradient problem in RNNs — Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*.  
29. **1997** – Deep Blue defeats world chess champion, AI milestone — IBM (1997).  
30. **1999-2001** – Gradient Boosting machines combining weak learners sequentially to correct errors — Friedman, J. (1999-2001). *Greedy Function Approximation: A Gradient Boosting Machine*.
31. **2001** – Random Forests, powerful ensemble method for classification and regression — Breiman, L. (2001). *Random Forests*.  
32. **2002** – Torch framework democratizing machine learning research — Torch Development Team (2002).
33. **2006** – Deep Belief Networks enabling unsupervised pre-training of deep networks — Hinton, G. et al. (2006). *A Fast Learning Algorithm for Deep Belief Nets*.  
34. **2007** – CUDA platform enabling massively parallel GPU computation — NVIDIA (2007).  
35. **2009** – ImageNet dataset establishing large-scale visual recognition benchmark — Deng, J. et al. (2009).  
36. **2010** – ReLU activation function enabling deeper network training — Nair, V. & Hinton, G. (2010).  
37. **2010** – Xavier initialization solving gradient flow in deep networks — Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*.  
38. **2011** – Siri democratizing AI through voice interfaces — Apple Inc. (2011).  
39. **2012** – Dropout preventing overfitting through random neuron deactivation — Hinton, G. et al. (2012). *Improving neural networks by preventing co-adaptation of feature detectors*.  
40. **2012** – AlexNet breakthrough: deep CNNs + GPUs dominating computer vision — Krizhevsky, A., Sutskever, I., & Hinton, G. (2012).  
41. **2013** – Word2Vec creating dense semantic word representations — Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.  
42. **2014** – Attention mechanism enabling focus on relevant input regions — Bahdanau, D. et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*.  
43. **2014** – Variational Autoencoders combining probabilistic modeling with deep learning — Kingma, D. P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*.  
44. **2014** – Generative Adversarial Networks: adversarial training for realistic data generation — Goodfellow, I. et al. (2014). *Generative Adversarial Nets*.  
45. **2015** – Batch Normalization stabilizing and accelerating deep network training — Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.  
46. **2015** – Adam optimizer combining momentum with adaptive learning rates — Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*.  
47. **2015** – ResNet using skip connections to train extremely deep networks — He, K. et al. (2015). *Deep Residual Learning for Image Recognition*.  
48. **2015** – U-Net encoder-decoder architecture for precise image segmentation — Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.  
49. **2015** – Diffusion models using thermodynamic principles for generation — Sohl-Dickstein, J. et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*.  
50. **2015** – YOLO unified single-shot object detection architecture — Redmon, J. et al. (2015). *You Only Look Once: Unified, Real-Time Object Detection*.  
51. **2016** – Layer Normalization improving training stability across sequence lengths — Ba, J. L. et al. (2016). *Layer Normalization*.  
52. **2016** – Neural style transfer combining content and artistic style in images — Gatys, L. A. et al. (2016). *Image Style Transfer Using Convolutional Neural Networks*.  
53. **2016** – WaveNet generating raw audio waveforms with dilated convolutions — van den Oord, A. et al. (2016). *WaveNet: A Generative Model for Raw Audio*.  
54. **2016** – AlphaGo achieving superhuman performance in complex strategy game — Silver, D. et al. (2016). Google DeepMind.  
55. **2017** – PyTorch enabling dynamic neural networks with eager execution — Paszke, A. et al. (2017). Facebook AI Research.  
56. **2017** – Transformer architecture replacing RNNs with self-attention mechanisms — Vaswani, A. et al. (2017). *Attention Is All You Need*.  
57. **2018** – GPT-1 demonstrating unsupervised pre-training for language understanding — Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*.  
58. **2018** – BERT achieving bidirectional context understanding through masked language modeling — Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.  
59. **2020** – GPT-3 exhibiting emergent few-shot learning capabilities at massive scale — Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners*.  
60. **2020** – DDPM making diffusion models practical for high-quality image synthesis — Ho, J. et al. (2020). *Denoising Diffusion Probabilistic Models*.
61. **2021** – Vision Transformer proving Transformers can excel beyond NLP — Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.  
62. **2021** – CLIP learning joint text-image representations through contrastive learning — Radford, A. et al. (2021). *Learning Transferable Visual Representations from Natural Language Supervision*.  
63. **2021** – S4 achieving linear-time sequence modeling with structured state spaces — Gu, A. et al. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces*.  
64. **2021** – RAVE enabling real-time neural audio synthesis and manipulation — Caillon, A. & Esling, P. (2021).  *RAVE: A Real-time Audio Variational Autoencoder for End-to-End Sound Modeling*.
65. **2022** – ChatGPT demonstrating conversational AI capabilities to mainstream audiences — OpenAI (2022). Based on GPT-3.5.  
66. **2022** – Stable Diffusion open-sourcing high-quality text-to-image generation — Rombach, R. et al. (2022). Stability AI.  
67. **2022** – DiT (Diffusion Transformer) replacing U-Net with Transformer architecture in diffusion — Peebles, W. & Xie, S. (2022). *Scalable Diffusion Models with Transformers*.  
68. **2023** – LLaMA demonstrating efficient training of competitive open-source language models — Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*.  
69. **2023** – Mamba achieving linear scaling for sequence length with selective state spaces — Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.

---

## Mathematical Foundations

<!-- Calculus & Optimization Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Calculus & Linear Algebra</div>
        <div class="timeline-text">Basis for optimization algorithms and machine learning model operations</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1948;">
        <div class="timeline-dot" style="--year: 1676;"></div>
        <div class="timeline-item" style="--year: 1676;">
            <div class="timeline-content">
                <div class="timeline-year">1676</div>
                <div class="timeline-name">Chain Rule</div>
                <div class="timeline-author">Leibniz, G. W.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1676;"></div>
        <div class="timeline-dot" style="--year: 1805;"></div>
        <div class="timeline-item" style="--year: 1805;">
            <div class="timeline-content">
                <div class="timeline-year">1805</div>
                <div class="timeline-name">Least Squares</div>
                <div class="timeline-author">Legendre, A. M.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1805;"></div>
        <div class="timeline-dot" style="--year: 1809;"></div>
        <div class="timeline-item" style="--year: 1809;">
            <div class="timeline-content">
                <div class="timeline-year">1809</div>
                <div class="timeline-name">Normal Equations</div>
                <div class="timeline-author">Gauss, C. F.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1809;"></div>
        <div class="timeline-dot" style="--year: 1847;"></div>
        <div class="timeline-item" style="--year: 1847;">
            <div class="timeline-content">
                <div class="timeline-year">1847</div>
                <div class="timeline-name">Gradient Descent</div>
                <div class="timeline-author">Cauchy, A. L.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1847;"></div>
        <div class="timeline-dot" style="--year: 1858;"></div>
        <div class="timeline-item" style="--year: 1858;">
            <div class="timeline-content">
                <div class="timeline-year">1858</div>
                <div class="timeline-name">Eigenvalue Theory</div>
                <div class="timeline-author">Cayley & Hamilton</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1858;"></div>
        <div class="timeline-dot" style="--year: 1901;"></div>
        <div class="timeline-item" style="--year: 1901;">
            <div class="timeline-content">
                <div class="timeline-year">1901</div>
                <div class="timeline-name">PCA</div>
                <div class="timeline-author">Pearson, K.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1901;"></div>
    </div>
</div>

<!-- Probability & Statistics Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Probability & Statistics</div>
        <div class="timeline-text">Basis for Bayesian methods, statistical inference, and generative models</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1948;">
        <div class="timeline-dot" style="--year: 1763;"></div>
        <div class="timeline-item" style="--year: 1763;">
            <div class="timeline-content">
                <div class="timeline-year">1763</div>
                <div class="timeline-name">Bayes' Theorem</div>
                <div class="timeline-author">Bayes, T.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1763;"></div>
        <div class="timeline-dot" style="--year: 1812;"></div>
        <div class="timeline-item" style="--year: 1812;">
            <div class="timeline-content">
                <div class="timeline-year">1812</div>
                <div class="timeline-name">Bayesian Probability</div>
                <div class="timeline-author">Laplace, P. S.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1812;"></div>
        <div class="timeline-dot" style="--year: 1815;"></div>
        <div class="timeline-item" style="--year: 1815;">
            <div class="timeline-content">
                <div class="timeline-year">1815</div>
                <div class="timeline-name">Gaussian Distribution</div>
                <div class="timeline-author">Gauss, C. F.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1815;"></div>
        <div class="timeline-dot" style="--year: 1830;"></div>
        <div class="timeline-item" style="--year: 1830;">
            <div class="timeline-content">
                <div class="timeline-year">1830</div>
                <div class="timeline-name">Central Limit Theorem</div>
                <div class="timeline-author">Various</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1830;"></div>
        <div class="timeline-dot" style="--year: 1922;"></div>
        <div class="timeline-item" style="--year: 1922;">
            <div class="timeline-content">
                <div class="timeline-year">1922</div>
                <div class="timeline-name">Maximum Likelihood</div>
                <div class="timeline-author">Fisher, R.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1922;"></div>
    </div>
</div>

<!-- Information & Computation Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Information & Computation</div>
        <div class="timeline-text">Foundations of algorithmic thinking and information theory</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1676; --end-year: 1948;">
        <div class="timeline-dot" style="--year: 1843;"></div>
        <div class="timeline-item" style="--year: 1843;">
            <div class="timeline-content">
                <div class="timeline-year">1843</div>
                <div class="timeline-name">First Computer Algorithm</div>
                <div class="timeline-author">Lovelace, A.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1843;"></div>
        <div class="timeline-dot" style="--year: 1936;"></div>
        <div class="timeline-item" style="--year: 1936;">
            <div class="timeline-content">
                <div class="timeline-year">1936</div>
                <div class="timeline-name">Turing Machine</div>
                <div class="timeline-author">Turing, A.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1936;"></div>
        <div class="timeline-dot" style="--year: 1947;"></div>
        <div class="timeline-item" style="--year: 1947;">
            <div class="timeline-content">
                <div class="timeline-year">1947</div>
                <div class="timeline-name">Linear Programming</div>
                <div class="timeline-author">Dantzig, G.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1947;"></div>
        <div class="timeline-dot positioned" style="--year: 1948;"></div>
        <div class="timeline-item" style="--year: 1948;">
            <div class="timeline-content">
                <div class="timeline-year">1948</div>
                <div class="timeline-name">Information Theory</div>
                <div class="timeline-author">Shannon, C.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1948;"></div>
    </div>
</div>

---

## State-of-the-art use cases

---

## Setup

---

## References
