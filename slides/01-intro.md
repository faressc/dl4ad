# Deep Learning for<br>Audio Data

Note:
    - Seminar Deep Learning for Audio Data

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
- Setup and optimization of deep learning workflows, pipelines and lifecycles

<div class="highlight">
  <p>Emphasis on: Creative applications of deep learning in the<br>music production domain</p>
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
            <td>Multi-layer Perceptrons</td>
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
    - *Deep Learning for Audio Data* [13575](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13575)
- AKT students can take this course as part of *Wahlpflichtbereich Vertiefung*

### Grading

- Project in Deep Learning for Audio Data (2/3 Grade)

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

# History of Deep Learning

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

## Deep Learning Evolution

<!-- Neural Networks & Architectures Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Architectures & Layers</div>
        <div class="timeline-text">Evolution of network architectures and layer innovations</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;">
        <div class="timeline-dot" style="--year: 1943;"></div>
        <div class="timeline-item" style="--year: 1943;">
            <div class="timeline-content">
                <div class="timeline-year">1943</div>
                <div class="timeline-name">Artificial Neurons</div>
                <div class="timeline-author">McCulloch & Pitts</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1943;"></div>
        <div class="timeline-dot" style="--year: 1957;"></div>
        <div class="timeline-item" style="--year: 1957;">
            <div class="timeline-content">
                <div class="timeline-year">1957</div>
                <div class="timeline-name">Perceptron</div>
                <div class="timeline-author">Rosenblatt, F.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1957;"></div>
        <div class="timeline-dot" style="--year: 1965;"></div>
        <div class="timeline-item" style="--year: 1965;">
            <div class="timeline-content">
                <div class="timeline-year">1965</div>
                <div class="timeline-name">Deep Networks</div>
                <div class="timeline-author">Ivakhnenko & Lapa</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1965;"></div>
        <div class="timeline-dot" style="--year: 1979;"></div>
        <div class="timeline-item" style="--year: 1979;">
            <div class="timeline-content">
                <div class="timeline-year">1979</div>
                <div class="timeline-name">Convolutional Networks</div>
                <div class="timeline-author">Fukushima, K.</div>
            </div>
        </div> 
        <div class="timeline-connector" style="--year: 1979;"></div>
        <div class="timeline-dot" style="--year: 1982;"></div>
        <div class="timeline-item" style="--year: 1982;">
            <div class="timeline-content">
                <div class="timeline-year">1982</div>
                <div class="timeline-name">Recurrent Networks</div>
                <div class="timeline-author">Hopfield</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1982;"></div>
        <div class="timeline-dot" style="--year: 1997;"></div>
        <div class="timeline-item" style="--year: 1997;">
            <div class="timeline-content">
                <div class="timeline-year">1997</div>
                <div class="timeline-name">LSTM</div>
                <div class="timeline-author">Hochreiter & Schmidhuber</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1997;"></div>
        <div class="timeline-dot" style="--year: 2006;"></div>
        <div class="timeline-item" style="--year: 2006;">
            <div class="timeline-content">
                <div class="timeline-year">2006</div>
                <div class="timeline-name">Deep Belief Networks</div>
                <div class="timeline-author">Hinton, G. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2006;"></div>
        <div class="timeline-dot positioned" style="--year: 2012;"></div>
        <div class="timeline-item" style="--year: 2012;">
            <div class="timeline-content">
                <div class="timeline-year">2012</div>
                <div class="timeline-name">AlexNet</div>
                <div class="timeline-author">Krizhevsky et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2012;"></div>
    </div>
</div>

<!-- Training & Optimization Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Methods for efficient learning and gradient-based optimization</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;">
        <div class="timeline-dot" style="--year: 1967;"></div>
        <div class="timeline-item" style="--year: 1967;">
            <div class="timeline-content">
                <div class="timeline-year">1967</div>
                <div class="timeline-name">Stochastic Gradient Descent</div>
                <div class="timeline-author">Amari, S.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1967;"></div>
        <div class="timeline-dot" style="--year: 1970;"></div>
        <div class="timeline-item" style="--year: 1970;">
            <div class="timeline-content">
                <div class="timeline-year">1970</div>
                <div class="timeline-name">Automatic Differentiation</div>
                <div class="timeline-author">Linnainmaa, S.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1970;"></div>
        <div class="timeline-dot" style="--year: 1986;"></div>
        <div class="timeline-item" style="--year: 1986;">
            <div class="timeline-content">
                <div class="timeline-year">1986</div>
                <div class="timeline-name">Backpropagation</div>
                <div class="timeline-author">Hinton et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1986;"></div>
        <div class="timeline-dot" style="--year: 1992;"></div>
        <div class="timeline-item" style="--year: 1992;">
            <div class="timeline-content">
                <div class="timeline-year">1992</div>
                <div class="timeline-name">Weight Decay</div>
                <div class="timeline-author">Krogh & Hertz</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1992;"></div>
        <div class="timeline-dot" style="--year: 2010;"></div>
        <div class="timeline-item" style="--year: 2010;">
            <div class="timeline-content">
                <div class="timeline-year">2010</div>
                <div class="timeline-name">ReLU & Xavier Init</div>
                <div class="timeline-author">Nair, Hinton & Glorot</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2010;"></div>
        <div class="timeline-dot positioned" style="--year: 2012;"></div>
        <div class="timeline-item" style="--year: 2012;">
            <div class="timeline-content">
                <div class="timeline-year">2012</div>
                <div class="timeline-name">Dropout</div>
                <div class="timeline-author">Hinton, G. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2012;"></div>
    </div>
</div>

<!-- Software & Datasets Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Datasets</div>
        <div class="timeline-text">Tools, platforms, and milestones that enabled practical deep learning</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 1943; --end-year: 2012;">
        <div class="timeline-dot" style="--year: 1997;"></div>
        <div class="timeline-item" style="--year: 1997;">
            <div class="timeline-content">
                <div class="timeline-year">1997</div>
                <div class="timeline-name">Deep Blue</div>
                <div class="timeline-author">IBM</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1997;"></div>
        <div class="timeline-dot" style="--year: 1998;"></div>
        <div class="timeline-item" style="--year: 1998;">
            <div class="timeline-content">
                <div class="timeline-year">1998</div>
                <div class="timeline-name">MNIST Dataset & LeNet 5</div>
                <div class="timeline-author">LeCun, Y. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 1998;"></div>
        <div class="timeline-dot" style="--year: 2002;"></div>
        <div class="timeline-item" style="--year: 2002;">
            <div class="timeline-content">
                <div class="timeline-year">2002</div>
                <div class="timeline-name">Torch Framework</div>
                <div class="timeline-author">Torch Team</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2002;"></div>
        <div class="timeline-dot" style="--year: 2007;"></div>
        <div class="timeline-item" style="--year: 2007;">
            <div class="timeline-content">
                <div class="timeline-year">2007</div>
                <div class="timeline-name">CUDA Platform</div>
                <div class="timeline-author">NVIDIA</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2007;"></div>
        <div class="timeline-dot" style="--year: 2009;"></div>
        <div class="timeline-item" style="--year: 2009;">
            <div class="timeline-content">
                <div class="timeline-year">2009</div>
                <div class="timeline-name">ImageNet Dataset</div>
                <div class="timeline-author">Deng, J. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2009;"></div>
        <div class="timeline-dot positioned" style="--year: 2011;"></div>
        <div class="timeline-item" style="--year: 2011;">
            <div class="timeline-content">
                <div class="timeline-year">2011</div>
                <div class="timeline-name">Siri</div>
                <div class="timeline-author">Apple Inc.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2011;"></div>
    </div>
</div>

---

## Modern Deep Learning

<!-- Layers & Architectures Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Architectures & Models</div>
        <div class="timeline-text">Advanced architectures and generative models transforming AI capabilities</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
        <div class="timeline-dot" style="--year: 2014;"></div>
        <div class="timeline-item" style="--year: 2014;">
            <div class="timeline-content">
                <div class="timeline-year">2014</div>
                <div class="timeline-name">VAEs & GANs</div>
                <div class="timeline-author">Kingma & Goodfellow</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2014;"></div>
        <div class="timeline-dot" style="--year: 2015;"></div>
        <div class="timeline-item" style="--year: 2015;">
            <div class="timeline-content">
                <div class="timeline-year">2015</div>
                <div class="timeline-name">ResNet & Diffusion</div>
                <div class="timeline-author">He et al. & Sohl-Dickstein et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2015;"></div>
        <div class="timeline-dot" style="--year: 2016;"></div>
        <div class="timeline-item" style="--year: 2016;">
            <div class="timeline-content">
                <div class="timeline-year">2016</div>
                <div class="timeline-name">Style Transfer & WaveNet</div>
                <div class="timeline-author">Gatys & van den Oord</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2016;"></div>
        <div class="timeline-dot" style="--year: 2017;"></div>
        <div class="timeline-item" style="--year: 2017;">
            <div class="timeline-content">
                <div class="timeline-year">2017</div>
                <div class="timeline-name">Transformers</div>
                <div class="timeline-author">Vaswani et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2017;"></div>
        <div class="timeline-dot" style="--year: 2021;"></div>
        <div class="timeline-item" style="--year: 2021;">
            <div class="timeline-content">
                <div class="timeline-year">2021</div>
                <div class="timeline-name">ViT & CLIP</div>
                <div class="timeline-author">Dosovitskiy & Radford</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2021;"></div>
        <div class="timeline-dot" style="--year: 2022;"></div>
        <div class="timeline-item" style="--year: 2022;">
            <div class="timeline-content">
                <div class="timeline-year">2022</div>
                <div class="timeline-name">DiT (Diffusion Transformer)</div>
                <div class="timeline-author">Peebles & Xie</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2022;"></div>
        <div class="timeline-dot positioned" style="--year: 2023;"></div>
        <div class="timeline-item" style="--year: 2023;">
            <div class="timeline-content">
                <div class="timeline-year">2023</div>
                <div class="timeline-name">Mamba</div>
                <div class="timeline-author">Gu & Dao</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2023;"></div>
    </div>
</div>

<!-- Training & Optimization Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Training & Optimization</div>
        <div class="timeline-text">Advanced learning techniques and representation learning breakthroughs</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
        <div class="timeline-dot" style="--year: 2013;"></div>
        <div class="timeline-item" style="--year: 2013;">
            <div class="timeline-content">
                <div class="timeline-year">2013</div>
                <div class="timeline-name">Word2Vec</div>
                <div class="timeline-author">Mikolov, T. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2013;"></div>
        <div class="timeline-dot" style="--year: 2014;"></div>
        <div class="timeline-item" style="--year: 2014;">
            <div class="timeline-content">
                <div class="timeline-year">2014</div>
                <div class="timeline-name">Attention Mechanism</div>
                <div class="timeline-author">Bahdanau, D. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2014;"></div>
        <div class="timeline-dot" style="--year: 2015;"></div>
        <div class="timeline-item" style="--year: 2015;">
            <div class="timeline-content">
                <div class="timeline-year">2015</div>
                <div class="timeline-name">BatchNorm & Adam</div>
                <div class="timeline-author">Ioffe & Kingma</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2015;"></div>
        <div class="timeline-dot" style="--year: 2016;"></div>
        <div class="timeline-item" style="--year: 2016;">
            <div class="timeline-content">
                <div class="timeline-year">2016</div>
                <div class="timeline-name">Layer Normalization</div>
                <div class="timeline-author">Ba, J. L. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2016;"></div>
        <div class="timeline-dot positioned" style="--year: 2020;"></div>
        <div class="timeline-item" style="--year: 2020;">
            <div class="timeline-content">
                <div class="timeline-year">2020</div>
                <div class="timeline-name">DDPM</div>
                <div class="timeline-author">Ho, J. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2020;"></div>
    </div>
</div>

<!-- Software & Applications Timeline -->
<div class="timeline-container" style="flex-direction: row;">
    <div style="width: 20%;">
        <div class="timeline-title">Software & Applications</div>
        <div class="timeline-text">Practical deployment and mainstream adoption of deep learning systems</div>
    </div>
    <div class="timeline" style="width: 80%; --start-year: 2013; --end-year: 2023;">
        <div class="timeline-dot" style="--year: 2016;"></div>
        <div class="timeline-item" style="--year: 2016;">
            <div class="timeline-content">
                <div class="timeline-year">2016</div>
                <div class="timeline-name">AlphaGo</div>
                <div class="timeline-author">Silver, D. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2016;"></div>
        <div class="timeline-dot" style="--year: 2017;"></div>
        <div class="timeline-item" style="--year: 2017;">
            <div class="timeline-content">
                <div class="timeline-year">2017</div>
                <div class="timeline-name">PyTorch</div>
                <div class="timeline-author">Paszke, A. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2017;"></div>
        <div class="timeline-dot" style="--year: 2018;"></div>
        <div class="timeline-item" style="--year: 2018;">
            <div class="timeline-content">
                <div class="timeline-year">2018</div>
                <div class="timeline-name">GPT-1</div>
                <div class="timeline-author">Radford & Devlin</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2018;"></div>
        <div class="timeline-dot" style="--year: 2020;"></div>
        <div class="timeline-item" style="--year: 2020;">
            <div class="timeline-content">
                <div class="timeline-year">2020</div>
                <div class="timeline-name">GPT-3</div>
                <div class="timeline-author">Brown, T. B. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2020;"></div>
        <div class="timeline-dot" style="--year: 2022;"></div>
        <div class="timeline-item" style="--year: 2022;">
            <div class="timeline-content">
                <div class="timeline-year">2022</div>
                <div class="timeline-name">ChatGPT & Stable Diffusion</div>
                <div class="timeline-author">OpenAI & Stability AI</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2022;"></div>
        <div class="timeline-dot positioned" style="--year: 2023;"></div>
        <div class="timeline-item" style="--year: 2023;">
            <div class="timeline-content">
                <div class="timeline-year">2023</div>
                <div class="timeline-name">LLaMA</div>
                <div class="timeline-author">Touvron, H. et al.</div>
            </div>
        </div>
        <div class="timeline-connector" style="--year: 2023;"></div>
    </div>
</div>

---

# Modern Neural Audio Systems

---

## Overview

---



---

## Setup

---

## References
