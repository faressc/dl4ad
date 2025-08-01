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
                <img src="assets/images/repo_qr_code.png" alt="Course Repo QR (Light)" style="width: 360px;" class="picture-light">
                <!-- Dark theme QR code -->
                <img src="assets/images/repo_qr_code_dark.png" alt="Course Repo QR (Dark)" style="width: 360px;" class="picture-dark">
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
            <td>Introduction / Machine Learning Fundamentals I</td>
        </tr>
        <tr>
            <td class="date">23.10.2025</td>
            <td>Machine Learning Fundamentals II</td>
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
            <td>Project Pitches</td>
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
        <strong>Note:</strong> The course is limited to 16 participants (4 groups of 4). Selection will be based on a lottery.
    </div>
</div>

---

## Assessment

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

- This course is part of the module [Machine Learning and Big Data Processing with Audio and Music](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?number=11005) (6 ECTS)
- The module consists of:
    - [Big Data Processing with Audio and Music](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13702)
    - [Machine Learning for Audio Data](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13575)
- AKT students can take this course as part of "Wahlpflichtbereich Vertiefung"

---

## Prerequisites

- Python Knowledge – [Python and Numpy Refresher](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)
- Linear Algebra – [Linear Algebra Review and Reference](https://see.stanford.edu/materials/aimlcs229/cs229-linalg.pdf)
- Calculus – [Calculus Review](https://people.uncw.edu/hermanr/pde1/pdebook/CalcRev.pdf)
- Probability Theory– [Probability Theory Review](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf)
- Statistics – [Statistics Cheat Sheet](https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics)

**Digital Signal Processing**
<div class="highlight">
    <p>Successful completion of the modules <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40700">40700</a> Signale und Systeme, <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=10002">10002</a> Digitale Signalverarbeitung or similar is required.</p>
</div>

Note: If you don't know if your course counts, drop me an email.

---

## Additional Resources

- [Creative Machine Learning Course](https://github.com/acids-ircam/creative_ml) – Prof. Philippe Esling
- [Understanding Deep Learning](https://github.com/udlbook/udlbook) – Simon J.D. Prince
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) – Andrew Ng
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) – Andrew Ng
- [Milestone Paper Overview](https://github.com/faressc/dl4ad/blob/main/extras/milestone_paper_overview.md)

**TU Berlin Modules**

- Machine Learning I/II ([40550](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40550)/[40551](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40551))
- Deep Learning I/II ([41071](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41071)/[41072](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41072))

---

## Brief Machine Learning History

---

## State-of-the-art use cases

---

# Machine Learning

---
