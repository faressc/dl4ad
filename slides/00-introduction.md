<h1>Deep Learning for<br>Audio Data</h1>

Note:
    - Seminar Deep Learning for Audio Data

<!-- .slide: data-state="no-header" -->
---

## Lecturer

<div style="display: flex; align-items: center; gap: 40px;">
    <div style="flex: 0 0 600px;">
        <figure style="text-align: center;">
            <img src="assets/images/00-introduction/profile_picture.jpg" alt="Fares Schulz" style="width: 100%; max-width: 600px; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px;">
            <figcaption>
                <strong>Fares Schulz</strong><br>
                <a href="mailto:fares.schulz@tu-berlin.de" class="small">fares.schulz@tu-berlin.de</a>
            </figcaption>
        </figure>
    </div>
    <div style="flex: 1 0; max-width: 1000px; display: flex; flex-direction: column; text-align: left;">
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

## AI Overview

<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 20px; font-size: 0.85em;">
    <div style="flex: 1; padding-right: 30px;">
        <div style="text-align: center;">
            <img src="assets/images/00-introduction/ai_vs_ml_vs_nn_vs_dl.svg" alt="AI vs ML vs NN vs DL" style="max-width: 80%; height: auto;">
        </div>
        <div style="margin-top: 15px; font-size: 0.8em; color: var(--fs-text-muted-color); font-style: italic; text-align: center;">
            <strong>Nested Relationship:</strong> AI ⊃ ML ⊃ NN ⊃ DL
        </div>
    </div>
    <div style="flex: 1; padding-left: 30px; border-left: 2px solid #ddd;">
        <h4>Hierarchical Relationship</h4>
        <ul>
            <li><strong>Artificial Intelligence (AI):</strong> Machines performing tasks requiring human-like intelligence</li>
            <li><strong>Machine Learning (ML):</strong> Algorithms that learn patterns from data without explicit programming</li>
            <li><strong>Neural Networks (NN):</strong> Interconnected nodes inspired by biological neurons</li>
            <li><strong>Deep Learning (DL):</strong> Uses multi-layered neural networks to model complex patterns</li>
        </ul>
    </div>
</div>

<div class="image-overlay fragment" style="position: absolute; width: 60%; padding: 60px; text-align: center;">
  <p><strong>Deep Learning for Audio Data</strong> = Applications of deep learning techniques to audio data.</p>
</div>

Notes:

- Let's start with a quick overview of artificial intelligence
- With AI, we refer to machines that can perform tasks that typically require human-like intelligence
- Machine learning is a subset of AI that focuses on algorithms that can learn patterns from data without being explicitly programmed
- Neural networks are a specific type of machine learning model inspired by the structure and function of biological neurons
- Deep learning is a subset of neural networks that uses multiple layers to model complex patterns in data

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
            <li>Learning resources: Slides, Jupyter notebooks and more available on the course repository</li>
            <li>Project proposals/selection and organizational info: On ISIS</li>
        </ul>
    </div>
    <div style="flex: 0 0 450px;">
        <figure>
            <div style="text-align: center;">
                <!-- Light theme QR code -->
                <img src="assets/images/00-introduction/repo_qr_code-light.png" alt="Course Repo QR (Light)" style="width: 360px;" class="picture-light">
                <!-- Dark theme QR code -->
                <img src="assets/images/00-introduction/repo_qr_code-light.png" alt="Course Repo QR (Dark)" style="width: 360px;" class="picture-dark">
                <figcaption><a href="https://github.com/faressc/dl4ad" class="small">github.com/faressc/dl4ad</a></figcaption>
            </div>
        </figure>
    </div>
</div>

Note:
    - New branch at the end of the semester

---

## Prerequisites

- Python Knowledge – [Python and Numpy Refresher](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)
- Linear Algebra – [Linear Algebra Review](https://see.stanford.edu/materials/aimlcs229/cs229-linalg.pdf) / [UDL Book Appendix B](https://github.com/udlbook/udlbook)
- Calculus – [Calculus Review](https://people.uncw.edu/hermanr/pde1/pdebook/CalcRev.pdf) / [UDL Book Appendix B](https://github.com/udlbook/udlbook)
- Probability Theory– [Probability Theory Review](https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf) / [UDL Book Appendix C](https://github.com/udlbook/udlbook)
- Statistics – [Statistics Cheat Sheet](https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics) / [UDL Book Appendix C](https://github.com/udlbook/udlbook)

**Formal requirements:**

<div class="highlight">
    <p><em>Signale und Systeme</em> <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40700">40700</a>, <em>Digitale Signalverarbeitung</em> <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=10002">10002</a>, or equivalent</p>
    <p><em>Empirisch-wissenschaftliches Arbeiten</em> <a href="https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=10390">10390</a> or equivalent</p>
</div>

Note: If you don't know if your course counts, drop me an email.

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
            <td>History of Neural Audio Systems</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">26.10.2025</td>
            <td>Course Application Deadline</td>
        </tr>
        <tr>
            <td class="date">30.10.2025</td>
            <td>Machine Learning Fundamentals</td>
        </tr>
        <tr>
            <td class="date">06.11.2025</td>
            <td>Perceptrons</td>
        </tr>
        <tr>
            <td class="date">13.11.2025</td>
            <td>Convolutional and Recurrent Layers</td>
        </tr>
        <tr>
            <td class="date">20.11.2025</td>
            <td>Preprocessing / Training Deep Architectures</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">23.11.2025</td>
            <td>Project Proposal Deadline & Group Selection</td>
        </tr>
        <tr>
            <td class="date">27.11.2025</td>
            <td>Autoencoders / Transformers</td>
        </tr>
        <tr>
            <td class="date">04.12.2025</td>
            <td>Tricks of the Trade</td>
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
            <td>Real-Time Inference</td>
        </tr>
        <tr class="fragment custom appear-table highlight">
            <td class="date">18.12.2025</td>
            <td>Project Pitches (14:00 - 18:00)</td>
        </tr>
        <tr>
            <td class="date">08.01.2026</td>
            <td>Bayesian Inference</td>
        </tr>
        <tr>
            <td class="date">15.01.2026</td>
            <td>Variational Inference</td>
        </tr>
        <tr>
            <td class="date">22.01.2026</td>
            <td>Variational Autoencoder</td>
        </tr>
        <tr>
            <td class="date">29.01.2026</td>
            <td>Adversarial Training</td>
        </tr>
        <tr>
            <td class="date">05.02.2026</td>
            <td>Diffusion Models</td>
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
            <li>Evidence of completion of the formal requirements "Signale und Systeme" or "Digitale Signalverarbeitung" and "Empirisch-wissenschaftliches Arbeiten" (e.g., excerpt from certificate of grades) or equivalent</li>
        </ul>
    </div>
    <div>
        <strong>Note:</strong> The course is limited to 16 participants. Selection will be based on a lottery. After receiving confirmation, enroll in MOSES until 6. November 2025.
    </div>
</div>

---

## Graded Deliverables

**Project Presentation** (1/3 Grade) – *Date:* 12.02.2026

<ul class="small">
    <li>10-minute presentation per group and 5-minute Q&A session</li>
</ul>

**Git repository** (1/3 Grade) – *Deadline:* 31.03.2026

<ul class="small">
    <li>README with clear, step-by-step instructions for running your code</li>
    <li>Environment setup file (e.g., <code>requirements.txt</code>, <code>pyproject.toml</code>)</li>
    <li>Well-commented source code</li>
</ul>

**Project Paper** (1/3 Grade) – *Deadline:* 31.03.2026

<ul class="small">
    <li>Maximum 4 pages</li>
    <li>Use the <a href="https://www.ieee.org/conferences/publishing/templates.html">IEEE conference template</a></li>
</ul>

---

## Overarching Module

- This course is part of the module *Machine Learning and Big Data Processing with Audio and Music* [11005](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/beschreibung/anzeigen.html?number=11005) (6 ECTS)
- The module consists of:
    - *Music Information Retrieval und Machine Learning für Audio* [13702](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13702)
    - *Deep Learning for Audio Data* [13575](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13575)
- AKT students can take this course as part of *Wahlpflichtbereich Vertiefung*

**Grading:**

- Project in Deep Learning for Audio Data (2/3 Grade)
- Lecture exercises in Music Information Retrieval und Machine Learning für Audio (1/3 Grade)

---

## Projects

<div style="display: flex; flex-direction: column; gap: 20px; font-size: 0.9em;">
    <div>
        <strong>Project Proposal Process:</strong>
        <ul>
            <li>Submit your project proposal starting now</li>
            <li>Receive feedback and incorporate it into your proposal</li>
            <li><strong>Project proposal deadline:</strong> 23.11.2025</li>
        </ul>
    </div>
    <div>
        <strong>Group Formation & Registration:</strong>
        <ul>
            <li>4 groups of 4 students</li>
            <li>After the proposal deadline, most appropriate proposals will be selected</li>
            <li>Project proposers are guaranteed a spot in their proposed project group</li>
            <li>Project group selection opens on ISIS after confirmation</li>
            <li><em>First come, first served</em> basis for remaining spots</li>
        </ul>
    </div>
</div>

---

## Additional Resources

- [Creative Machine Learning](https://github.com/acids-ircam/creative_ml) – Prof. Philippe Esling
- [Understanding Deep Learning](https://github.com/udlbook/udlbook) – Simon J.D. Prince
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) – Andrew Ng
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) – Andrew Ng
- [Machine Learning History](https://github.com/faressc/dl4ad/blob/main/extras/machine_learning_history.md)
- [Neural Audio Systems Milestones](https://github.com/faressc/dl4ad/blob/main/extras/neural_audio_systems_milestones.md)

**TU Berlin Modules**

- *Machine Learning I/II* ([40550](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40550)/[40551](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=40551))
- *Deep Learning I/II* ([41071](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41071)/[41072](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/ansehen.html?nummer=41072))

---

<h1 style="margin: 18% 0 120px 0;">Thank You for Listening!</h1>

<div style="text-align: center; margin-top: 50px; font-size: 1.2em; color: var(--fs-text-color);">
<strong>Any Questions?</strong>
</div>
