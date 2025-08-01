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

- Theory and application of deep learning techniques for audio data

<h3>Covers:</h3>

- Mathematical and algorithmic fundamentals of machine learning with focus on deep learning and neural networks
- Various architectures and methods for processing, generating and analyzing audio signals

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
            <li>Jupyter Notebooks: Practical implementations</li>
            <li>Projects: Apply concepts to real-world, hands-on audio applications</li>
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

---

## Dates and Topics

<div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.6em;">

<div style="flex: 1; min-width: 40%;">

<table>
    <thead>
        <tr>
            <th>Date</th>
            <th>Topic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Week&nbsp;1</td>
            <td>Introduction / Machine Learning I</td>
        </tr>
        <tr>
            <td>Week&nbsp;2</td>
            <td>Machine Learning II</td>
        </tr>
        <tr>
            <td>Week&nbsp;3</td>
            <td>Time-Frequency Representations</td>
        </tr>
        <tr>
            <td>Week&nbsp;4</td>
            <td>Convolutional Neural Networks for Audio</td>
        </tr>
        <tr>
            <td>Week&nbsp;5</td>
            <td>Recurrent Neural Networks for Sequential Data</td>
        </tr>
        <tr>
            <td>Week&nbsp;6</td>
            <td>Attention Mechanisms and Transformers</td>
        </tr>
        <tr>
            <td>Week&nbsp;7</td>
            <td>Audio Classification and Recognition</td>
        </tr>
        <tr>
            <td>Week&nbsp;8</td>
            <td>Speech Processing and Recognition</td>
        </tr>
    </tbody>
</table>

</div>

<div style="flex: 1; min-width: 40%;">

<table>
    <thead>
        <tr>
            <th>Date</th>
            <th>Topic</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Week&nbsp;9</td>
            <td>Music Information Retrieval</td>
        </tr>
        <tr>
            <td>Week&nbsp;10</td>
            <td>Audio Generation and Synthesis</td>
        </tr>
        <tr>
            <td>Week&nbsp;11</td>
            <td>Self-Supervised Learning for Audio</td>
        </tr>
        <tr>
            <td>Week&nbsp;12</td>
            <td>Multi-Modal Audio-Visual Learning</td>
        </tr>
        <tr>
            <td>Week&nbsp;13</td>
            <td>Advanced Architectures and Techniques</td>
        </tr>
        <tr>
            <td>Week&nbsp;14</td>
            <td>Project Presentations I</td>
        </tr>
        <tr>
            <td>Week&nbsp;15</td>
            <td>Project Presentations II</td>
        </tr>
        <tr>
            <td>Week&nbsp;16</td>
            <td>Final Review and Discussion</td>
        </tr>
    </tbody>
</table>

</div>

</div>

---

## Assessment

---

## Modulebeschreibung

---

## Prerequisites

---

## Additional Resources

---

## Deep Learning History

---

## State-of-the-art use cases

---

# Machine Learning

---
