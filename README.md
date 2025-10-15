# Deep Learning for Audio Data (DL4AD)

**Course Repository** | TU Berlin | Audio Communication Group  
**Instructor:** Fares Schulz ([fares.schulz@tu-berlin.de](mailto:fares.schulz@tu-berlin.de))<br>
**Teaching Assistant:** Lina Campanella ([l.campanella@tu-berlin.de](mailto:l.campanella@tu-berlin.de))

---

## About

This repository contains all learning resources for **Deep Learning for Audio Data**. The seminar covers the mathematical and algorithmic fundamentals of machine learning, with particular emphasis on deep learning and neural networks for audio data. Core topics include:

- Mathematical and algorithmic foundations of machine learning, deep learning, and neural networks
- Data exploration, preprocessing, and augmentation techniques for audio
- Feature extraction and representation learning for audio signals
- Machine learning models and architectures tailored to audio applications
- Training and evaluation strategies, including relevant metrics and validation approaches
- Model export, deployment, and inference for production environments
- Deep learning workflows, pipelines, and lifecycles – setup, optimization, and best practices

A special emphasis is placed on creative applications of deep learning in the music production domain, exploring how these techniques can be applied to music generation, sound design, audio effects, and other artistic contexts.

You can find more information about the course on [MOSES](https://moseskonto.tu-berlin.de/moses/veranstaltungen/lehrveranstaltungsvorlagen/anzeigen.html?veranstaltungsvorlage=13575).

---

## Repository Structure

```
dl4ad/
├── slides/                          # Lecture slides (Reveal.js presentations)
│   ├── 01-introduction.md
│   ├── 02-history.md
│   ├── 03-machine_learning_fundamentals.md
│   ├── ... (more slides)
│   └── assets/                      # Images, videos, fonts for slides
│
├── notebooks/                       # Jupyter notebooks for practical exercises
│   ── 01_introduction.ipynb
│   ── 02_machine_learning_fundamentals.ipynb
│   ── ... (more notebooks)
│
├── extras/                          # Additional learning resources
│   ├── machine_learning_history.md
│   ├── neural_audio_systems_milestones.md
│   └── animations/                  # Manim animations source code
│
├── scripts/                         # Development scripts
│   ├── build.js                     # Build slides
│   └── dev-server.js                # Development server with hot reload
│
├── requirements.txt                 # Python dependencies
├── package.json                     # Node.js dependencies for slides
└── README.md                        # This file
```

---

## Getting Started

### Prerequisites

#### Required Software

- **Python 3.8+** (recommended: Python 3.13)
- **Node.js 18+** (for viewing slides locally)
- **Git**
- **VS Code** (recommended IDE, but any IDE works)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/faressc/dl4ad.git
cd dl4ad
```

#### 2. Set Up Python Environment

Create and activate a virtual environment:

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Install the requirements for generating animations:

```bash
pip install -r requirements_animations.txt
```

#### 3. (Optional) Set Up Node.js Environment for building/viewing slides locally

Install Node.js dependencies:

```bash
npm install
```

---

## Usage

### Working with Jupyter Notebooks

1. **Open VS Code** in the repository directory
2. **Open a notebook** from the `notebooks/` folder
3. **Select Python interpreter** from your `.venv` environment
4. **Run cells** to execute code

### Viewing Slides

#### Online

The latest slides are hosted online at: [https://faressc.github.io/dl4ad/](https://faressc.github.io/dl4ad/)

#### Build and Serve Slides Locally

Start the development server with hot reload:

```bash
npm start
# or
npm run dev
```

Then open your browser to `http://localhost:8080`

Build static slides:

```bash
npm run build
```

The built slides will be in the `dist/` folder.

### Generating Animations (Optional)

This repository uses [Manim](https://www.manim.community/) for creating mathematical animations.

Generate animations:

```bash
manim extras/animations/03-machine_learning_fundamentals.py
```

Configuration is in `manim.cfg`.

---

## License

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Acknowledgments

### Course Contributors

Course developed by the **Computer Music and Neural Audio Systems Research Team**  
**Audio Communication Group**  
**Technische Universität Berlin**

The following individuals contributed significantly to the course materials:

- [Fares Schulz](https://github.com/faressc)
- [Lina Campanella](https://github.com/linaclca) (Jupyter notebooks)

For more information about our research group, visit the [Audio Communication Group website](https://www.tu.berlin/en/ak/).

### Inspired by

This course has been inspired by the following excellent course:

- [Creative Machine Learning](https://github.com/acids-ircam/creative_ml) by [Philippe Esling](https://github.com/esling)