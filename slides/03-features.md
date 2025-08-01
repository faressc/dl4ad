## Features

Navigate with arrow keys or space bar

Press 'S' for speaker view

Press 'ESC' for slide overview

---

## Vertical Slides

This is a vertical slide

Press down arrow to continue

--

### HTML Structure Example

```html
<div class="reveal">
    <div class="slides">
        <section>
            <h1>Slide Title</h1>
            <p>Slide content goes here</p>
        </section>
    </div>
</div>
```

--

### Advanced JavaScript

```javascript
// Reveal.js initialization
Reveal.initialize({
    hash: true,
    transition: 'slide',
    dependencies: [
        { src: 'plugin/highlight/highlight.js' },
        { src: 'plugin/notes/notes.js' }
    ]
});

// Custom event listener
Reveal.addEventListener('slidechanged', function(event) {
    console.log('Slide changed:', event.currentSlide);
});
```

---

## Simple Table Example

<div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.6em;">

<div style="flex: 1; min-width: 40%;">

| Date | Topic |
|------|-------|
| Week&nbsp;1 | Introduction / Machine Learning I |
| Week&nbsp;2 | Machine Learning II |
| Week&nbsp;3 | Time-Frequency Representations |
| Week&nbsp;4 | Convolutional Neural Networks for Audio |
| Week&nbsp;5 | Recurrent Neural Networks for Sequential Data |
| Week&nbsp;6 | Attention Mechanisms and Transformers |
| Week&nbsp;7 | Audio Classification and Recognition |
| Week&nbsp;8 | Speech Processing and Recognition |

</div>

<div style="flex: 1; min-width: 40%;">

| Date | Topic |
|------|-------|
| Week&nbsp;9 | Music Information Retrieval |
| Week&nbsp;10 | Audio Generation and Synthesis |
| Week&nbsp;11 | Self-Supervised Learning for Audio |
| Week&nbsp;12 | Multi-Modal Audio-Visual Learning |
| Week&nbsp;13 | Advanced Architectures and Techniques |
| Week&nbsp;14 | Project Presentations I |
| Week&nbsp;15 | Project Presentations II |
| Week&nbsp;16 | Final Review and Discussion |

</div>

</div>
