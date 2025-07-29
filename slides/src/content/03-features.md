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