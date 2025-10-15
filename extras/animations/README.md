# Linear Regression Video Generation

This directory contains Manim scripts for generating the animations

## Installation

Install Manim Community Edition:

```bash
pip install manim
```

## Usage

### Render using the default Cairo CPU renderer:

```bash
manim file_name.py SceneName
```

### Render using the OpenGL GPU renderer:

```bash
manim --renderer=opengl file_name.py SceneName 
```

> Note: The OpenGL renderer requires the flag `--write_to_movie` to save the video file.

### Preview the video after rendering:

```bash
manim -p file_name.py SceneName
```

In opengl renderer mode you can also debug interactively with `self.interactive_embed()` in your scene. But you need to install the `ipython` package first:

```bash
pip install ipython
```

When you reach the line `self.interactive_embed()`, the rendering will pause and you can interact with the scene in an IPython shell. Type `exit` to continue rendering. You can also change the code itself and save the file, then the rendering will start over with the new code.

## Command Options

- `-pql`: Preview in low quality (faster rendering)
- `-pqm`: Preview in medium quality
- `-pqh`: Preview in high quality (production)
- `-p`: Preview the video after rendering
- `-s`: Save the last frame as an image

## Output

The rendered videos will be saved in the `media/videos` directory by default.

## Configuration

You can customize the output directory and other settings by creating a `manim.cfg` file. For more details, refer to the [Manim documentation](https://docs.manim.community/en/stable/).

> Note: Because of some reason the we cannot define the opengl renderer as default in the `manim.cfg` file, so you have to specify the `--renderer=opengl` flag every time you want to use it.
