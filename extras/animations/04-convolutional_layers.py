from manim import *
import numpy as np


class FIRConvolution1D(Scene):
    def construct(self):
        
        # Create discrete audio signal directly
        n_samples = 15
        x_values = np.linspace(0, 4, n_samples)
        signal = np.sin(2 * np.pi * 0.8 * x_values) + 0.3 * np.sin(2 * np.pi * 2.5 * x_values)
        
        # Define FIR filter
        filter_kernel = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
        filter_length = len(filter_kernel)
        
        # Step 1: Show the input signal
        signal_title = Text("Input Signal x[n]", font_size=24, color=BLUE)
        signal_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(signal_title))
        
        # Create axes for signal
        signal_axes = Axes(
            x_range=[0, n_samples - 1, 5],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=2,
            axis_config={"color": WHITE, "include_tip": False}
        )
        signal_axes.shift(UP * 1.2)
        
        signal_label = signal_axes.get_x_axis_label("n", edge=DOWN, direction=DOWN)
        
        self.play(Create(signal_axes), Write(signal_label))
        
        # Plot signal as discrete points
        signal_dots = VGroup()
        signal_stems = VGroup()
        
        for i in range(n_samples):
            y_val = signal[i]
            stem = Line(
                signal_axes.c2p(i, 0),
                signal_axes.c2p(i, y_val),
                color=BLUE,
                stroke_width=2
            )
            signal_stems.add(stem)
            
            dot = Dot(signal_axes.c2p(i, y_val), color=BLUE, radius=0.06)
            signal_dots.add(dot)
        
        self.play(
            LaggedStart(*[Create(stem) for stem in signal_stems], lag_ratio=0.05),
            LaggedStart(*[Create(dot) for dot in signal_dots], lag_ratio=0.05)
        )
        self.wait(1)

        # Step 2: Show the FIR filter
        filter_title = Text("FIR Filter h[k]", font_size=24, color=RED)
        filter_title.next_to(signal_axes, DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(Write(filter_title))
        
        # Create axes for filter
        filter_axes = Axes(
            x_range=[-0.5, filter_length - 0.5, 1],
            y_range=[0, 0.35, 0.1],
            x_length=3.5,
            y_length=1.2,
            axis_config={"color": WHITE, "include_tip": False},
        )
        filter_axes.next_to(signal_axes, DOWN, buff=1.2).shift(LEFT * 3)
        
        # Plot filter as bars
        bars = VGroup()
        for i, h in enumerate(filter_kernel):
            bar = Rectangle(
                width=0.35,
                height=h * 3.4,
                color=RED,
                fill_opacity=0.7,
                stroke_width=2
            )
            bar.move_to(filter_axes.c2p(i, h / 2))
            bars.add(bar)
        
        filter_label = filter_axes.get_x_axis_label("k", edge=DOWN, direction=DOWN)
        
        # Show filter equation
        filter_eq = MathTex(
            r"h[k] = [0.1, 0.25, 0.3, 0.25, 0.1]",
            font_size=18,
            color=RED
        )
        filter_eq.next_to(filter_axes, DOWN*2, buff=0.3)
        
        self.play(Create(filter_axes), Write(filter_label))
        self.play(LaggedStart(*[FadeIn(bar, shift=UP*0.3) for bar in bars], lag_ratio=0.2))
        self.play(Write(filter_eq))
        self.wait(1)

        # Step 3: Show convolution equation
        conv_formula = MathTex(
            r"y[n] = \sum_{k=0}^{4} h[k] \cdot x[n - k]",
            font_size=24,
            color=GREEN
        )
        conv_formula.next_to(filter_eq, RIGHT, buff=1)
        self.play(Write(conv_formula))
        self.wait(2)
        
        # Step 4: Flip the filter
        self.play(
            FadeOut(signal_title),
            FadeOut(filter_title)
        )

        flip_title = Text("Step 1: Flip the filter", font_size=24, color=YELLOW)
        flip_title.to_edge(UP)
        self.play(Write(flip_title))

        # Show flipped filter values
        flipped_filter = filter_kernel[::-1]
        filter_flipped_eq = MathTex(
            r"h[-k] = [0.1, 0.25, 0.3, 0.25, 0.1]",
            font_size=18,
            color=ORANGE
        )
        filter_flipped_eq.next_to(filter_eq, DOWN, buff=0.2)

        self.play(Write(filter_flipped_eq))
        self.wait(1)
        
        # Move filter and equations to right corner
        filter_group = VGroup(filter_axes, filter_label, bars, filter_eq, filter_flipped_eq)
        target_position = RIGHT * 5.5 + UP * 2.5
        
        self.play(
            FadeOut(filter_eq, filter_flipped_eq),
            filter_group.animate.scale(0.7).move_to(target_position),
            run_time=1.5
        )
        self.wait(1)
        
        # Step 5: Slide the filter and compute convolution
        self.play(FadeOut(flip_title))
        
        slide_title = Text("Step 2: Slide filter and compute", font_size=24, color=YELLOW)
        slide_title.to_edge(UP)
        self.play(Write(slide_title))
        
        # Create output axes
        output_axes = Axes(
            x_range=[0, n_samples - 1, 5],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=2,
            axis_config={"color": WHITE, "include_tip": False},
        )
        output_axes.shift(DOWN * 1.8)
        
        output_label = output_axes.get_x_axis_label("n", edge=DOWN, direction=DOWN)
        y_label = MathTex("y[n]", font_size=24, color=GREEN)
        y_label.next_to(output_axes, LEFT)
        
        self.play(Create(output_axes), Write(output_label), Write(y_label))
        self.wait(1)
        
        # Compute full convolution
        output_signal = np.convolve(signal, filter_kernel, mode='same')
        
        # Animate sliding the filter
        output_dots = VGroup()
        output_stems = VGroup()
        
        m_start = filter_length // 2
        m_end = n_samples - filter_length // 2
        
        for m in range(m_start, m_end):
            # Highlight the current window being processed
            window_highlights = VGroup()
            
            for n in range(filter_length):
                idx = m - filter_length // 2 + n
                if 0 <= idx < n_samples:
                    highlight = Circle(
                        radius=0.12,
                        color=YELLOW,
                        stroke_width=3,
                        fill_opacity=0.2,
                        fill_color=YELLOW
                    ).move_to(signal_axes.c2p(idx, signal[idx]))
                    window_highlights.add(highlight)
            
            # Show current position indicator
            position_label = MathTex(f"n = {m}", font_size=18, color=YELLOW)
            position_label.next_to(signal_axes, RIGHT, buff=0.5)
            
            # Create output point
            y_val = output_signal[m]
            output_stem = Line(
                output_axes.c2p(m, 0),
                output_axes.c2p(m, y_val),
                color=GREEN,
                stroke_width=2
            )
            output_dot = Dot(output_axes.c2p(m, y_val), color=GREEN, radius=0.06)
            
            if m == m_start:
                # First iteration - show everything
                self.play(
                    FadeIn(window_highlights),
                    Write(position_label),
                    run_time=0.6
                )
                self.wait(0.3)
                self.play(
                    Create(output_stem),
                    Create(output_dot),
                    run_time=0.4
                )
                output_stems.add(output_stem)
                output_dots.add(output_dot)
                self.wait(0.3)
                self.play(FadeOut(window_highlights), FadeOut(position_label), run_time=0.3)
            else:
                # Subsequent iterations - faster
                self.play(
                    FadeIn(window_highlights),
                    Write(position_label),
                    run_time=0.15
                )
                self.add(output_stem, output_dot)
                output_stems.add(output_stem)
                output_dots.add(output_dot)
                self.play(FadeOut(window_highlights), FadeOut(position_label), run_time=0.15)
        
        self.wait(1)
        
        # Clean up
        self.play(FadeOut(slide_title))
        
        # Step 6: Show the result is smoothed
        result_title = Text("Result: Low-pass Filtered Signal", font_size=24, color=GREEN)
        result_title.to_edge(UP)
        self.play(Write(result_title))
        self.wait(1)
        
        # Add comparison labels
        input_label = Text("Input: High + Low frequencies", font_size=18, color=BLUE)
        input_label.next_to(signal_axes, DOWN, buff=0.8)
        
        output_note = Text("Output: High frequencies removed", font_size=18, color=GREEN)
        output_note.next_to(output_axes, DOWN, buff=0.8)
        
        self.play(FadeOut(conv_formula))
        self.play(Write(input_label), Write(output_note))
        self.wait(2)
        
        # Show the smoothing effect with annotation
        smoothing_text = Text("The filter smooths the signal - lowpass filter", font_size=20, color=YELLOW)
        smoothing_text.to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(smoothing_text))
        self.wait(3)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
