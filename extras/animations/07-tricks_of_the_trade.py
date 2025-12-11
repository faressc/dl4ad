from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
import numpy as np

class GELUActivationVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3.5, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("z", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        # Add axis value labels
        x_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        x_1_label = MathTex("1", font_size=24).next_to(axes.c2p(1, 0), DOWN, buff=0.2)
        y_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), LEFT, buff=0.2)
        y_1_label = MathTex("1", font_size=24).next_to(axes.c2p(0, 1), LEFT, buff=0.2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Write(x_0_label), Write(x_1_label), Write(y_0_label), Write(y_1_label))
        self.wait(1)
        
        # Define GELU function (approximation)
        def gelu(z):
            return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        
        # Define GELU derivative (numerical approximation)
        def gelu_derivative(z):
            eps = 1e-7
            return (gelu(z + eps) - gelu(z - eps)) / (2 * eps)
        
        # Plot GELU function in white
        gelu_curve = axes.plot(
            gelu,
            color=WHITE,
            x_range=[-3, 3],
            stroke_width=4
        )
        
        # Show GELU equation
        gelu_eq = MathTex(
            r"\text{GELU}(z) = z \cdot \Phi(z)",
            font_size=36,
            color=WHITE
        )
        gelu_eq.to_edge(UP + LEFT)
        
        self.play(Write(gelu_eq))
        self.play(Create(gelu_curve))
        self.wait(1)
        
        # Show approximation
        approx_label = MathTex(
            r"\approx 0.5z(1 + \tanh[\sqrt{2/\pi}(z + 0.044715z^3)])",
            font_size=28,
            color=YELLOW
        )
        approx_label.next_to(gelu_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(approx_label))
        self.wait(1)
        
        # Plot derivative in red
        deriv_curve = axes.plot(
            gelu_derivative,
            color=RED,
            x_range=[-3, 3],
            stroke_width=4
        )
        
        # Show derivative label
        deriv_label = MathTex(
            r"\frac{d\text{GELU}}{dz}",
            font_size=36,
            color=RED
        )
        deriv_label.next_to(approx_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_label))
        self.play(Create(deriv_curve))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class SwishActivationVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3.5, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("z", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        # Add axis value labels
        x_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        x_1_label = MathTex("1", font_size=24).next_to(axes.c2p(1, 0), DOWN, buff=0.2)
        y_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), LEFT, buff=0.2)
        y_1_label = MathTex("1", font_size=24).next_to(axes.c2p(0, 1), LEFT, buff=0.2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Write(x_0_label), Write(x_1_label), Write(y_0_label), Write(y_1_label))
        self.wait(1)
        
        # Define sigmoid function
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        # Define Swish/SiLU function
        def swish(z):
            return z * sigmoid(z)
        
        # Define Swish derivative
        def swish_derivative(z):
            s = sigmoid(z)
            return swish(z) + s * (1 - swish(z))
        
        # Plot Swish function in white
        swish_curve = axes.plot(
            swish,
            color=WHITE,
            x_range=[-3, 3],
            stroke_width=4
        )
        
        # Show Swish equation
        swish_eq = MathTex(
            r"\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}",
            font_size=36,
            color=WHITE
        )
        swish_eq.to_edge(UP + LEFT)
        
        self.play(Write(swish_eq))
        self.play(Create(swish_curve))
        self.wait(1)
        
        # Show alternative name
        silu_label = MathTex(
            r"\text{Also known as SiLU (Sigmoid Linear Unit)}",
            font_size=28,
            color=YELLOW
        )
        silu_label.next_to(swish_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(silu_label))
        self.wait(1)
        
        # Plot derivative in red
        deriv_curve = axes.plot(
            swish_derivative,
            color=RED,
            x_range=[-3, 3],
            stroke_width=4
        )
        
        # Show derivative equation
        deriv_eq = MathTex(
            r"\frac{d\text{Swish}}{dz} = \text{Swish}(z) + \sigma(z)(1 - \text{Swish}(z))",
            font_size=28,
            color=RED
        )
        deriv_eq.next_to(silu_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_eq))
        self.play(Create(deriv_curve))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class SoftmaxActivationVisualization(Scene):
    def construct(self):
        # Title
        title = MathTex(
            r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}",
            font_size=40,
            color=WHITE
        )
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Example input vector
        input_label = Text("Input logits:", font_size=28, color=YELLOW)
        input_label.next_to(title, DOWN, buff=0.8)
        input_label.to_edge(LEFT, buff=1)
        
        z_values = [2.0, 1.0, 0.1]
        z_labels = VGroup(*[
            MathTex(f"z_{i+1} = {z:.1f}", font_size=32)
            for i, z in enumerate(z_values)
        ])
        z_labels.arrange(0.2 * DOWN, aligned_edge=LEFT, buff=0.3)
        z_labels.next_to(input_label, 0.2 * DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(input_label))
        self.play(Write(z_labels))
        self.wait(1)
        
        # Compute softmax
        exp_values = [np.exp(z) for z in z_values]
        sum_exp = sum(exp_values)
        softmax_values = [e / sum_exp for e in exp_values]
        
        # Show exponentials
        exp_label = Text("Exponentials:", font_size=28, color=YELLOW)
        exp_label.next_to(z_labels, 0.2 * DOWN, buff=0.8)
        exp_label.align_to(input_label, LEFT)
        
        exp_texts = VGroup(*[
            MathTex(f"e^{{z_{i+1}}} = {e:.2f}", font_size=32)
            for i, e in enumerate(exp_values)
        ])
        exp_texts.arrange(0.2 * DOWN, aligned_edge=LEFT, buff=0.3)
        exp_texts.next_to(exp_label, 0.2 * DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(exp_label))
        self.play(Write(exp_texts))
        self.wait(1)
        
        # Show sum
        sum_label = MathTex(
            f"\\sum_j e^{{z_j}} = {sum_exp:.2f}",
            font_size=32,
            color=RED
        )
        sum_label.next_to(exp_texts, 0.2 * DOWN, buff=0.5)
        sum_label.align_to(input_label, LEFT)
        
        self.play(Write(sum_label))
        self.wait(1)
        
        # Show softmax outputs on the right side
        output_label = Text("Softmax outputs:", font_size=28, color=YELLOW)
        output_label.to_edge(RIGHT, buff=1)
        output_label.align_to(input_label, UP)
        
        softmax_texts = VGroup(*[
            MathTex(f"p_{i+1} = {p:.3f}", font_size=32)
            for i, p in enumerate(softmax_values)
        ])
        softmax_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        softmax_texts.next_to(output_label, DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(Write(output_label))
        self.play(Write(softmax_texts))
        self.wait(1)
        
        # Show probability sum
        prob_sum = MathTex(
            f"\\sum_i p_i = {sum(softmax_values):.3f} = 1",
            font_size=32,
            color=GREEN
        )
        prob_sum.next_to(softmax_texts, DOWN, buff=0.5)
        prob_sum.align_to(output_label, LEFT)
        
        self.play(Write(prob_sum))
        self.wait(1)
        
        # Create bar chart visualization
        bar_chart_group = VGroup()
        
        # Clear some space
        self.play(
            FadeOut(z_labels), FadeOut(exp_texts), FadeOut(exp_label), 
            FadeOut(sum_label), FadeOut(input_label)
        )
        self.wait(0.5)
        
        # Create bars
        chart_title = Text("Probability Distribution", font_size=28, color=YELLOW)
        chart_title.to_edge(LEFT, buff=1)
        chart_title.move_to([0, 1.5, 0])
        
        bar_width = 1.2
        bar_spacing = 2.0
        max_height = 3.5
        
        bars = VGroup()
        bar_labels = VGroup()
        value_labels = VGroup()
        
        for i, (p, z) in enumerate(zip(softmax_values, z_values)):
            # Create bar
            bar = Rectangle(
                width=bar_width,
                height=p * max_height,
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=2
            )
            x_pos = -2.5 + i * bar_spacing
            bar.move_to([x_pos, p * max_height / 2 - 2, 0])
            
            # Class label
            class_label = MathTex(f"\\text{{Class}} \\ {i+1}", font_size=24)
            class_label.next_to(bar, DOWN, buff=0.2)
            
            # Value label
            value_label = MathTex(f"{p:.3f}", font_size=24, color=YELLOW)
            value_label.next_to(bar, UP, buff=0.1)
            
            bars.add(bar)
            bar_labels.add(class_label)
            value_labels.add(value_label)
        
        self.play(Write(chart_title))
        self.play(Create(bars), Write(bar_labels))
        self.play(Write(value_labels))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class StepDecaySchedule(Scene):
    def construct(self):
        # Title
        title = Text("Step Decay Learning Rate Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Step decay parameters
        initial_lr = 0.1
        drop_factor = 0.5
        epochs_drop = 25
        
        # Create step decay visualization with horizontal line segments
        step_lines = VGroup()
        num_steps = int(100 / epochs_drop)
        
        for i in range(num_steps + 1):
            lr = initial_lr * (drop_factor ** i)
            start_epoch = i * epochs_drop
            end_epoch = min((i + 1) * epochs_drop, 100)
            
            # Horizontal line for this step
            line = Line(
                axes.c2p(start_epoch, lr),
                axes.c2p(end_epoch, lr),
                color=BLUE,
                stroke_width=4
            )
            step_lines.add(line)
        
        # Show equation
        equation = MathTex(
            r"\eta_t = \eta_0 \times \gamma^{\lfloor t / T \rfloor}",
            font_size=32,
            color=YELLOW
        )
        equation.next_to(title, DOWN, buff=0.5)
        
        params = MathTex(
            r"\eta_0 = 0.1, \quad \gamma = 0.5, \quad T = 25",
            font_size=28,
            color=GREEN
        )
        params.next_to(equation, DOWN, buff=0.3)
        
        self.play(Write(equation))
        self.play(Write(params))
        self.play(Create(step_lines), run_time=2)
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class LinearDecaySchedule(Scene):
    def construct(self):
        # Title
        title = Text("Linear Decay Learning Rate Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Linear decay parameters
        initial_lr = 0.1
        final_lr = 0.01
        total_epochs = 100
        
        # Create linear decay function
        def linear_decay(epoch):
            return initial_lr - (initial_lr - final_lr) * (epoch / total_epochs)
        
        # Plot the schedule
        linear_curve = axes.plot(
            linear_decay,
            x_range=[0, total_epochs],
            color=BLUE,
            stroke_width=4
        )
        
        # Show equation
        equation = MathTex(
            r"\eta_t = \eta_0 - \frac{(\eta_0 - \eta_{min}) \cdot t}{T}",
            font_size=32,
            color=YELLOW
        )
        equation.next_to(title, DOWN, buff=0.5)
        
        params = MathTex(
            r"\eta_0 = 0.1, \quad \eta_{min} = 0.01, \quad T = 100",
            font_size=28,
            color=GREEN
        )
        params.next_to(equation, DOWN, buff=0.3)
        
        self.play(Write(equation))
        self.play(Write(params))
        self.play(Create(linear_curve), run_time=2)
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class ExponentialDecaySchedule(Scene):
    def construct(self):
        # Title
        title = Text("Exponential Decay Learning Rate Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Exponential decay parameters
        initial_lr = 0.1
        decay_rate = 0.95
        
        # Create exponential decay function
        def exponential_decay(epoch):
            return initial_lr * (decay_rate ** epoch)
        
        # Plot the schedule
        exp_curve = axes.plot(
            exponential_decay,
            x_range=[0, 100],
            color=BLUE,
            stroke_width=4
        )
        
        # Show equation
        equation = MathTex(
            r"\eta_t = \eta_0 \times \gamma^t",
            font_size=32,
            color=YELLOW
        )
        equation.next_to(title, DOWN, buff=0.5)
        
        params = MathTex(
            r"\eta_0 = 0.1, \quad \gamma = 0.95",
            font_size=28,
            color=GREEN
        )
        params.next_to(equation, DOWN, buff=0.3)
        
        self.play(Write(equation))
        self.play(Write(params))
        self.play(Create(exp_curve), run_time=2)
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class CosineAnnealingSchedule(Scene):
    def construct(self):
        # Title
        title = Text("Cosine Annealing Learning Rate Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Cosine annealing parameters
        eta_max = 0.1
        eta_min = 0.01
        T_max = 100
        
        # Create cosine annealing function
        def cosine_annealing(epoch):
            return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
        
        # Plot the schedule
        cosine_curve = axes.plot(
            cosine_annealing,
            x_range=[0, T_max],
            color=BLUE,
            stroke_width=4
        )
        
        # Show equation
        equation = MathTex(
            r"\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)",
            font_size=28,
            color=YELLOW
        )
        equation.next_to(title, DOWN, buff=0.5)
        
        params = MathTex(
            r"\eta_{max} = 0.1, \quad \eta_{min} = 0.01, \quad T = 100",
            font_size=28,
            color=GREEN
        )
        params.next_to(equation, DOWN, buff=0.3)
        
        self.play(Write(equation))
        self.play(Write(params))
        self.play(Create(cosine_curve), run_time=2)
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class WarmRestartsSchedule(Scene):
    def construct(self):
        # Title
        title = Text("Warm Restarts (SGDR) Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Warm restarts parameters
        eta_max = 0.1
        eta_min = 0.01
        T_0 = 25  # Initial cycle length
        T_mult = 1  # Keep cycle length constant for simplicity
        
        # Create warm restarts function for a single cycle
        def cosine_cycle(t_cur, T_i):
            return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t_cur / T_i))
        
        # Plot each cycle separately to avoid interpolation at restart points
        restart_curves = VGroup()
        num_cycles = 4
        
        for i in range(num_cycles):
            cycle_start = i * T_0
            cycle_end = min((i + 1) * T_0, 100)
            
            # Create function for this specific cycle
            def cycle_func(t):
                t_cur = t - cycle_start
                return cosine_cycle(t_cur, T_0)
            
            cycle_curve = axes.plot(
                cycle_func,
                x_range=[cycle_start, cycle_end],
                color=BLUE,
                stroke_width=4
            )
            restart_curves.add(cycle_curve)
        
        # Show equation
        equation = MathTex(
            r"\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi T_{cur}}{T_i}\right)\right)",
            font_size=26,
            color=YELLOW
        )
        equation.next_to(title, DOWN, buff=0.5)
        
        params = MathTex(
            r"\eta_{max} = 0.1, \quad \eta_{min} = 0.01, \quad T_0 = 25",
            font_size=28,
            color=GREEN
        )
        params.next_to(equation, DOWN, buff=0.3)
        
        self.play(Write(equation))
        self.play(Write(params))
        self.play(Create(restart_curves), run_time=2)
        self.wait(1)
        
        # Add markers at restart points
        restart_epochs = [25, 50, 75]
        restart_markers = VGroup()
        for epoch in restart_epochs:
            lr_value = eta_max  # Learning rate resets to max at each restart
            marker = Dot(axes.c2p(epoch, lr_value), color=RED, radius=0.08)
            restart_markers.add(marker)
        
        marker_label = Text("Restart points (save snapshots)", font_size=24, color=RED)
        marker_label.next_to(params, DOWN, buff=0.3)
        
        self.play(Create(restart_markers))
        self.play(Write(marker_label))
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class OneCyclePolicySchedule(Scene):
    def construct(self):
        # Title
        title = Text("One Cycle Policy Learning Rate Schedule", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.12, 0.02],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Labels
        x_label = axes.get_x_axis_label("\\text{Step } t", edge=DOWN, direction=DOWN)
        y_label = axes.get_y_axis_label("\\text{Learning Rate } \\eta", edge=LEFT, direction=LEFT).rotate(90 * DEGREES).shift(LEFT * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # One cycle parameters
        eta_max = 0.1
        eta_min = 0.01
        total_epochs = 100
        pct_start = 0.3  # 30% for warmup
        
        # Create one cycle function
        def one_cycle(epoch):
            if epoch < pct_start * total_epochs:
                # Warmup phase: linear increase
                return eta_min + (eta_max - eta_min) * (epoch / (pct_start * total_epochs))
            else:
                # Annealing phase: cosine decay
                progress = (epoch - pct_start * total_epochs) / ((1 - pct_start) * total_epochs)
                return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))
        
        # Plot the schedule using axes.plot for better compatibility
        cycle_curve = axes.plot(
            one_cycle,
            x_range=[0, 100],
            color=BLUE,
            stroke_width=4
        )
        
        # Show phases
        warmup_label = Text("Warmup", font_size=24, color=GREEN)
        warmup_label.next_to(axes.c2p(15, 0.11), UP, buff=0.1)
        
        anneal_label = Text("Cosine Annealing", font_size=24, color=ORANGE)
        anneal_label.next_to(axes.c2p(65, 0.07), UP, buff=0.1)
        
        description = Text("Warmup (30%) → Cosine Decay (70%)", font_size=28, color=YELLOW)
        description.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(description))
        self.play(Create(cycle_curve), run_time=2)
        self.play(Write(warmup_label), Write(anneal_label))
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

