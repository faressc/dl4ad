from manim import *
import numpy as np


class SigmoidActivationVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-0.2, 1.2, 0.2],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("z", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        # Add y-axis value labels
        y_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), LEFT, buff=0.2)
        y_1_label = MathTex("1", font_size=24).next_to(axes.c2p(0, 1), LEFT, buff=0.2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Write(y_0_label), Write(y_1_label))
        self.wait(1)
        
        # Define sigmoid function
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        # Define sigmoid derivative
        def sigmoid_derivative(z):
            s = sigmoid(z)
            return s * (1 - s)
        
        # Plot sigmoid function in white
        sigmoid_curve = axes.plot(
            sigmoid,
            color=WHITE,
            x_range=[-6, 6],
            stroke_width=4
        )
        
        # Show sigmoid equation
        sigmoid_eq = MathTex(
            r"\sigma(z) = \frac{1}{1 + e^{-z}}",
            font_size=36,
            color=WHITE
        )
        sigmoid_eq.to_edge(UP + LEFT)
        
        self.play(Write(sigmoid_eq))
        self.play(Create(sigmoid_curve))
        self.wait(1)
        
        # Show output range
        range_label = MathTex(
            r"\sigma \in (0, 1)",
            font_size=32,
            color=YELLOW
        )
        range_label.next_to(sigmoid_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(range_label))
        self.wait(1)
        
        # Plot derivative in red
        deriv_curve = axes.plot(
            sigmoid_derivative,
            color=RED,
            x_range=[-6, 6],
            stroke_width=4
        )
        
        # Show derivative equation
        deriv_eq = MathTex(
            r"\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))",
            font_size=36,
            color=RED
        )
        deriv_eq.next_to(range_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_eq))
        self.play(Create(deriv_curve))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class TanhActivationVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1.2, 1.2, 0.5],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("z", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        # Add axis value labels
        x_neg1_label = MathTex("-1", font_size=24).next_to(axes.c2p(-1, 0), DOWN, buff=0.2)
        x_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        x_1_label = MathTex("1", font_size=24).next_to(axes.c2p(1, 0), DOWN, buff=0.2)
        y_neg1_label = MathTex("-1", font_size=24).next_to(axes.c2p(0, -1), LEFT, buff=0.2)
        y_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), LEFT, buff=0.2)
        y_1_label = MathTex("1", font_size=24).next_to(axes.c2p(0, 1), LEFT, buff=0.2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(
            Write(x_neg1_label), Write(x_0_label), Write(x_1_label),
            Write(y_neg1_label), Write(y_0_label), Write(y_1_label)
        )
        self.wait(1)
        
        # Define tanh derivative
        def tanh_derivative(z):
            return 1 - np.tanh(z)**2
        
        # Plot tanh function in white
        tanh_curve = axes.plot(
            np.tanh,
            color=WHITE,
            x_range=[-4, 4],
            stroke_width=4
        )
        
        # Show tanh equation
        tanh_eq = MathTex(
            r"\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}",
            font_size=36,
            color=WHITE
        )
        tanh_eq.to_edge(UP + LEFT)
        
        self.play(Write(tanh_eq))
        self.play(Create(tanh_curve))
        self.wait(1)
        
        # Show output range
        range_label = MathTex(
            r"\tanh \in (-1, 1)",
            font_size=32,
            color=YELLOW
        )
        range_label.next_to(tanh_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(range_label))
        self.wait(1)
        
        # Plot derivative in red
        deriv_curve = axes.plot(
            tanh_derivative,
            color=RED,
            x_range=[-4, 4],
            stroke_width=4
        )
        
        # Show derivative equation
        deriv_eq = MathTex(
            r"\frac{d\tanh}{dz} = 1 - \tanh^2(z)",
            font_size=36,
            color=RED
        )
        deriv_eq.next_to(range_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_eq))
        self.play(Create(deriv_curve))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class ReLUActivationVisualization(Scene):
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
        
        # Plot ReLU function in white (piecewise)
        relu_curve_neg = axes.plot(
            lambda z: 0,
            color=WHITE,
            x_range=[-3, 0],
            stroke_width=4
        )
        relu_curve_pos = axes.plot(
            lambda z: z,
            color=WHITE,
            x_range=[0, 3],
            stroke_width=4
        )
        
        # Show ReLU equation
        relu_eq = MathTex(
            r"\text{ReLU}(z) = \max(0, z)",
            font_size=36,
            color=WHITE
        )
        relu_eq.to_edge(UP + LEFT)
        
        self.play(Write(relu_eq))
        self.play(Create(relu_curve_neg), Create(relu_curve_pos))
        self.wait(1)
        
        # Show output range
        range_label = MathTex(
            r"\text{ReLU} \in [0, \infty)",
            font_size=32,
            color=YELLOW
        )
        range_label.next_to(relu_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(range_label))
        self.wait(1)
        
        # Plot derivative in red (step function)
        deriv_curve_neg = axes.plot(
            lambda z: 0,
            color=RED,
            x_range=[-3, 0],
            stroke_width=4
        )
        deriv_curve_pos = axes.plot(
            lambda z: 1,
            color=RED,
            x_range=[0, 3],
            stroke_width=4
        )
        
        # Show derivative equation
        deriv_eq = MathTex(
            r"\frac{d\text{ReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}",
            font_size=32,
            color=RED
        )
        deriv_eq.next_to(range_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_eq))
        self.play(Create(deriv_curve_neg), Create(deriv_curve_pos))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class LeakyReLUActivationVisualization(Scene):
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
        
        # Define alpha for Leaky ReLU
        alpha = 0.1
        
        # Plot Leaky ReLU function in white (piecewise)
        leaky_relu_curve_neg = axes.plot(
            lambda z: alpha * z,
            color=WHITE,
            x_range=[-3, 0],
            stroke_width=4
        )
        leaky_relu_curve_pos = axes.plot(
            lambda z: z,
            color=WHITE,
            x_range=[0, 3],
            stroke_width=4
        )
        
        # Show Leaky ReLU equation
        leaky_relu_eq = MathTex(
            r"\text{LeakyReLU}(z) = \max(0.1z, z)",
            font_size=36,
            color=WHITE
        )
        leaky_relu_eq.to_edge(UP + LEFT)
        
        self.play(Write(leaky_relu_eq))
        self.play(Create(leaky_relu_curve_neg), Create(leaky_relu_curve_pos))
        self.wait(1)
        
        # Show output range
        range_label = MathTex(
            r"\text{LeakyReLU} \in (-\infty, \infty)",
            font_size=32,
            color=YELLOW
        )
        range_label.next_to(leaky_relu_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(range_label))
        self.wait(1)
        
        # Plot derivative in red (step function with alpha)
        deriv_curve_neg = axes.plot(
            lambda z: alpha,
            color=RED,
            x_range=[-3, 0],
            stroke_width=4
        )
        deriv_curve_pos = axes.plot(
            lambda z: 1,
            color=RED,
            x_range=[0, 3],
            stroke_width=4
        )
        
        # Show derivative equation
        deriv_eq = MathTex(
            r"\frac{d\text{LeakyReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0.1 & \text{otherwise} \end{cases}",
            font_size=32,
            color=RED
        )
        deriv_eq.next_to(range_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(deriv_eq))
        self.play(Create(deriv_curve_neg), Create(deriv_curve_pos))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class CrossEntropyLossVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 5, 1],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = MathTex(r"\hat{y} \text{ (predicted probability)}", font_size=32).next_to(axes.x_axis, DOWN)
        y_label = MathTex(r"\mathcal{L}", font_size=32).next_to(axes.y_axis, LEFT)
        
        # Add x-axis value labels
        x_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        x_05_label = MathTex("0.5", font_size=24).next_to(axes.c2p(0.5, 0), DOWN, buff=0.2)
        x_1_label = MathTex("1", font_size=24).next_to(axes.c2p(1, 0), DOWN, buff=0.2)
        
        # Add y-axis value labels
        y_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), LEFT, buff=0.2)

        axes_group = VGroup(axes, x_label, y_label, x_0_label, x_05_label, x_1_label, y_0_label)
        
        self.play(Create(axes_group))
        self.wait(1)
        
        # Start with logarithm explanation
        log_title = MathTex(
            r"\text{Logarithm: } -\log(\hat{y})",
            font_size=36,
            color=YELLOW
        )
        log_title.to_edge(UP)
        
        self.play(Write(log_title))
        self.wait(1)
        
        # Define negative log function
        def neg_log(x):
            eps = 1e-7
            x = np.clip(x, eps, 1 - eps)
            return -np.log(x)
        
        # Plot negative log in yellow
        neg_log_curve = axes.plot(
            neg_log,
            color=YELLOW,
            x_range=[0.05, 0.99],
            stroke_width=4
        )
        
        self.play(Create(neg_log_curve))
        self.wait(2)
        
        # Fade out log explanation
        self.play(
            FadeOut(log_title),
            FadeOut(neg_log_curve)
        )
        self.wait(0.5)
        
        # Show cross-entropy equation
        ce_eq = MathTex(
            r"\mathcal{L}(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]",
            font_size=32,
            color=WHITE
        )
        ce_eq.to_edge(UP)
        
        self.play(Write(ce_eq))
        self.wait(1)
        
        # Define cross-entropy loss for y=1 (true label is 1)
        def ce_loss_y1(y_pred):
            # Add small epsilon to avoid log(0)
            eps = 1e-7
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.log(y_pred)
        
        # Define cross-entropy loss for y=0 (true label is 0)
        def ce_loss_y0(y_pred):
            eps = 1e-7
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.log(1 - y_pred)
        
        # Plot loss for y=1 (blue curve)
        loss_y1_curve = axes.plot(
            ce_loss_y1,
            color=BLUE,
            x_range=[0.05, 0.99],
            stroke_width=4
        )
        
        # Show label for y=1
        y1_label = MathTex(
            r"y = 1",
            font_size=32,
            color=BLUE
        )
        y1_label.next_to(ce_eq, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Write(y1_label))
        self.play(Create(loss_y1_curve))
        self.wait(1)
        
        # Add annotation for y=1 case
        y1_desc = Tex(
            r"When true label is 1,\\loss increases as $\hat{y} \to 0$",
            font_size=28,
            color=BLUE
        )
        y1_desc.next_to(y1_label, DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(Write(y1_desc))
        self.wait(1)
        
        # Plot loss for y=0 (red curve)
        loss_y0_curve = axes.plot(
            ce_loss_y0,
            color=RED,
            x_range=[0.05, 0.99],
            stroke_width=4
        )
        
        # Show label for y=0
        y0_label = MathTex(
            r"y = 0",
            font_size=32,
            color=RED
        )
        y0_label.next_to(y1_desc, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Write(y0_label))
        self.play(Create(loss_y0_curve))
        self.wait(1)
        
        # Add annotation for y=0 case
        y0_desc = Tex(
            r"When true label is 0,\\loss increases as $\hat{y} \to 1$",
            font_size=28,
            color=RED
        )
        y0_desc.next_to(y0_label, DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(Write(y0_desc))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
