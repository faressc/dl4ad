from manim import *
from manim.opengl import *
import numpy as np


class LinearRegressionSimple(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 25, 5],
            x_length=7,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP)
        
        axes_group = VGroup(axes, x_label, y_label)
        axes_group.shift(RIGHT * 1.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # Generate sample data points (y = 2x + 3 with noise)
        np.random.seed(42)
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_true = 2 * x_data + 3
        y_data = y_true + np.random.normal(0, 1.5, len(x_data))
        
        # Create dots for data points
        dots = VGroup()
        for x, y in zip(x_data, y_data):
            dot = Dot(axes.c2p(x, y), color=WHITE, radius=0.08)
            dots.add(dot)
        
        # Animate data points appearing one by one
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.2))
        self.wait(1)
        
        # Show equation form
        definition = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_1", font_size=36)
        definition.to_edge(LEFT).shift(UP * 1.5)
        self.play(Write(definition))
        equation = MathTex(r"y = \theta_0 + \theta_1 x", font_size=36)
        equation.next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(equation))
        self.wait(1)
        
        # Try a few "bad" lines first
        candidate_lines = [
            (5, 0.5, RED),
            (1, 3, ORANGE),
            (8, 1.5, PURPLE),
        ]
        
        for theta_0, theta_1, color in candidate_lines:
            line = axes.plot(lambda x: theta_0 + theta_1 * x, color=color, x_range=[0, 10])
            params = MathTex(f"\\theta_0={theta_0}, \\theta_1={theta_1}", font_size=36, color=color)
            params.next_to(equation, DOWN, aligned_edge=LEFT)
            
            self.play(Create(line), Write(params))
            self.wait(0.5)
            self.play(FadeOut(line), FadeOut(params))
        
        # Calculate actual best fit using least squares
        A = np.vstack([np.ones(len(x_data)), x_data]).T
        theta_0_best, theta_1_best = np.linalg.lstsq(A, y_data, rcond=None)[0]
        
        # Show the best fit line
        best_fit_line = axes.plot(
            lambda x: theta_0_best + theta_1_best * x,
            color=GREEN,
            x_range=[0, 10]
        )
        
        best_params = MathTex(
            f"\\theta_0={theta_0_best:.2f}, \\theta_1={theta_1_best:.2f}",
            font_size=36,
            color=GREEN
        )
        best_params.next_to(equation, DOWN, aligned_edge=LEFT)
        
        best_label = Text("Best Fit!", font_size=32, color=GREEN)
        best_label.next_to(best_params, DOWN, aligned_edge=LEFT)
        
        self.play(Create(best_fit_line), Write(best_params))
        self.play(Write(best_label))
        self.wait(1)

        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class BinaryClassificationSimple(Scene):
    def construct(self):
        
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=7,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False,
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x_1", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("x_2", edge=UP, direction=UP)
        
        axes_group = VGroup(axes, x_label, y_label)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # Generate sample data points for two classes
        np.random.seed(42)
        
        # Class 0 (Blue) - centered around (-2, -1)
        n_samples = 15
        class_0_x = np.random.normal(-2, 0.8, n_samples)
        class_0_y = np.random.normal(-1, 0.8, n_samples)
        
        # Class 1 (Red) - centered around (2, 1)
        class_1_x = np.random.normal(2, 0.8, n_samples)
        class_1_y = np.random.normal(1, 0.8, n_samples)
        
        # Create dots for class 0 (Blue circles)
        dots_class_0 = VGroup()
        for x, y in zip(class_0_x, class_0_y):
            dot = Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            dots_class_0.add(dot)
        
        # Create dots for class 1 (Red crosses)
        dots_class_1 = VGroup()
        for x, y in zip(class_1_x, class_1_y):
            dot = Dot(axes.c2p(x, y), color=RED, radius=0.08)
            dots_class_1.add(dot)
        
        # Animate data points appearing
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots_class_0], lag_ratio=0.1))
        self.wait(0.5)
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots_class_1], lag_ratio=0.1))
        self.wait(1)
        
        # Add legend
        legend_class_0 = VGroup(
            Dot(color=BLUE, radius=0.08),
            Text("Class 0", font_size=24)
        ).arrange(RIGHT, buff=0.2)
        
        legend_class_1 = VGroup(
            Dot(color=RED, radius=0.08),
            Text("Class 1", font_size=24)
        ).arrange(RIGHT, buff=0.2)
        
        legend = VGroup(legend_class_0, legend_class_1).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_edge(RIGHT).shift(UP * 2)
        
        self.play(FadeIn(legend))
        self.wait(1)
        
        # Show equation form
        definition = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_1^{(2)}", font_size=36)
        definition.to_edge(LEFT).shift(UP * 1.5)
        self.play(Write(definition))
        equation = MathTex(r"y = \theta_0 + \theta_1 x_1 + \theta_2 x_2", font_size=36)
        equation.next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(equation))
        # Show classification rule
        classification = MathTex(
            r"\text{Class } = \begin{cases} 0 & \text{if } y \leq 0 \\ 1 & \text{if } y > 0 \end{cases}",
            font_size=36
        )
        classification.next_to(equation, DOWN*6, aligned_edge=LEFT)
        self.play(Write(classification))
        self.wait(0.5)

        self.interactive_embed()

        # Try a few "bad" decision boundaries first
        candidate_boundaries = [
            (-2, -0.5, 0.3, ORANGE),   # (theta_0, theta_1, theta_2, color)
            (1, 0.8, -0.2, PURPLE),
            (0, -1, 1, YELLOW),
        ]
        
        for theta_0, theta_1, theta_2, color in candidate_boundaries:
            # Decision boundary is where theta_0 + theta_1*x1 + theta_2*x2 = 0
            # Solving for x2: x2 = -(theta_0 + theta_1*x1) / theta_2
            if abs(theta_2) > 0.01:  # Avoid division by zero
                boundary = axes.plot(
                    lambda x: -(theta_0 + theta_1 * x) / theta_2,
                    color=color,
                    x_range=[-5, 5]
                )
            else:  # Vertical line
                boundary = axes.get_vertical_line(axes.c2p(-theta_0/theta_1, 0), color=color)
            
            params = MathTex(
                f"\\theta_0={theta_0:.1f}, \\theta_1={theta_1:.1f}, \\theta_2={theta_2:.1f}",
                font_size=36,
                color=color
            )

            params.to_edge(RIGHT+DOWN*3.5)
            
            self.play(Create(boundary), Write(params))
            self.wait(0.5)
            self.play(FadeOut(boundary), FadeOut(params))
        

        # Calculate actual decision boundary using simple approach
        # Simple approach: find the line that separates the class centers
        theta_0_best = 0.0
        theta_1_best = 3.0
        theta_2_best = 2.0
        
        # Show the best decision boundary
        best_boundary = axes.plot(
            lambda x: -(theta_0_best + theta_1_best * x) / theta_2_best,
            color=GREEN,
            x_range=[-5, 5],
            stroke_width=4
        )
        
        best_params = MathTex(
            f"\\theta_0={theta_0_best:.1f}, \\theta_1={theta_1_best:.1f}, \\theta_2={theta_2_best:.1f}",
            font_size=32,
            color=GREEN
        )
        best_params.to_edge(RIGHT+DOWN*3.5)
        
        best_label = Text("Best Boundary!", font_size=36, color=GREEN)
        best_label.next_to(best_params, DOWN, aligned_edge=LEFT)
        
        self.play(Create(best_boundary), Write(best_params))
        self.play(Write(best_label))
        self.wait(1)

        self.interactive_embed()  # For debugging and interaction

        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)



