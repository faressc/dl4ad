from manim import *
from manim.opengl import *
import numpy as np


class LinearRegressionSimple(Scene):
    def construct(self):
        # Title
        title = Text("Linear Regression Example", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        self.interactive_embed()
        
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
        axes_group.shift(DOWN * 0.5)
        axes_group.scale(0.9)
        
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
        definition = MathTex(r"f_{\theta} \in \mathcal{F}_1", font_size=36)
        definition.to_edge(LEFT).shift(UP * 1.5)
        self.play(Write(definition))
        equation = MathTex(r"y = w_1 x + w_0", font_size=36)
        equation.next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(equation))
        self.wait(1)
        
        # Try a few "bad" lines first
        candidate_lines = [
            (0.5, 5, RED),
            (3, 1, ORANGE),
            (1.5, 8, PURPLE),
        ]
        
        for w_1, w_0, color in candidate_lines:
            line = axes.plot(lambda x: w_1 * x + w_0, color=color, x_range=[0, 10])
            params = MathTex(f"w_1={w_1}, w_0={w_0}", font_size=36, color=color)
            params.next_to(equation, DOWN, aligned_edge=LEFT)
            
            self.play(Create(line), Write(params))
            self.wait(0.5)
            self.play(FadeOut(line), FadeOut(params))
        
        # Calculate actual best fit using least squares
        A = np.vstack([x_data, np.ones(len(x_data))]).T
        w_0_best, w_1_best = np.linalg.lstsq(A, y_data, rcond=None)[0]
        
        # Show the best fit line
        best_fit_line = axes.plot(
            lambda x: w_0_best * x + w_1_best,
            color=GREEN,
            x_range=[0, 10]
        )
        
        best_params = MathTex(
            f"m={w_0_best:.2f}, b={w_1_best:.2f}",
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
        # Title
        title = Text("Binary Classification Example", font_size=48)
        title.to_edge(UP * 2)
        self.play(Write(title))
        self.wait(1)

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
        axes_group.shift(DOWN * 0.5)
        axes_group.scale(0.9)
        
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
        definition = MathTex(r"f_{\theta} \in \mathcal{F}_{\text{logistic}}", font_size=36)
        definition.to_edge(LEFT).shift(UP * 1.5)
        self.play(Write(definition))
        equation = MathTex(r"y = \sigma(w_2 x_2 + w_1 x_1 + w_0)", font_size=36)
        equation.next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(equation))
        sigmoid = MathTex(r"\sigma(z) = \frac{1}{1 + e^{-z}}", font_size=36)
        sigmoid.next_to(equation, DOWN, aligned_edge=LEFT)
        self.play(Write(sigmoid))
        self.wait(1)
        
        # Try a few "bad" decision boundaries first
        candidate_boundaries = [
            (0.3, -0.5, -2, ORANGE),   # (w2, w1, w0, color)
            (-0.2, 0.8, 1, PURPLE),
            (1, -1, 0, YELLOW),
        ]
        
        for w_2, w_1, w_0, color in candidate_boundaries:
            # Decision boundary is where w2*x2 + w1*x1 + w0 = 0
            # Solving for x2: x2 = -(w1*x1 + w0) / w2
            if abs(w_2) > 0.01:  # Avoid division by zero
                boundary = axes.plot(
                    lambda x: -(w_1 * x + w_0) / w_2,
                    color=color,
                    x_range=[-5, 5]
                )
            else:  # Vertical line
                boundary = axes.get_vertical_line(axes.c2p(-w_0/w_1, 0), color=color)
            
            params = MathTex(
                f"w_2={w_2:.1f}, w_1={w_1:.1f}, w_0={w_0:.1f}",
                font_size=36,
                color=color
            )

            params.to_edge(RIGHT+DOWN*3.5)
            
            self.play(Create(boundary), Write(params))
            self.wait(0.5)
            self.play(FadeOut(boundary), FadeOut(params))
        

        # Calculate actual decision boundary using simple approach
        # Simple approach: find the line that separates the class centers
        w_2_best = 1.0
        w_1_best = 1.0
        w_0_best = 0.0
        
        # Show the best decision boundary
        best_boundary = axes.plot(
            lambda x: -(w_1_best * x + w_0_best) / w_2_best,
            color=GREEN,
            x_range=[-5, 5],
            stroke_width=4
        )
        
        best_params = MathTex(
            f"w_2={w_2_best:.1f}, w_1={w_1_best:.1f}, w_0={w_0_best:.1f}",
            font_size=32,
            color=GREEN
        )
        best_params.to_edge(RIGHT+DOWN*3.5)
        
        best_label = Text("Best Boundary!", font_size=36, color=GREEN)
        best_label.next_to(best_params, DOWN, aligned_edge=LEFT)
        
        self.play(Create(best_boundary), Write(best_params))
        self.play(Write(best_label))
        self.wait(1)

        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        self.interactive_embed()  # For debugging and interaction


