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
        equation = MathTex(r"\hat{y} = \theta_0 + \theta_1 x", font_size=36)
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
            f"\\theta_0^*={theta_0_best:.2f}, \\theta_1^*={theta_1_best:.2f}",
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


class QuadraticRegressionOverUnderfit(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 8, 2],
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
        axes_group.shift(DOWN * 1)
        
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # Generate sample data points (quadratic with noise)
        np.random.seed(42)
        x_data = np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
        theta_true = np.array([1, 0, 0.5])
        y_true = theta_true[0] + theta_true[1] * x_data + theta_true[2] * x_data**2  # True quadratic function
        y_data = y_true + np.random.normal(0, 0.6, len(x_data))

        # Draw the real distribution
        real_distribution = axes.plot(
            lambda x: theta_true[0] + theta_true[1] * x + theta_true[2] * x**2,
            color=WHITE,
            x_range=(-3, 3)
        )

        # 0. REAL DISTRIBUTION
        real_label = Text("Real Distribution", font_size=28, color=WHITE)
        real_label.to_edge(LEFT).shift(UP * 1.5)

        definition_real = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_2", font_size=28)
        definition_real.next_to(real_label, DOWN, aligned_edge=LEFT)

        equation_real = MathTex(r"y = \theta_0 + \theta_1 x + \theta_2 x^2 + \epsilon", font_size=28)
        equation_real.next_to(definition_real, DOWN, aligned_edge=LEFT)

        self.play(Create(real_distribution), Write(real_label))
        self.play(Write(definition_real), Write(equation_real))

        # Create dots for data points
        dots = VGroup()
        for x, y in zip(x_data, y_data):
            dot = Dot(axes.c2p(x, y), color=WHITE, radius=0.08)
            dots.add(dot)
        
        # Animate data points appearing
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.2))
        self.wait(1)

        # Draw Errors from Noise
        error_lines = VGroup()
        for x, y, y_t in zip(x_data, y_data, y_true):
            # Draw vertical line from true value to observed value
            start_point = axes.c2p(x, y_t)
            end_point = axes.c2p(x, y)
            error_line = Line(start_point, end_point, color=YELLOW, stroke_width=2)
            error_lines.add(error_line)
        
        # Add label for errors
        error_label = MathTex(r"\epsilon \sim \mathcal{N}(0, \sigma^2)", font_size=28, color=YELLOW)
        error_label.next_to(equation_real, DOWN, aligned_edge=LEFT)
        
        self.play(LaggedStart(*[Create(line) for line in error_lines], lag_ratio=0.1))
        self.play(Write(error_label))
        self.wait(2)
        
        # Fade out error visualization
        self.play(FadeOut(error_lines), FadeOut(error_label))
        self.wait(0.5)
        
        # Title
        title = Text("Underfitting vs Good Fit vs Overfitting", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        self.play(FadeOut(real_distribution), FadeOut(real_label), FadeOut(definition_real), FadeOut(equation_real))

        # 1. UNDERFITTING - Linear model (too simple)
        underfit_label = Text("Underfitting (Too Simple)", font_size=28, color=RED)
        underfit_label.to_edge(LEFT).shift(UP * 1.5)
        
        definition_underfit = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_1", font_size=32)
        definition_underfit.next_to(underfit_label, DOWN, aligned_edge=LEFT)
        
        equation_underfit = MathTex(r"\hat{y} = \theta_0 + \theta_1 x", font_size=32)
        equation_underfit.next_to(definition_underfit, DOWN, aligned_edge=LEFT)
        
        self.play(Write(underfit_label), Write(definition_underfit), Write(equation_underfit))
        
        # Fit a linear model (underfitting)
        A_linear = np.vstack([np.ones(len(x_data)), x_data]).T
        theta_linear = np.linalg.lstsq(A_linear, y_data, rcond=None)[0]
        
        underfit_line = axes.plot(
            lambda x: theta_linear[0] + theta_linear[1] * x,
            color=RED,
            x_range=[-3, 3]
        )
        
        params_underfit = MathTex(
            f"\\theta_0^*={theta_linear[0]:.2f}, \\theta_1^*={theta_linear[1]:.2f}",
            font_size=28,
            color=RED
        )
        params_underfit.next_to(equation_underfit, DOWN, aligned_edge=LEFT)
        
        self.play(Create(underfit_line), Write(params_underfit))
        self.wait(2)
        
        # Fade out underfitting
        self.play(
            FadeOut(underfit_line),
            FadeOut(underfit_label),
            FadeOut(definition_underfit),
            FadeOut(equation_underfit),
            FadeOut(params_underfit)
        )
        self.wait(0.5)
        
        # 2. GOOD FIT - Quadratic model (just right)
        goodfit_label = Text("Good Fit (Just Right)", font_size=28, color=GREEN)
        goodfit_label.to_edge(LEFT).shift(UP * 1.5)
        
        definition_goodfit = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_2", font_size=32)
        definition_goodfit.next_to(goodfit_label, DOWN, aligned_edge=LEFT)
        
        equation_goodfit = MathTex(r"\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2", font_size=32)
        equation_goodfit.next_to(definition_goodfit, DOWN, aligned_edge=LEFT)
        
        self.play(Write(goodfit_label), Write(definition_goodfit), Write(equation_goodfit))
        
        # Fit a quadratic model (good fit)
        A_quad = np.vstack([np.ones(len(x_data)), x_data, x_data**2]).T
        theta_quad = np.linalg.lstsq(A_quad, y_data, rcond=None)[0]
        
        goodfit_curve = axes.plot(
            lambda x: theta_quad[0] + theta_quad[1] * x + theta_quad[2] * x**2,
            color=GREEN,
            x_range=[-3, 3]
        )
        
        params_goodfit = MathTex(
            f"\\theta_0^*={theta_quad[0]:.2f}, \\theta_1^*={theta_quad[1]:.2f}, \\theta_2^*={theta_quad[2]:.2f}",
            font_size=28,
            color=GREEN
        )
        params_goodfit.next_to(equation_goodfit, DOWN, aligned_edge=LEFT)
        
        self.play(Create(goodfit_curve), Write(params_goodfit))
        self.wait(2)
        
        # Fade out good fit
        self.play(
            FadeOut(goodfit_curve),
            FadeOut(goodfit_label),
            FadeOut(definition_goodfit),
            FadeOut(equation_goodfit),
            FadeOut(params_goodfit)
        )
        self.wait(0.5)
        
        # 3. OVERFITTING - High degree polynomial (too complex)
        overfit_label = Text("Overfitting (Too Complex)", font_size=28, color=ORANGE)
        overfit_label.to_edge(LEFT).shift(UP * 2)
        
        definition_overfit = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_{10}", font_size=32)
        definition_overfit.next_to(overfit_label, DOWN, aligned_edge=LEFT)
        
        equation_overfit = MathTex(
            r"\hat{y} = \theta_0 + \theta_1 x + \cdots + \theta_{10} x^{10}",
            font_size=32
        )
        equation_overfit.next_to(definition_overfit, DOWN, aligned_edge=LEFT)
        
        self.play(Write(overfit_label), Write(definition_overfit), Write(equation_overfit))
        
        # Fit a 10th degree polynomial (overfitting)
        degree = 10
        A_poly = np.vstack([x_data**i for i in range(degree + 1)]).T
        theta_poly = np.linalg.lstsq(A_poly, y_data, rcond=None)[0]
        
        def poly_func(x):
            return sum(theta_poly[i] * x**i for i in range(degree + 1))
        
        overfit_curve = axes.plot(
            poly_func,
            color=ORANGE,
            x_range=[-2.5, 2.5],
            use_smoothing=True
        )
        
        self.play(Create(overfit_curve))
        self.wait(2)
        
        # Show all three together for comparison
        comparison_label = Text("Comparison", font_size=32, color=YELLOW)
        comparison_label.next_to(title, DOWN)
        
        self.play(
            FadeOut(overfit_label),
            FadeOut(definition_overfit),
            FadeOut(equation_overfit),
            Write(comparison_label)
        )
        
        # Recreate all three curves
        underfit_line_final = axes.plot(
            lambda x: theta_linear[0] + theta_linear[1] * x,
            color=RED,
            x_range=[-3, 3],
            stroke_width=3
        )
        
        goodfit_curve_final = axes.plot(
            lambda x: theta_quad[0] + theta_quad[1] * x + theta_quad[2] * x**2,
            color=GREEN,
            x_range=[-3, 3],
            stroke_width=3
        )
        
        # Add the real distribution back
        real_distribution_final = axes.plot(
            lambda x: theta_true[0] + theta_true[1] * x + theta_true[2] * x**2,
            color=WHITE,
            x_range=[-3, 3],
            stroke_width=4,
            stroke_opacity=0.7
        )
        
        self.play(
            Create(underfit_line_final),
            Create(goodfit_curve_final),
            Create(real_distribution_final)
        )
        
        # Add legend
        legend_under = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=3),
            Text("Underfit", font_size=20, color=RED)
        ).arrange(RIGHT, buff=0.2)
        
        legend_good = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=GREEN, stroke_width=3),
            Text("Good Fit", font_size=20, color=GREEN)
        ).arrange(RIGHT, buff=0.2)
        
        legend_over = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=ORANGE, stroke_width=3),
            Text("Overfit", font_size=20, color=ORANGE)
        ).arrange(RIGHT, buff=0.2)
        
        legend_real = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=WHITE, stroke_width=4, stroke_opacity=0.7),
            Text("Real Distribution", font_size=20, color=WHITE)
        ).arrange(RIGHT, buff=0.2)
        
        legend = VGroup(legend_real, legend_under, legend_good, legend_over).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2
        )
        legend.to_edge(LEFT).shift(DOWN * 1.5)
        
        self.play(FadeIn(legend))
        self.wait(3)
        
        # Error Analysis Section
        self.play(
            FadeOut(comparison_label),
            FadeOut(legend)
        )
        
        error_title = Text("Error Analysis: Residuals vs Noise", font_size=32, color=YELLOW)
        error_title.next_to(title, DOWN)
        self.play(Write(error_title))
        self.wait(1)
        
        # Calculate errors for each model
        y_pred_linear = theta_linear[0] + theta_linear[1] * x_data
        y_pred_quad = theta_quad[0] + theta_quad[1] * x_data + theta_quad[2] * x_data**2
        y_pred_poly = np.array([poly_func(x) for x in x_data])
        
        # Calculate MSE for each model and noise
        mse_noise = np.mean((y_data - y_true)**2)
        mse_underfit = np.mean((y_data - y_pred_linear)**2)
        mse_goodfit = np.mean((y_data - y_pred_quad)**2)
        mse_overfit = np.mean((y_data - y_pred_poly)**2)
        
        # Create error comparison table
        error_labels = VGroup(
            MathTex(r"\frac{1}{N}\|\boldsymbol{\epsilon}\|_2^2", font_size=28, color=YELLOW),
            MathTex(r"\frac{1}{N}\|\mathbf{r}_{\text{underfit}}\|_2^2", font_size=28, color=RED),
            MathTex(r"\frac{1}{N}\|\mathbf{r}_{\text{good fit}}\|_2^2", font_size=28, color=GREEN),
            MathTex(r"\frac{1}{N}\|\mathbf{r}_{\text{overfit}}\|_2^2", font_size=28, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        error_values = VGroup(
            MathTex(f" = {mse_noise:.3f}", font_size=28, color=YELLOW),
            MathTex(f" = {mse_underfit:.3f}", font_size=28, color=RED),
            MathTex(f" = {mse_goodfit:.3f}", font_size=28, color=GREEN),
            MathTex(f" = {mse_overfit:.3f}", font_size=28, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Align the values with their corresponding labels
        for i, (label, value) in enumerate(zip(error_labels, error_values)):
            value.align_to(label, UP)
        
        # Position labels and values
        error_labels.to_edge(LEFT).shift(UP * 0.5)
        error_values.next_to(error_labels, RIGHT, buff=0.5)
        
        self.play(
            Write(error_labels),
            Write(error_values)
        )
        self.wait(4)
        
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
        equation = MathTex(r"\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2", font_size=36)
        equation.next_to(definition, DOWN, aligned_edge=LEFT)
        self.play(Write(equation))
        # Show classification rule
        classification = MathTex(
            r"\text{Class } = \begin{cases} 0 & \text{if } \hat{y} \leq 0 \\ 1 & \text{if } \hat{y} > 0 \end{cases}",
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
            f"\\theta_0^*={theta_0_best:.1f}, \\theta_1^*={theta_1_best:.1f}, \\theta_2^*={theta_2_best:.1f}",
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



