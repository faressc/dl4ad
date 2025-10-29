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

class QuantileRegressionOverUnderfitWithValidation(Scene):
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
        x_data_all = np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
        theta_true = np.array([1, 0, 0.5])
        y_true_all = theta_true[0] + theta_true[1] * x_data_all + theta_true[2] * x_data_all**2
        y_data_all = y_true_all + np.random.normal(0, 0.6, len(x_data_all))

        # Draw the real distribution
        real_distribution = axes.plot(
            lambda x: theta_true[0] + theta_true[1] * x + theta_true[2] * x**2,
            color=WHITE,
            x_range=(-3, 3),
            stroke_width=2,
            stroke_opacity=0.5
        )

        # Show real distribution
        real_label = Text("Real Distribution", font_size=24, color=WHITE)
        real_label.to_edge(LEFT).shift(UP * 2.5)

        definition_real = MathTex(r"f_{\boldsymbol{\theta}} \in \mathcal{F}_2", font_size=24)
        definition_real.next_to(real_label, DOWN, aligned_edge=LEFT)

        equation_real = MathTex(r"y = \theta_0 + \theta_1 x + \theta_2 x^2 + \epsilon", font_size=24)
        equation_real.next_to(definition_real, DOWN, aligned_edge=LEFT)

        self.play(Create(real_distribution), Write(real_label))
        self.play(Write(definition_real), Write(equation_real))
        self.wait(1)

        # Split into training and validation sets (60-40 split)
        train_indices = [0, 1, 3, 5, 7, 9, 10]  # 7 points
        val_indices = [2, 4, 6, 8]  # 4 points
        
        x_train = x_data_all[train_indices]
        y_train = y_data_all[train_indices]
        x_val = x_data_all[val_indices]
        y_val = y_data_all[val_indices]

        # Title
        title = Text("Train vs Validation Error Analysis", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Create dots for training data (BLUE)
        dots_train = VGroup()
        for x, y in zip(x_train, y_train):
            dot = Dot(axes.c2p(x, y), color=BLUE, radius=0.10)
            dots_train.add(dot)
        
        # Create dots for validation data (YELLOW)
        dots_val = VGroup()
        for x, y in zip(x_val, y_val):
            dot = Dot(axes.c2p(x, y), color=YELLOW, radius=0.10)
            dots_val.add(dot)
        
        # Fade out real distribution info
        self.play(FadeOut(real_label), FadeOut(definition_real), FadeOut(equation_real))
        
        # Show data split
        split_label = Text("Data Split", font_size=28, color=WHITE)
        split_label.to_edge(LEFT).shift(UP * 2)
        
        # Animate training points first
        train_label = VGroup(
            Dot(color=BLUE, radius=0.08),
            Text("Training Data", font_size=20, color=BLUE)
        ).arrange(RIGHT, buff=0.2)
        train_label.next_to(split_label, DOWN, aligned_edge=LEFT)
        
        self.play(Write(split_label), FadeIn(train_label))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots_train], lag_ratio=0.2))
        self.wait(1)
        
        # Animate validation points
        val_label = VGroup(
            Dot(color=YELLOW, radius=0.08),
            Text("Validation Data", font_size=20, color=YELLOW)
        ).arrange(RIGHT, buff=0.2)
        val_label.next_to(train_label, DOWN, aligned_edge=LEFT)
        
        self.play(FadeIn(val_label))
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots_val], lag_ratio=0.2))
        self.wait(2)
        
        # Fade out split labels
        self.play(FadeOut(split_label), FadeOut(train_label), FadeOut(val_label))
        self.wait(0.5)

        # Test different model complexities
        models = [
            {
                'name': 'Underfitting',
                'degree': 1,
                'color': RED,
                'label': r'\mathcal{F}_1: \hat{y} = \theta_0 + \theta_1 x'
            },
            {
                'name': 'Good Fit',
                'degree': 2,
                'color': GREEN,
                'label': r'\mathcal{F}_2: \hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2'
            },
            {
                'name': 'Overfitting',
                'degree': 6,
                'color': ORANGE,
                'label': r'\mathcal{F}_6: \hat{y} = \theta_0 + \cdots + \theta_6 x^6'
            }
        ]

        all_train_errors = []
        all_val_errors = []
        all_curves = []

        for model in models:
            degree = model['degree']
            color = model['color']
            name = model['name']
            
            # Model label
            model_label = Text(f"{name}", font_size=28, color=color)
            model_label.to_edge(LEFT).shift(UP * 2)
            
            equation = MathTex(model['label'], font_size=28, color=color)
            equation.next_to(model_label, DOWN, aligned_edge=LEFT)
            
            self.play(Write(model_label), Write(equation))
            
            # Fit model on training data
            A_train = np.vstack([x_train**i for i in range(degree + 1)]).T
            theta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
            
            def poly_func(x):
                return sum(theta[i] * x**i for i in range(degree + 1))
            
            # Create the fitted curve
            curve = axes.plot(
                poly_func,
                color=color,
                x_range=[-2.5, 2.5],
                use_smoothing=True,
                stroke_width=3
            )
            
            self.play(Create(curve))
            self.wait(1)
            
            # Calculate training error
            y_pred_train = np.array([poly_func(x) for x in x_train])
            train_error = np.mean((y_train - y_pred_train)**2)
            
            # Calculate validation error
            y_pred_val = np.array([poly_func(x) for x in x_val])
            val_error = np.mean((y_val - y_pred_val)**2)
            
            # Store errors
            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            all_curves.append(curve)
            
            self.wait(1)
            
            # Fade out for next model
            self.play(
                FadeOut(curve),
                FadeOut(model_label),
                FadeOut(equation)
            )
            self.wait(0.5)
        
        # Comparison: Show all models together
        comparison_title = Text("Model Comparison", font_size=32, color=WHITE)
        comparison_title.next_to(title, DOWN)
        self.play(Write(comparison_title))
        
        # Show real distribution again
        self.play(real_distribution.animate.set_stroke(opacity=0.7, width=4))
        
        # Show all curves
        for i, model in enumerate(models):
            degree = model['degree']
            color = model['color']
            
            A_train = np.vstack([x_train**i for i in range(degree + 1)]).T
            theta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
            
            def poly_func(x, t=theta, d=degree):
                return sum(t[j] * x**j for j in range(d + 1))
            
            curve = axes.plot(
                poly_func,
                color=color,
                x_range=[-2.5, 2.5],
                use_smoothing=True,
                stroke_width=3
            )
            
            self.play(Create(curve))
        
        self.wait(2)
        
        # Create error comparison chart
        chart_title = Text("Error Comparison: Key Insight", font_size=28, color=WHITE)
        chart_title.to_edge(LEFT).shift(UP * 2)
        
        error_table = VGroup()
        headers = VGroup(
            Text("Model", font_size=20),
            MathTex(r"\frac{1}{N}\|\mathbf{r}_{\text{train}}\|_2^2", font_size=18, color=BLUE),
            MathTex(r"\frac{1}{N}\|\mathbf{r}_{\text{val}}\|_2^2", font_size=18, color=YELLOW)
        ).arrange(RIGHT, buff=0.6)
        error_table.add(headers)
        
        for i, model in enumerate(models):
            row = VGroup(
                Text(model['name'][:8], font_size=18, color=model['color']),
                Text(f"{all_train_errors[i]:.3f}", font_size=18, color=BLUE),
                Text(f"{all_val_errors[i]:.3f}", font_size=18, color=YELLOW)
            ).arrange(RIGHT, buff=0.6)
            # Align with headers
            for j, cell in enumerate(row):
                cell.align_to(headers[j], LEFT)
            error_table.add(row)
        
        error_table.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        error_table.next_to(chart_title, DOWN, aligned_edge=LEFT)
        
        self.play(FadeOut(comparison_title))
        self.play(Write(chart_title), Write(error_table))
        self.wait(4)
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

class BiasVarianceTradeoff(Scene):
    def construct(self):
        # Title
        title = Text("Bias-Variance Tradeoff", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
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
        axes_group.shift(RIGHT * 1.5 + DOWN * 0.5)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # True function (quadratic)
        theta_true = np.array([1, 0, 0.5])
        
        # Draw the true distribution
        true_func = axes.plot(
            lambda x: theta_true[0] + theta_true[1] * x + theta_true[2] * x**2,
            color=WHITE,
            x_range=(-3, 3),
            stroke_width=4,
            stroke_opacity=0.8
        )
        
        true_label = Text("True Function f(x)", font_size=24, color=WHITE)
        true_label.to_edge(LEFT).shift(UP * 1.5)
        
        self.play(Create(true_func), Write(true_label))
        self.wait(1)
        
        # Generate multiple datasets with different x values
        np.random.seed(43)
        n_datasets = 5
        n_points = 24
        
        datasets = []
        for i in range(n_datasets):
            # Each dataset has slightly different x sampling points
            x_sample = np.sort(np.random.uniform(-2.5, 2.5, n_points))
            y_true = theta_true[0] + theta_true[1] * x_sample + theta_true[2] * x_sample**2
            y_noisy = y_true + np.random.normal(0, 0.8, len(x_sample))
            datasets.append((x_sample, y_noisy))
        
        # Show concept: multiple datasets
        concept_text = Text("Multiple Training Sets from Same Distribution", font_size=18)
        concept_text.next_to(true_label, DOWN, aligned_edge=LEFT)
        self.play(Write(concept_text))
        self.wait(1)
        
        # Show all datasets with dots
        all_dataset_dots = VGroup()
        colors = [BLUE, GREEN, RED, PURPLE, YELLOW]
        
        for dataset_idx, (x_data, y_data) in enumerate(datasets):
            dataset_dots = VGroup()
            for x, y in zip(x_data, y_data):
                dot = Dot(axes.c2p(x, y), color=colors[dataset_idx], radius=0.06)
                dataset_dots.add(dot)
            all_dataset_dots.add(dataset_dots)
        
        # Animate all datasets appearing
        for dataset_dots in all_dataset_dots:
            self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dataset_dots], lag_ratio=0.05))
        
        self.wait(2)
        self.play(FadeOut(all_dataset_dots), FadeOut(concept_text))
        self.play(FadeOut(true_label))
        self.wait(0.5)
        
        # Test three model complexities
        models = [
            {
                'name': 'High Bias (Underfit)',
                'degree': 1,
                'color': RED,
                'short_name': 'underfit'
            },
            {
                'name': 'Balanced',
                'degree': 2,
                'color': GREEN,
                'short_name': 'balanced'
            },
            {
                'name': 'High Variance (Overfit)',
                'degree': 6,
                'color': ORANGE,
                'short_name': 'overfit'
            }
        ]
        
        # For each model type, fit on all datasets and visualize
        for model_idx, model in enumerate(models):
            degree = model['degree']
            color = model['color']
            name = model['name']
            
            # Clear previous
            if model_idx > 0:
                self.wait(0.5)
            
            # Model info
            model_label = Text(name, font_size=28, color=color)
            model_label.to_edge(LEFT).shift(UP * 1.5)
            
            if degree == 1:
                eq_text = r"\mathcal{F}_1: \hat{y} = \theta_0 + \theta_1 x"
            elif degree == 2:
                eq_text = r"\mathcal{F}_2: \hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2"
            else:
                eq_text = r"\mathcal{F}_6: \hat{y} = \theta_0 + \cdots + \theta_6 x^6"
            
            equation = MathTex(eq_text, font_size=24, color=color)
            equation.next_to(model_label, DOWN, aligned_edge=LEFT)
            
            self.play(Write(model_label), Write(equation))
            self.wait(0.5)
            
            # Fit model on each dataset and plot
            fitted_curves = VGroup()
            # Use a common grid for evaluation
            x_eval = np.linspace(-2.5, 2.5, 50)
            all_predictions = []
            
            for dataset_idx, (x_data, y_data) in enumerate(datasets):
                # Fit model
                A = np.vstack([x_data**i for i in range(degree + 1)]).T
                theta = np.linalg.lstsq(A, y_data, rcond=None)[0]
                
                def poly_func(x, t=theta.copy(), d=degree):
                    return sum(t[j] * x**j for j in range(d + 1))
                
                # Create curve
                curve = axes.plot(
                    poly_func,
                    color=color,
                    x_range=[-2.5, 2.5],
                    use_smoothing=True,
                    stroke_width=2,
                    stroke_opacity=0.4
                )
                fitted_curves.add(curve)
                
                # Store predictions on common grid
                predictions = np.array([poly_func(x) for x in x_eval])
                all_predictions.append(predictions)
            
            # Animate all fitted curves appearing
            self.play(LaggedStart(*[Create(curve) for curve in fitted_curves], lag_ratio=0.2))
            self.wait(1)
            
            # Calculate average prediction across all models
            all_predictions = np.array(all_predictions)
            avg_predictions = np.mean(all_predictions, axis=0)
            
            # Interpolate average prediction curve
            from scipy.interpolate import interp1d
            avg_func_interp = interp1d(x_eval, avg_predictions, kind='cubic', fill_value='extrapolate')
            
            avg_curve = axes.plot(
                avg_func_interp,
                color=color,
                x_range=[-2.5, 2.5],
                stroke_width=4
            )
            
            avg_label = Text("Average Model", font_size=20, color=color)
            avg_label.next_to(equation, DOWN, aligned_edge=LEFT)
            
            self.play(Create(avg_curve), Write(avg_label))
            self.wait(1)
            
            # Calculate bias and variance on evaluation grid
            # Bias: difference between average prediction and true function
            true_vals = theta_true[0] + theta_true[1] * x_eval + theta_true[2] * x_eval**2
            bias_squared = np.mean((avg_predictions - true_vals)**2)
            
            # Variance: average squared deviation of predictions from their mean
            variance = np.mean(np.var(all_predictions, axis=0))
            
            # Show metrics
            bias_text = MathTex(
                f"\\text{{Bias}}^2 = {bias_squared:.3f}",
                font_size=24,
                color=color
            )
            bias_text.next_to(avg_label, DOWN, aligned_edge=LEFT)
            
            var_text = MathTex(
                f"\\text{{Variance}} = {variance:.3f}",
                font_size=24,
                color=color
            )
            var_text.next_to(bias_text, DOWN, aligned_edge=LEFT)
            
            total_error = bias_squared + variance
            total_text = MathTex(
                f"\\text{{Total}} = {total_error:.3f}",
                font_size=24,
                color=YELLOW
            )
            total_text.next_to(var_text, DOWN, aligned_edge=LEFT)
            
            self.play(Write(bias_text), Write(var_text))
            self.wait(0.5)
            self.play(Write(total_text))
            self.wait(2)
            
            # Fade out everything except axes and true function
            self.play(
                *[FadeOut(curve) for curve in fitted_curves],
                FadeOut(avg_curve),
                FadeOut(model_label),
                FadeOut(equation),
                FadeOut(avg_label),
                FadeOut(bias_text),
                FadeOut(var_text),
                FadeOut(total_text)
            )
        
        # Summary comparison
        summary_title = Text("Summary: The Tradeoff", font_size=28, color=YELLOW)
        summary_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(summary_title))
        
        # Create summary table
        summary_table = VGroup()
        
        headers = VGroup(
            Text("Model", font_size=20),
            MathTex(r"\text{Bias}^2", font_size=20),
            MathTex(r"\text{Variance}", font_size=20),
            MathTex(r"\text{Total}", font_size=20, color=YELLOW)
        ).arrange(RIGHT, buff=0.5)
        summary_table.add(headers)
        
        # Recalculate for summary
        summary_data = []
        x_eval = np.linspace(-2.5, 2.5, 50)
        
        for model in models:
            degree = model['degree']
            color = model['color']
            
            all_predictions = []
            
            for x_data, y_data in datasets:
                A = np.vstack([x_data**i for i in range(degree + 1)]).T
                theta = np.linalg.lstsq(A, y_data, rcond=None)[0]
                
                predictions = np.array([sum(theta[j] * x**j for j in range(degree + 1)) for x in x_eval])
                all_predictions.append(predictions)
            
            all_predictions = np.array(all_predictions)
            avg_predictions = np.mean(all_predictions, axis=0)
            true_vals = theta_true[0] + theta_true[1] * x_eval + theta_true[2] * x_eval**2
            
            bias_sq = np.mean((avg_predictions - true_vals)**2)
            variance = np.mean(np.var(all_predictions, axis=0))
            total = bias_sq + variance
            
            summary_data.append((model['short_name'].capitalize(), bias_sq, variance, total, color))
        
        for name, bias_sq, var, total, color in summary_data:
            row = VGroup(
                Text(name[:8], font_size=18, color=color),
                Text(f"{bias_sq:.3f}", font_size=18),
                Text(f"{var:.3f}", font_size=18),
                Text(f"{total:.3f}", font_size=18, color=YELLOW)
            ).arrange(RIGHT, buff=0.5)
            
            for j, cell in enumerate(row):
                cell.align_to(headers[j], LEFT)
            summary_table.add(row)
        
        summary_table.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        summary_table.next_to(summary_title, DOWN, aligned_edge=LEFT)
        
        self.play(Write(summary_table))
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

class EmpiricalRiskMinimization(Scene):
    def construct(self):
        # Create axes for data visualization
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 25, 5],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Add labels
        x_label = MathTex(r"\mathbf{x} \in \mathcal{X}", font_size=24)
        y_label = MathTex(r"\mathbf{y} \in \mathcal{Y}", font_size=24)
        x_label.next_to(axes.x_axis, DOWN)
        y_label.next_to(axes.y_axis, LEFT)
        
        axes_group = VGroup(axes, x_label, y_label)
        axes_group.shift(RIGHT * 2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Step 1: Show the dataset
        dataset_title = Text("I: Dataset", font_size=28, color=BLUE)
        dataset_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(dataset_title))
        
        # Dataset notation
        dataset_notation = MathTex(
            r"D = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N",
            font_size=28
        )
        dataset_notation.next_to(dataset_title, DOWN, aligned_edge=LEFT)
        self.play(Write(dataset_notation))
        
        # Generate sample data points
        np.random.seed(42)
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_true = 2 * x_data + 3
        y_data = y_true + np.random.normal(0, 1.5, len(x_data))
        N = len(x_data)
        
        # Create dots for data points
        dots = VGroup()
        for x, y in zip(x_data, y_data):
            dot = Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            dots.add(dot)
        
        # Animate data points appearing
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.2))
        
        # Show N (number of samples)
        n_label = MathTex(f"N = {N}", font_size=24, color=BLUE)
        n_label.next_to(dataset_notation, DOWN, aligned_edge=LEFT)
        self.play(Write(n_label))
        self.wait(2)
        
        # Step 2: Function family
        self.play(
            FadeOut(dataset_title),
            FadeOut(dataset_notation),
            FadeOut(n_label)
        )

        function_title = Text("II: Function Family", font_size=28, color=GREEN)
        function_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(function_title))
        
        # Function notation
        function_notation = MathTex(
            r"f_{\boldsymbol{\theta}}: \mathcal{X} \to \mathcal{Y}",
            font_size=28
        )
        function_notation.next_to(function_title, DOWN, aligned_edge=LEFT)
        self.play(Write(function_notation))
        
        # Example: linear function
        example_func = MathTex(
            r"\hat{y} = \theta_0 + \theta_1 x",
            font_size=28,
            color=GREEN
        )
        example_func.next_to(function_notation, DOWN, aligned_edge=LEFT)
        self.play(Write(example_func))
        
        # Show a candidate function
        theta_0_candidate = 5
        theta_1_candidate = 1.5
        candidate_line = axes.plot(
            lambda x: theta_0_candidate + theta_1_candidate * x,
            color=GREEN,
            x_range=[0, 10]
        )
        self.play(Create(candidate_line))
        self.wait(2)
        
        # Step 3: Loss function
        self.play(
            FadeOut(function_title),
            FadeOut(function_notation),
            FadeOut(example_func)
        )

        loss_title = Text("III: Loss Function", font_size=28, color=YELLOW)
        loss_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(loss_title))
        
        # Loss notation
        loss_notation = MathTex(
            r"\ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)",
            font_size=28
        )
        loss_notation.next_to(loss_title, DOWN, aligned_edge=LEFT)
        self.play(Write(loss_notation))
        
        # Show errors as vertical lines
        error_lines = VGroup()
        for x, y in zip(x_data, y_data):
            y_pred = theta_0_candidate + theta_1_candidate * x
            start_point = axes.c2p(x, y)
            end_point = axes.c2p(x, y_pred)
            error_line = Line(start_point, end_point, color=YELLOW, stroke_width=3)
            error_lines.add(error_line)
        
        self.play(LaggedStart(*[Create(line) for line in error_lines], lag_ratio=0.1))
        self.wait(2)
        
        # Step 4: Empirical Risk
        self.play(
            FadeOut(loss_title),
            FadeOut(loss_notation),
            FadeOut(error_lines)
        )

        risk_title = Text("IV: Empirical Risk", font_size=28, color=ORANGE)
        risk_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(risk_title))
        
        # Empirical risk formula
        risk_formula = MathTex(
            r"\hat{R}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)",
            font_size=26
        )
        risk_formula.next_to(risk_title, DOWN, aligned_edge=LEFT)
        self.play(Write(risk_formula))
        
        # Calculate current risk
        y_pred_candidate = theta_0_candidate + theta_1_candidate * x_data
        current_risk = np.mean((y_data - y_pred_candidate)**2)
        
        risk_value = MathTex(
            f"\\hat{{R}}(\\boldsymbol{{\\theta}}) = {current_risk:.3f}",
            font_size=24,
            color=ORANGE
        )
        risk_value.next_to(risk_formula, DOWN, aligned_edge=LEFT)
        self.play(Write(risk_value))
        self.wait(2)
        
        # Step 5: Minimize to find optimal parameters
        self.play(
            FadeOut(risk_title),
            FadeOut(risk_formula),
            FadeOut(risk_value),
            FadeOut(candidate_line)
        )

        minimize_title = Text("V: Minimize Empirical Risk", font_size=28, color=RED)
        minimize_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(minimize_title))
        
        self.interactive_embed()

        # Minimization goal
        minimize_goal = MathTex(
            r"\boldsymbol{\theta}^* = \arg\min\limits_{\boldsymbol{\theta}} \hat{R}(\boldsymbol{\theta})",
            font_size=26
        )
        minimize_goal.next_to(minimize_title, DOWN, aligned_edge=LEFT)
        self.play(Write(minimize_goal))
        
        # Try several candidates
        candidates = [
            (8, 1.0, PURPLE),
            (3, 2.5, ORANGE),
        ]
        
        for theta_0, theta_1, color in candidates:
            line = axes.plot(lambda x: theta_0 + theta_1 * x, color=color, x_range=[0, 10])
            y_pred = theta_0 + theta_1 * x_data
            risk = np.mean((y_data - y_pred)**2)
            
            risk_text = MathTex(
                f"\\hat{{R}} = {risk:.3f}",
                font_size=24,
                color=color
            )
            risk_text.next_to(minimize_goal, DOWN, aligned_edge=LEFT)
            
            self.play(Create(line), Write(risk_text))
            self.wait(0.8)
            self.play(FadeOut(line), FadeOut(risk_text))
        
        # Calculate and show optimal fit
        A = np.vstack([np.ones(len(x_data)), x_data]).T
        theta_0_best, theta_1_best = np.linalg.lstsq(A, y_data, rcond=None)[0]
        
        best_fit_line = axes.plot(
            lambda x: theta_0_best + theta_1_best * x,
            color=GREEN,
            x_range=[0, 10],
            stroke_width=4
        )
        
        y_pred_best = theta_0_best + theta_1_best * x_data
        best_risk = np.mean((y_data - y_pred_best)**2)
        
        best_params = MathTex(
            f"\\boldsymbol{{\\theta}}^* = ({theta_0_best:.2f}, {theta_1_best:.2f})",
            font_size=26,
            color=GREEN
        )
        best_params.next_to(minimize_goal, DOWN, aligned_edge=LEFT)
        
        best_risk_text = MathTex(
            f"\\hat{{R}}(\\boldsymbol{{\\theta}}^*) = {best_risk:.3f}",
            font_size=24,
            color=GREEN
        )
        best_risk_text.next_to(best_params, DOWN, aligned_edge=LEFT)
        
        optimal_label = Text("Optimal Solution!", font_size=28, color=GREEN)
        optimal_label.next_to(best_risk_text, DOWN, aligned_edge=LEFT)
        
        self.play(Create(best_fit_line))
        self.play(Write(best_params), Write(best_risk_text))
        self.play(Write(optimal_label))
        self.wait(2)
        
        # Final summary
        self.play(
            FadeOut(minimize_title),
            FadeOut(minimize_goal),
            FadeOut(best_params),
            FadeOut(best_risk_text),
            FadeOut(optimal_label)
        )
        
        summary_title = Text("Summary: Machine Learning", font_size=28, color=WHITE)
        summary_title.to_edge(LEFT).shift(UP * 2.5)
        self.play(Write(summary_title))
        
        summary_points = VGroup(
            MathTex(r"\text{I. Dataset: } D = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N", font_size=20),
            MathTex(r"\text{II. Function: } f_{\boldsymbol{\theta}}: \mathcal{X} \to \mathcal{Y}", font_size=20),
            MathTex(r"\text{III. Loss: } \ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i), \mathbf{y}_i)", font_size=20),
            MathTex(r"\text{IV. Empirical Risk: } \hat{R}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \ell(\cdot)", font_size=20),
            MathTex(r"\text{V. Goal: } \boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \hat{R}(\boldsymbol{\theta})", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary_points.next_to(summary_title, DOWN, aligned_edge=LEFT)
        
        self.play(Write(summary_points))
        self.wait(4)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class LossLandscapeVisualization(ThreeDScene):
    def construct(self):
        
        # Create 3D axes for loss landscape
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 20, 4],
            x_length=6,
            y_length=6,
            z_length=3,
            axis_config={"color": WHITE},
            tips=False
        )
        
        # Labels for parameters
        theta_0_label = MathTex(r"\theta_0", font_size=28)
        theta_1_label = MathTex(r"\theta_1", font_size=28)
        loss_label = MathTex(r"\hat{R}(\boldsymbol{\theta})", font_size=28)
        
        # Position labels (will be updated with camera)
        theta_0_label.next_to(axes.x_axis, RIGHT)
        theta_1_label.next_to(axes.y_axis, UP)
        loss_label.next_to(axes.z_axis, OUT)

        axes_group = VGroup(axes, theta_0_label, theta_1_label, loss_label)
        
        # Define loss landscape (quadratic bowl)
        def loss_function(theta_0, theta_1):
            # Minimum at (0.5, -1)
            return 1. * (theta_0 - 0.5)**2 + 1.6 * (theta_1 + 1)**2 - 1.5
        
        # Create surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, loss_function(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.4,
            checkerboard_colors=[GRAY_A, GRAY_B]
        )
        
        # Show the 3D setup
        self.play(Create(axes_group))
        self.wait(1)

        
        self.play(
            self.camera.animate.set_euler_angles(theta=-30 * DEGREES, phi=60 * DEGREES, gamma=0).scale(1.5),
            Create(surface),
            run_time=3
        )
        
        self.begin_ambient_camera_rotation(rate=0.05)
        
        # Equation display
        equation = MathTex(
            r"\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \hat{R}(\boldsymbol{\theta}_t)",
            font_size=32
        )
        equation.to_edge(DOWN).shift(UP * 0.5)
        equation.fix_in_frame()
        self.play(Write(equation))
        self.wait(1)
        
        # Test different learning rates
        learning_rates = [
            {"eta": 0.008, "color": RED, "name": "Too Small (η=0.008)"},
            {"eta": 0.06, "color": GREEN, "name": "Good (η=0.06)"},
            {"eta": 0.7, "color": ORANGE, "name": "Too Large (η=0.7)"},
        ]

        self.interactive_embed()  # For debugging and interaction
        
        for lr_config in learning_rates:
            eta = lr_config["eta"]
            color = lr_config["color"]
            name = lr_config["name"]
            
            # Learning rate label
            lr_label = Text(f"Learning Rate: {name}", font_size=28, color=color)
            lr_label.to_edge(LEFT).shift(UP * 2.5)
            lr_label.fix_in_frame()
            self.play(Write(lr_label))
            
            # Starting point
            theta_0_init = 1.5
            theta_1_init = 1.5
            
            # Gradient descent iterations
            theta_0 = theta_0_init
            theta_1 = theta_1_init
            path_points = [(theta_0, theta_1)]
            
            # Compute gradient function
            def gradient(t0, t1):
                # Analytical gradient of our loss function
                grad_t0 = 2 * (t0 - 0.5)
                grad_t1 = 3.2 * (t1 + 1)
                return grad_t0, grad_t1
            
            print(f"Starting gradient descent with eta={eta}")
            
            # Run gradient descent
            max_iterations = 50
            for _ in range(max_iterations):
                grad_t0, grad_t1 = gradient(theta_0, theta_1)
                theta_0 = theta_0 - eta * grad_t0
                theta_1 = theta_1 - eta * grad_t1
                path_points.append((theta_0, theta_1))
                
                # Stop if converged or diverging
                if np.sqrt(grad_t0**2 + grad_t1**2) < 0.01:
                    break
                if abs(theta_0) > 10 or abs(theta_1) > 10:
                    break

                print(f"Iteration {_}: theta_0={theta_0:.4f}, theta_1={theta_1:.4f}")

            print(f"Finished gradient descent with eta={eta}")

            # Create path visualization
            path_3d = VGroup()
            dots_3d = VGroup()
            
            for i in range(len(path_points) - 1):
                t0_start, t1_start = path_points[i]
                t0_end, t1_end = path_points[i + 1]
                
                # Path segment
                start_point = axes.c2p(t0_start, t1_start, loss_function(t0_start, t1_start))
                end_point = axes.c2p(t0_end, t1_end, loss_function(t0_end, t1_end))

                line = Line(start_point, end_point, color=color, stroke_width=10)
                path_3d.add(line)
                
                # Add dot at each point
                if i == 0:
                    dot = Dot3D(radius=0.1, color=WHITE).move_to(start_point)
                    dots_3d.add(dot)
                dot = Dot3D(radius=0.08, color=WHITE).move_to(end_point)
                dots_3d.add(dot)

                print(f"Path point {i}: t0={t0_end:.4f}, t1={t1_end:.4f}")
            
            # Animate the path
            if len(path_points) > 1:
                # Show starting point
                self.play(Create(dots_3d[0]))
                self.wait(0.5)
                
                # Animate gradient descent steps
                for i, (line, dot) in enumerate(zip(path_3d, dots_3d[1:])):
                    if i < 10 or i % 3 == 0:  # Show first 10 steps, then every 3rd
                        self.play(
                            Create(line),
                            Create(dot),
                            run_time=0.2 if eta != 1.5 else 0.1
                        )
                    else:
                        self.add(line, dot)
                
                self.wait(1)
                
                # Check convergence
                final_t0, final_t1 = path_points[-1]
                if abs(final_t0 - 0.5) < 5 and abs(final_t1 + 1) < 5:
                    converged_text = Text("Converged!", font_size=24, color=color)
                    converged_text.next_to(lr_label, DOWN, aligned_edge=LEFT)
                    converged_text.fix_in_frame()
                    self.play(Write(converged_text))
                    self.wait(1)
                    self.play(FadeOut(converged_text))
                else:
                    diverged_text = Text("Diverged!", font_size=24, color=color)
                    diverged_text.next_to(lr_label, DOWN, aligned_edge=LEFT)
                    diverged_text.fix_in_frame()
                    self.play(Write(diverged_text))
                    self.wait(1)
                    self.play(FadeOut(diverged_text))

            self.interactive_embed()  # For debugging and interaction
            
            # Clean up for next learning rate
            self.wait(1)
            self.remove(dots_3d)
            self.play(
                FadeOut(path_3d),
                FadeOut(lr_label)
            )
            self.wait(0.5)
        
        # Final comparison: show all paths together
        comparison_title = Text("Comparison", font_size=32, color=YELLOW)
        comparison_title.to_edge(LEFT).shift(UP * 2.5)
        comparison_title.fix_in_frame()
        self.play(Write(comparison_title))

        paths = VGroup()
        
        # Recreate all paths (simplified - just final paths)
        for lr_config in learning_rates:
            eta = lr_config["eta"]
            color = lr_config["color"]
            
            theta_0 = theta_0_init
            theta_1 = theta_1_init
            path_points = [(theta_0, theta_1)]
            
            for _ in range(50):
                grad_t0, grad_t1 = gradient(theta_0, theta_1)
                theta_0 = theta_0 - eta * grad_t0
                theta_1 = theta_1 - eta * grad_t1
                path_points.append((theta_0, theta_1))
                
                if np.sqrt(grad_t0**2 + grad_t1**2) < 0.01:
                    break
                if abs(theta_0) > 10 or abs(theta_1) > 10:
                    break
            
            # Create simplified path
            path_3d = VGroup()
            for i in range(len(path_points) - 1):
                t0_start, t1_start = path_points[i]
                t0_end, t1_end = path_points[i + 1]
                start_point = axes.c2p(t0_start, t1_start, loss_function(t0_start, t1_start))
                end_point = axes.c2p(t0_end, t1_end, loss_function(t0_end, t1_end))
                line = Line(start_point, end_point, color=color, stroke_width=10)
                path_3d.add(line)

            paths.add(path_3d)
            
            self.play(Create(path_3d), run_time=0.5)
        
        # Add legend
        legend = VGroup()
        for lr_config in learning_rates:
            legend_item = VGroup(
                Line(ORIGIN, RIGHT * 0.5, color=lr_config["color"], stroke_width=3),
                Text(lr_config["name"], font_size=18, color=lr_config["color"])
            ).arrange(RIGHT, buff=0.2)
            legend.add(legend_item)
        
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_edge(LEFT).shift(DOWN * 1.5)
        legend.fix_in_frame()
        self.play(FadeIn(legend))
        
        # Continue gentle camera rotation for final view
        self.wait(3)
        
        # Stop rotation
        self.stop_ambient_camera_rotation()
        self.wait(1)
        
        # Fade out everything that is not a dot_3d (render errors...)
        self.play(
            FadeOut(axes_group),
            FadeOut(equation),
            FadeOut(comparison_title),
            FadeOut(legend),
            *[FadeOut(path) for path in paths]
        )
        self.remove(surface)
        self.wait(1)
    
