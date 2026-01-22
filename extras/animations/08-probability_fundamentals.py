from manim import *
import numpy as np
from scipy import stats


class BernoulliDistributionVisualization(Scene):
    """Visualize Bernoulli distribution PMF for different p values."""
    
    def construct(self):
        # Title
        title = Text("Bernoulli Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create axes for bar chart
        axes = Axes(
            x_range=[-0.5, 1.5, 1],
            y_range=[0, 1.2, 0.2],
            x_length=6,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("x", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(x)", font_size=32).next_to(axes.y_axis, UP)
        
        # Add x-axis tick labels
        x_0_label = MathTex("0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.2)
        x_1_label = MathTex("1", font_size=24).next_to(axes.c2p(1, 0), DOWN, buff=0.2)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Write(x_0_label), Write(x_1_label))
        self.wait(1)
        
        # Show PMF equation
        pmf_eq = MathTex(
            r"p_X(x) = p^x (1-p)^{1-x}",
            font_size=32,
            color=WHITE
        )
        pmf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pmf_eq))
        
        # Animate through different p values
        p_values = [0.5, 0.3, 0.7, 0.9, 0.1]
        colors = [BLUE, GREEN, ORANGE, RED, PURPLE]
        
        bars = None
        p_label = None
        
        for p, color in zip(p_values, colors):
            # Create bars for x=0 and x=1
            bar_0 = Rectangle(
                width=0.4,
                height=axes.c2p(0, 1-p)[1] - axes.c2p(0, 0)[1],
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=color
            )
            bar_0.move_to(axes.c2p(0, (1-p)/2), aligned_edge=DOWN)
            bar_0.align_to(axes.c2p(0, 0), DOWN)
            
            bar_1 = Rectangle(
                width=0.4,
                height=axes.c2p(0, p)[1] - axes.c2p(0, 0)[1],
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=color
            )
            bar_1.move_to(axes.c2p(1, p/2), aligned_edge=DOWN)
            bar_1.align_to(axes.c2p(1, 0), DOWN)
            
            new_bars = VGroup(bar_0, bar_1)
            
            new_p_label = MathTex(
                f"p = {p}",
                font_size=32,
                color=color
            )
            new_p_label.next_to(pmf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if bars is None:
                bars = new_bars
                p_label = new_p_label
                self.play(Create(bars), Write(p_label))
            else:
                self.play(
                    Transform(bars, new_bars),
                    Transform(p_label, new_p_label)
                )
            
            self.wait(1)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class BinomialDistributionVisualization(Scene):
    """Visualize Binomial distribution PMF for different n and p values."""
    
    def construct(self):
        # Title
        title = Text("Binomial Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show PMF equation
        pmf_eq = MathTex(
            r"p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}",
            font_size=32,
            color=WHITE
        )
        pmf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pmf_eq))
        
        # Create axes
        axes = Axes(
            x_range=[-1, 21, 2],
            y_range=[0, 0.35, 0.05],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("k", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(k)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Parameters to visualize
        params = [
            (20, 0.5, BLUE),
            (20, 0.3, GREEN),
            (20, 0.7, ORANGE),
            (10, 0.5, RED),
        ]
        
        bars = None
        param_label = None
        
        for n, p, color in params:
            # Compute PMF values
            k_values = np.arange(0, n + 1)
            pmf_values = stats.binom.pmf(k_values, n, p)
            
            # Create bars
            new_bars = VGroup()
            for k, pmf_val in zip(k_values, pmf_values):
                if pmf_val > 0.001:  # Only draw visible bars
                    bar_height = axes.c2p(0, pmf_val)[1] - axes.c2p(0, 0)[1]
                    bar = Rectangle(
                        width=0.3,
                        height=bar_height,
                        fill_color=color,
                        fill_opacity=0.8,
                        stroke_color=color,
                        stroke_width=1
                    )
                    bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)
                    new_bars.add(bar)
            
            new_param_label = MathTex(
                f"n = {n}, p = {p}",
                font_size=32,
                color=color
            )
            new_param_label.next_to(pmf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if bars is None:
                bars = new_bars
                param_label = new_param_label
                self.play(Create(bars), Write(param_label))
            else:
                self.play(
                    Transform(bars, new_bars),
                    Transform(param_label, new_param_label)
                )
            
            self.wait(1.5)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class PoissonDistributionVisualization(Scene):
    """Visualize Poisson distribution PMF for different lambda values."""
    
    def construct(self):
        # Title
        title = Text("Poisson Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show PMF equation
        pmf_eq = MathTex(
            r"p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}",
            font_size=32,
            color=WHITE
        )
        pmf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pmf_eq))
        
        # Create axes
        axes = Axes(
            x_range=[-1, 20, 2],
            y_range=[0, 0.4, 0.1],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("k", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(k)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Lambda values to visualize
        lambdas = [1, 3, 5, 10]
        colors = [BLUE, GREEN, ORANGE, RED]
        
        bars = None
        lambda_label = None
        
        for lam, color in zip(lambdas, colors):
            # Compute PMF values
            k_values = np.arange(0, 20)
            pmf_values = stats.poisson.pmf(k_values, lam)
            
            # Create bars
            new_bars = VGroup()
            for k, pmf_val in zip(k_values, pmf_values):
                if pmf_val > 0.001:
                    bar_height = axes.c2p(0, pmf_val)[1] - axes.c2p(0, 0)[1]
                    bar = Rectangle(
                        width=0.35,
                        height=bar_height,
                        fill_color=color,
                        fill_opacity=0.8,
                        stroke_color=color,
                        stroke_width=1
                    )
                    bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)
                    new_bars.add(bar)
            
            new_lambda_label = MathTex(
                f"\\lambda = {lam}",
                font_size=32,
                color=color
            )
            new_lambda_label.next_to(pmf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if bars is None:
                bars = new_bars
                lambda_label = new_lambda_label
                self.play(Create(bars), Write(lambda_label))
            else:
                self.play(
                    Transform(bars, new_bars),
                    Transform(lambda_label, new_lambda_label)
                )
            
            self.wait(1.5)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class NormalDistributionVisualization(Scene):
    """Visualize Normal (Gaussian) distribution PDF for different mu and sigma values."""
    
    def construct(self):
        # Title
        title = Text("Normal (Gaussian) Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show PDF equation
        pdf_eq = MathTex(
            r"p_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
            font_size=32,
            color=WHITE
        )
        pdf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pdf_eq))
        
        # Create axes
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 0.8, 0.1],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("x", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(x)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Parameters to visualize: (mu, sigma, color)
        params = [
            (0, 1, BLUE, r"\mu = 0, \sigma = 1"),
            (0, 0.5, GREEN, r"\mu = 0, \sigma = 0.5"),
            (0, 2, ORANGE, r"\mu = 0, \sigma = 2"),
            (2, 1, RED, r"\mu = 2, \sigma = 1"),
            (-2, 1.5, PURPLE, r"\mu = -2, \sigma = 1.5"),
        ]
        
        curve = None
        param_label = None
        
        for mu, sigma, color, label_str in params:
            # Define normal PDF
            def normal_pdf(x, mu=mu, sigma=sigma):
                return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
            
            # Plot curve
            new_curve = axes.plot(
                normal_pdf,
                color=color,
                x_range=[-6, 6],
                stroke_width=4
            )
            
            # Fill under curve
            area = axes.get_area(new_curve, x_range=[-6, 6], color=color, opacity=0.3)
            new_curve = VGroup(area, new_curve)
            
            new_param_label = MathTex(
                label_str,
                font_size=32,
                color=color
            )
            new_param_label.next_to(pdf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if curve is None:
                curve = new_curve
                param_label = new_param_label
                self.play(Create(curve), Write(param_label))
            else:
                self.play(
                    Transform(curve, new_curve),
                    Transform(param_label, new_param_label)
                )
            
            self.wait(1.5)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class ExponentialDistributionVisualization(Scene):
    """Visualize Exponential distribution PDF for different lambda values."""
    
    def construct(self):
        # Title
        title = Text("Exponential Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show PDF equation
        pdf_eq = MathTex(
            r"p_X(x) = \lambda e^{-\lambda x}, \quad x \geq 0",
            font_size=32,
            color=WHITE
        )
        pdf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pdf_eq))
        
        # Create axes
        axes = Axes(
            x_range=[-0.5, 5, 1],
            y_range=[0, 2.5, 0.5],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("x", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(x)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Lambda values to visualize
        lambdas = [0.5, 1, 1.5, 2]
        colors = [BLUE, GREEN, ORANGE, RED]
        
        curve = None
        lambda_label = None
        
        for lam, color in zip(lambdas, colors):
            # Define exponential PDF
            def exp_pdf(x, lam=lam):
                return np.where(x >= 0, lam * np.exp(-lam * x), 0)
            
            # Plot curve
            new_curve = axes.plot(
                exp_pdf,
                color=color,
                x_range=[0.001, 5],
                stroke_width=4
            )
            
            # Fill under curve
            area = axes.get_area(new_curve, x_range=[0, 5], color=color, opacity=0.3)
            new_curve = VGroup(area, new_curve)
            
            new_lambda_label = MathTex(
                f"\\lambda = {lam}",
                font_size=32,
                color=color
            )
            new_lambda_label.next_to(pdf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if curve is None:
                curve = new_curve
                lambda_label = new_lambda_label
                self.play(Create(curve), Write(lambda_label))
            else:
                self.play(
                    Transform(curve, new_curve),
                    Transform(lambda_label, new_lambda_label)
                )
            
            self.wait(1.5)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class UniformDistributionVisualization(Scene):
    """Visualize Uniform distribution PDF for different a and b values."""
    
    def construct(self):
        # Title
        title = Text("Uniform Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show PDF equation
        pdf_eq = MathTex(
            r"p_X(x) = \frac{1}{b-a}, \quad a \leq x \leq b",
            font_size=32,
            color=WHITE
        )
        pdf_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(pdf_eq))
        
        # Create axes
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[0, 1.2, 0.2],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("x", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p_X(x)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)
        
        # Parameters: (a, b, color)
        params = [
            (0, 1, BLUE),
            (1, 3, GREEN),
            (0, 4, ORANGE),
            (2, 3, RED),
        ]
        
        rect = None
        param_label = None
        
        for a, b, color in params:
            height = 1 / (b - a)
            
            # Create rectangle representing uniform PDF
            rect_width = axes.c2p(b, 0)[0] - axes.c2p(a, 0)[0]
            rect_height = axes.c2p(0, height)[1] - axes.c2p(0, 0)[1]
            
            new_rect = Rectangle(
                width=rect_width,
                height=rect_height,
                fill_color=color,
                fill_opacity=0.5,
                stroke_color=color,
                stroke_width=3
            )
            new_rect.move_to(axes.c2p((a + b) / 2, height / 2))
            
            new_param_label = MathTex(
                f"a = {a}, b = {b}",
                font_size=32,
                color=color
            )
            new_param_label.next_to(pdf_eq, DOWN, aligned_edge=LEFT, buff=0.5)
            
            if rect is None:
                rect = new_rect
                param_label = new_param_label
                self.play(Create(rect), Write(param_label))
            else:
                self.play(
                    Transform(rect, new_rect),
                    Transform(param_label, new_param_label)
                )
            
            self.wait(1.5)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class JointDistributionVisualization(ThreeDScene):
    """Visualize a bivariate normal joint distribution."""
    
    def construct(self):
        # Set up 3D camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Title
        title = Text("Joint Distribution: Bivariate Normal", font_size=28)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 0.2, 0.05],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        
        axes_labels = axes.get_axis_labels(
            x_label="X", y_label="Y", z_label="p(x,y)"
        )
        
        self.play(Create(axes), Write(axes_labels))
        self.wait()
        
        # Parameters for bivariate normal
        mu_x, mu_y = 0, 0
        sigma_x, sigma_y = 1, 1
        rho = 0  # Correlation coefficient
        
        # Create surface for bivariate normal
        def bivariate_normal(u, v, rho=rho):
            x, y = u, v
            z = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * \
                np.exp(-1 / (2 * (1 - rho**2)) * (
                    ((x - mu_x)**2 / sigma_x**2) +
                    ((y - mu_y)**2 / sigma_y**2) -
                    (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
                ))
            return axes.c2p(x, y, z)
        
        surface = Surface(
            bivariate_normal,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        self.play(Create(surface))
        self.wait()
        
        # Show equation
        eq = MathTex(
            r"p_{X,Y}(x,y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}}",
            font_size=24
        ).to_corner(UL).shift(DOWN * 1.5)
        self.add_fixed_in_frame_mobjects(eq)
        self.play(Write(eq))
        
        # Rotate camera to show joint distribution from different angles
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        
        # Animate changing correlation
        correlations = [0.5, 0.8, -0.5, 0]
        
        for rho in correlations:
            def new_bivariate_normal(u, v, rho=rho):
                x, y = u, v
                denom = 1 - rho**2
                if denom <= 0:
                    denom = 0.01
                z = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(denom))) * \
                    np.exp(-1 / (2 * denom) * (
                        ((x - mu_x)**2 / sigma_x**2) +
                        ((y - mu_y)**2 / sigma_y**2) -
                        (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
                    ))
                return axes.c2p(x, y, z)
            
            new_surface = Surface(
                new_bivariate_normal,
                u_range=[-3, 3],
                v_range=[-3, 3],
                resolution=(30, 30),
                fill_opacity=0.7,
                checkerboard_colors=[BLUE_D, BLUE_E]
            )
            
            rho_label = MathTex(f"\\rho = {rho}", font_size=28, color=YELLOW)
            rho_label.next_to(eq, DOWN, aligned_edge=LEFT)
            self.add_fixed_in_frame_mobjects(rho_label)
            
            self.play(Transform(surface, new_surface), Write(rho_label))
            self.wait(2)
            self.remove(rho_label)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class ConditionalDistributionVisualization(Scene):
    """Visualize conditional distribution from a joint distribution."""
    
    def construct(self):
        # Title
        title = Text("Conditional Distribution", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.8, 0.1],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.5)
        
        # Add labels
        x_label = MathTex("x", font_size=32).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p(x|y)", font_size=32).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Show conditional formula
        cond_eq = MathTex(
            r"p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}",
            font_size=32,
            color=WHITE
        )
        cond_eq.to_edge(LEFT).shift(UP * 2)
        self.play(Write(cond_eq))
        self.wait(1)
        
        # Show marginal distribution first (unconditional)
        def marginal_pdf(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
        
        marginal_curve = axes.plot(
            marginal_pdf,
            color=BLUE,
            x_range=[-4, 4],
            stroke_width=4
        )
        marginal_area = axes.get_area(marginal_curve, x_range=[-4, 4], color=BLUE, opacity=0.3)
        marginal_group = VGroup(marginal_area, marginal_curve)
        
        marginal_label = MathTex(r"p_X(x) = \mathcal{N}(0, 1)", font_size=28, color=BLUE)
        marginal_label.next_to(cond_eq, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Create(marginal_group), Write(marginal_label))
        self.wait(1)
        
        # Now show conditional distributions for different y values
        # Assuming X|Y=y ~ N(rho*y, 1-rho^2) for bivariate normal
        rho = 0.7
        
        y_values = [0, 1, -1, 2]
        colors = [GREEN, ORANGE, PURPLE, RED]
        
        self.play(FadeOut(marginal_group), FadeOut(marginal_label))
        
        curves = VGroup()
        labels = VGroup()
        
        for y_val, color in zip(y_values, colors):
            # Conditional mean and variance
            cond_mean = rho * y_val
            cond_var = 1 - rho**2
            cond_std = np.sqrt(cond_var)
            
            def cond_pdf(x, mu=cond_mean, sigma=cond_std):
                return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
            
            curve = axes.plot(
                cond_pdf,
                color=color,
                x_range=[-4, 4],
                stroke_width=3
            )
            curves.add(curve)
            
            label = MathTex(
                f"p(x|y={y_val})",
                font_size=24,
                color=color
            )
            labels.add(label)
        
        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        labels.next_to(cond_eq, DOWN, aligned_edge=LEFT, buff=0.5)
        
        for curve, label in zip(curves, labels):
            self.play(Create(curve), Write(label), run_time=0.8)
        
        self.wait(2)
        
        # Add explanation
        explanation = Text(
            "The conditional distribution shifts based on the observed value of Y",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)

class MLEVisualization(Scene):
    """Visualize Maximum Likelihood Estimation."""
    
    def construct(self):
        # Title
        title = Text("Maximum Likelihood Estimation (MLE)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show MLE equation
        mle_eq = MathTex(
            r"\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^{n} p(x_i | \theta)",
            font_size=32
        )
        mle_eq.next_to(title, DOWN, buff=0.5)
        self.play(Write(mle_eq))
        
        log_mle_eq = MathTex(
            r"= \arg\max_{\theta} \sum_{i=1}^{n} \log p(x_i | \theta)",
            font_size=32
        )
        log_mle_eq.next_to(mle_eq, DOWN, buff=0.3)
        self.play(Write(log_mle_eq))
        self.wait(1)
        
        # Create axes for visualization
        axes = Axes(
            x_range=[-4, 8, 1],
            y_range=[0, 0.5, 0.1],
            x_length=10,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 1.2)
        
        x_label = MathTex("x", font_size=28).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p(x|\\mu)", font_size=28).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Generate some sample data
        np.random.seed(42)
        true_mu = 3
        samples = np.random.normal(true_mu, 1, 10)
        
        # Plot sample points
        sample_dots = VGroup()
        for x in samples:
            dot = Dot(axes.c2p(x, 0.02), color=YELLOW, radius=0.08)
            sample_dots.add(dot)
        
        sample_label = Text("Observed samples", font_size=24, color=YELLOW)
        sample_label.next_to(axes, DOWN, buff=0.3)
        
        self.play(Create(sample_dots), Write(sample_label))
        self.wait(1)
        
        # Animate different mu values and show likelihood
        mu_values = [0, 1, 2, 3, 4, 5]
        
        curve = None
        mu_label = None
        likelihood_label = None
        
        for mu in mu_values:
            # Define normal PDF with this mu
            def normal_pdf(x, mu=mu):
                return (1 / np.sqrt(2 * np.pi)) * np.exp(-((x - mu)**2) / 2)
            
            # Calculate log-likelihood
            log_likelihood = np.sum(np.log(normal_pdf(samples, mu)))
            
            new_curve = axes.plot(
                normal_pdf,
                color=BLUE,
                x_range=[-4, 8],
                stroke_width=3
            )
            new_curve_area = axes.get_area(new_curve, x_range=[-4, 8], color=BLUE, opacity=0.2)
            new_curve_group = VGroup(new_curve_area, new_curve)
            
            new_mu_label = MathTex(
                f"\\mu = {mu}",
                font_size=28,
                color=BLUE
            )
            new_mu_label.to_corner(UR).shift(DOWN * 1.5)
            
            new_likelihood_label = MathTex(
                f"\\log L = {log_likelihood:.2f}",
                font_size=28,
                color=GREEN
            )
            new_likelihood_label.next_to(new_mu_label, DOWN, buff=0.3)
            
            if curve is None:
                curve = new_curve_group
                mu_label = new_mu_label
                likelihood_label = new_likelihood_label
                self.play(Create(curve), Write(mu_label), Write(likelihood_label))
            else:
                self.play(
                    Transform(curve, new_curve_group),
                    Transform(mu_label, new_mu_label),
                    Transform(likelihood_label, new_likelihood_label)
                )
            
            self.wait(0.8)
        
        # Show the MLE result
        mle_result = MathTex(
            f"\\hat{{\\mu}}_{{\\text{{MLE}}}} = \\bar{{x}} = {np.mean(samples):.2f}",
            font_size=32,
            color=GREEN
        )
        mle_result.to_edge(DOWN)
        self.play(Write(mle_result))
        
        # Draw vertical line at MLE
        mle_line = DashedLine(
            axes.c2p(np.mean(samples), 0),
            axes.c2p(np.mean(samples), 0.4),
            color=GREEN
        )
        self.play(Create(mle_line))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class MAPvsMLEVisualization(Scene):
    """Compare MAP and MLE estimation."""
    
    def construct(self):
        # Title
        title = Text("MAP vs MLE", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show equations side by side
        mle_eq = MathTex(
            r"\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} p(\mathcal{D}|\theta)",
            font_size=28
        )
        map_eq = MathTex(
            r"\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(\mathcal{D}|\theta) \cdot p(\theta)",
            font_size=28
        )
        
        eqs = VGroup(mle_eq, map_eq)
        eqs.arrange(DOWN, buff=0.5)
        eqs.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(mle_eq))
        self.play(Write(map_eq))
        self.wait(1)
        
        # Highlight the difference
        prior_box = SurroundingRectangle(map_eq[-8:], color=YELLOW)
        prior_label = Text("Prior on θ", font_size=24, color=YELLOW)
        prior_label.next_to(prior_box, RIGHT)
        
        self.play(Create(prior_box), Write(prior_label))
        self.wait(1)
        
        # Create axes for visualization
        axes = Axes(
            x_range=[-2, 6, 1],
            y_range=[0, 0.8, 0.2],
            x_length=10,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 1.5)
        
        x_label = MathTex("\\theta", font_size=28).next_to(axes.x_axis, RIGHT)
        
        self.play(Create(axes), Write(x_label))
        self.play(FadeOut(prior_box), FadeOut(prior_label))
        
        # Show likelihood function
        def likelihood(theta):
            # Simulate likelihood peaked at theta=3
            return 0.5 * np.exp(-((theta - 3)**2) / 2)
        
        likelihood_curve = axes.plot(
            likelihood,
            color=BLUE,
            x_range=[-2, 6],
            stroke_width=3
        )
        likelihood_label = MathTex(
            r"p(\mathcal{D}|\theta)",
            font_size=24,
            color=BLUE
        )
        likelihood_label.next_to(axes.c2p(3, 0.5), UP)
        
        self.play(Create(likelihood_curve), Write(likelihood_label))
        self.wait(1)
        
        # Show prior (centered at 0)
        def prior(theta):
            return 0.6 * np.exp(-((theta - 0)**2) / 1)
        
        prior_curve = axes.plot(
            prior,
            color=RED,
            x_range=[-2, 6],
            stroke_width=3
        )
        prior_label = MathTex(
            r"p(\theta)",
            font_size=24,
            color=RED
        )
        prior_label.next_to(axes.c2p(0, 0.6), UP)
        
        self.play(Create(prior_curve), Write(prior_label))
        self.wait(1)
        
        # Show posterior (product)
        def posterior(theta):
            return likelihood(theta) * prior(theta) * 5  # Scaled for visualization
        
        posterior_curve = axes.plot(
            posterior,
            color=PURPLE,
            x_range=[-2, 6],
            stroke_width=4
        )
        posterior_label = MathTex(
            r"p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta) \cdot p(\theta)",
            font_size=24,
            color=PURPLE
        )
        posterior_label.next_to(axes.c2p(1.5, 0.4), UR)
        
        self.play(Create(posterior_curve), Write(posterior_label))
        self.wait(1)
        
        # Mark MLE and MAP estimates
        mle_theta = 3.0
        map_theta = 1.5  # Posterior peak is shifted towards prior
        
        mle_line = DashedLine(
            axes.c2p(mle_theta, 0),
            axes.c2p(mle_theta, likelihood(mle_theta)),
            color=BLUE
        )
        mle_dot = Dot(axes.c2p(mle_theta, 0), color=BLUE, radius=0.1)
        mle_text = MathTex(r"\hat{\theta}_{\text{MLE}}", font_size=24, color=BLUE)
        mle_text.next_to(mle_dot, DOWN)
        
        map_line = DashedLine(
            axes.c2p(map_theta, 0),
            axes.c2p(map_theta, posterior(map_theta)),
            color=PURPLE
        )
        map_dot = Dot(axes.c2p(map_theta, 0), color=PURPLE, radius=0.1)
        map_text = MathTex(r"\hat{\theta}_{\text{MAP}}", font_size=24, color=PURPLE)
        map_text.next_to(map_dot, DOWN)
        
        self.play(
            Create(mle_line), Create(mle_dot), Write(mle_text),
            Create(map_line), Create(map_dot), Write(map_text)
        )
        self.wait(1)
        
        # Explanation
        explanation = Text(
            "MAP estimate is pulled towards the prior",
            font_size=28,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class RegularizationAsBayesianPrior(Scene):
    """Visualize how regularization corresponds to Bayesian priors."""
    
    def construct(self):
        # Title
        title = Text("Regularization = Bayesian Prior", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Show the connection
        connection = VGroup()
        
        l2_reg = MathTex(
            r"\text{L2 Regularization: } \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|^2",
            font_size=28
        )
        gaussian_prior = MathTex(
            r"\Leftrightarrow \text{Gaussian Prior: } p(\mathbf{w}) \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})",
            font_size=28,
            color=BLUE
        )
        
        l1_reg = MathTex(
            r"\text{L1 Regularization: } \mathcal{L} + \lambda\|\mathbf{w}\|_1",
            font_size=28
        )
        laplace_prior = MathTex(
            r"\Leftrightarrow \text{Laplace Prior: } p(\mathbf{w}) \sim \text{Laplace}(0, \lambda^{-1})",
            font_size=28,
            color=GREEN
        )
        
        connection.add(l2_reg, gaussian_prior, l1_reg, laplace_prior)
        connection.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        connection.next_to(title, DOWN, buff=0.6)
        
        for eq in connection:
            self.play(Write(eq))
            self.wait(0.5)
        
        self.wait(1)
        
        # Create axes to show the priors
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.8, 0.2],
            x_length=8,
            y_length=3,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 2)
        
        x_label = MathTex("w", font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p(w)", font_size=24).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot Gaussian prior
        def gaussian(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
        
        gaussian_curve = axes.plot(
            gaussian,
            color=BLUE,
            x_range=[-4, 4],
            stroke_width=3
        )
        gaussian_label = Text("Gaussian (L2)", font_size=20, color=BLUE)
        gaussian_label.next_to(axes.c2p(2.5, 0.3), RIGHT)
        
        self.play(Create(gaussian_curve), Write(gaussian_label))
        self.wait(1)
        
        # Plot Laplace prior
        def laplace(x):
            return 0.5 * np.exp(-np.abs(x))
        
        laplace_curve = axes.plot(
            laplace,
            color=GREEN,
            x_range=[-4, 4],
            stroke_width=3
        )
        laplace_label = Text("Laplace (L1)", font_size=20, color=GREEN)
        laplace_label.next_to(axes.c2p(-2.5, 0.3), LEFT)
        
        self.play(Create(laplace_curve), Write(laplace_label))
        self.wait(1)
        
        # Highlight key insight
        insight = Text(
            "Laplace prior encourages sparsity (weights = 0)",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(insight))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


if __name__ == "__main__":
    # Run a specific scene for testing
    from manim import config
    config.preview = True
    
    # Example: Run NormalDistributionVisualization
    scene = NormalDistributionVisualization()
    scene.render()
