from manim import *
import numpy as np
from scipy import stats


class EMVisualization1D(Scene):
    """Visualize the EM algorithm fitting a 1D Gaussian Mixture Model."""
    
    def construct(self):
        # Title
        title = Text("EM Algorithm: 1D Gaussian Mixture", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Generate synthetic data from two Gaussians
        np.random.seed(42)
        n1, n2 = 30, 25
        true_mu1, true_sigma1 = -2.0, 0.8
        true_mu2, true_sigma2 = 2.5, 1.0
        
        data1 = np.random.normal(true_mu1, true_sigma1, n1)
        data2 = np.random.normal(true_mu2, true_sigma2, n2)
        data = np.concatenate([data1, data2])
        n = len(data)
        
        # Create axes
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[0, 0.5, 0.1],
            x_length=11,
            y_length=4.5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 0.8)
        
        x_label = MathTex("x", font_size=28).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("p(x)", font_size=28).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Plot data points
        data_dots = VGroup()
        for x in data:
            dot = Dot(axes.c2p(x, 0.02), color=WHITE, radius=0.05)
            data_dots.add(dot)
        
        data_label = Text("Observed data", font_size=24, color=WHITE)
        data_label.to_edge(DOWN)
        
        self.play(Create(data_dots), Write(data_label), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(data_label))
        
        # Initialize GMM parameters (deliberately poor initialization)
        mu = np.array([0.0, 1.0])  # Initial means
        sigma = np.array([2.0, 2.0])  # Initial standard deviations
        pi = np.array([0.5, 0.5])  # Initial mixing weights
        K = 2
        
        colors = [BLUE, ORANGE]
        
        # Show initial state
        def create_mixture_curves(mu, sigma, pi, axes, colors):
            """Create curves for each component and the mixture."""
            curves = VGroup()
            
            # Individual components (stroke only, no fill)
            for k in range(K):
                def component_pdf(x, mu_k=mu[k], sigma_k=sigma[k], pi_k=pi[k]):
                    return pi_k * stats.norm.pdf(x, mu_k, sigma_k)
                
                curve = axes.plot(
                    component_pdf,
                    color=colors[k],
                    x_range=[-6, 6],
                    stroke_width=3,
                    use_smoothing=False
                )
                curve.set_fill(opacity=0)
                curves.add(curve)
            
            # Mixture curve (stroke only, no fill)
            def mixture_pdf(x):
                return sum(pi[k] * stats.norm.pdf(x, mu[k], sigma[k]) for k in range(K))
            
            mixture_curve = axes.plot(
                mixture_pdf,
                color=WHITE,
                x_range=[-6, 6],
                stroke_width=4,
                stroke_opacity=0.8,
                use_smoothing=False
            )
            mixture_curve.set_fill(opacity=0)
            curves.add(mixture_curve)
            
            return curves
        
        # Create initial curves
        curves = create_mixture_curves(mu, sigma, pi, axes, colors)
        
        # Parameter display
        param_text = VGroup()
        for k in range(K):
            text = MathTex(
                f"\\mu_{k+1} = {mu[k]:.2f}, \\sigma_{k+1} = {sigma[k]:.2f}, \\pi_{k+1} = {pi[k]:.2f}",
                font_size=24,
                color=colors[k]
            )
            param_text.add(text)
        param_text.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        param_text.to_corner(UR).shift(DOWN * 0.8)
        
        iteration_label = Text("Initialization", font_size=28, color=YELLOW)
        iteration_label.to_corner(UL).shift(DOWN * 0.8)
        
        self.play(
            Create(curves),
            Write(param_text),
            Write(iteration_label)
        )
        self.wait(1.5)
        
        # EM iterations
        num_iterations = 20
        
        for t in range(num_iterations):
            # E-step: Compute responsibilities
            gamma = np.zeros((n, K))
            for i in range(n):
                for k in range(K):
                    gamma[i, k] = pi[k] * stats.norm.pdf(data[i], mu[k], sigma[k])
                gamma[i, :] /= gamma[i, :].sum()
            
            # Show responsibilities by coloring points
            if t == 0:
                # First iteration: show E-step animation
                e_step_label = Text("E-Step: Compute responsibilities", font_size=24, color=GREEN)
                e_step_label.to_edge(DOWN)
                self.play(Write(e_step_label))
                
                # Color points by responsibility
                new_dots = VGroup()
                for i, x in enumerate(data):
                    # Blend colors based on responsibilities
                    r1, r2 = gamma[i, 0], gamma[i, 1]
                    # Create bicolor representation
                    if r1 > r2:
                        dot_color = interpolate_color(colors[1], colors[0], r1)
                    else:
                        dot_color = interpolate_color(colors[0], colors[1], r2)
                    dot = Dot(axes.c2p(x, 0.02), color=dot_color, radius=0.05)
                    new_dots.add(dot)
                
                self.play(Transform(data_dots, new_dots), run_time=1)
                self.wait(0.5)
                self.play(FadeOut(e_step_label))
            else:
                # Update dot colors silently
                new_dots = VGroup()
                for i, x in enumerate(data):
                    r1, r2 = gamma[i, 0], gamma[i, 1]
                    if r1 > r2:
                        dot_color = interpolate_color(colors[1], colors[0], r1)
                    else:
                        dot_color = interpolate_color(colors[0], colors[1], r2)
                    dot = Dot(axes.c2p(x, 0.02), color=dot_color, radius=0.05)
                    new_dots.add(dot)
                self.play(Transform(data_dots, new_dots), run_time=0.3)
            
            # M-step: Update parameters
            N_k = gamma.sum(axis=0)
            
            new_mu = np.zeros(K)
            new_sigma = np.zeros(K)
            new_pi = np.zeros(K)
            
            for k in range(K):
                new_mu[k] = np.sum(gamma[:, k] * data) / N_k[k]
                new_sigma[k] = np.sqrt(np.sum(gamma[:, k] * (data - new_mu[k])**2) / N_k[k])
                new_pi[k] = N_k[k] / n
            
            mu, sigma, pi = new_mu, new_sigma, new_pi
            
            # Calculate log-likelihood
            log_likelihood = 0
            for i in range(n):
                ll_i = sum(pi[k] * stats.norm.pdf(data[i], mu[k], sigma[k]) for k in range(K))
                log_likelihood += np.log(ll_i)
            
            # Create new curves
            new_curves = create_mixture_curves(mu, sigma, pi, axes, colors)
            
            # Update parameter display
            new_param_text = VGroup()
            for k in range(K):
                text = MathTex(
                    f"\\mu_{k+1} = {mu[k]:.2f}, \\sigma_{k+1} = {sigma[k]:.2f}, \\pi_{k+1} = {pi[k]:.2f}",
                    font_size=24,
                    color=colors[k]
                )
                new_param_text.add(text)
            new_param_text.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            new_param_text.to_corner(UR).shift(DOWN * 0.8)
            
            # Update iteration label
            new_iteration_label = Text(f"Iteration {t+1}", font_size=28, color=YELLOW)
            new_iteration_label.to_corner(UL).shift(DOWN * 0.8)
            
            # Log-likelihood display
            ll_text = MathTex(
                f"\\log L = {log_likelihood:.2f}",
                font_size=24,
                color=GREEN
            )
            ll_text.next_to(new_param_text, DOWN, buff=0.3, aligned_edge=LEFT)
            
            if t == 0:
                m_step_label = Text("M-Step: Update parameters", font_size=24, color=BLUE)
                m_step_label.to_edge(DOWN)
                self.play(Write(m_step_label))
                self.wait(0.3)
                
                self.play(
                    Transform(curves, new_curves),
                    Transform(param_text, new_param_text),
                    Transform(iteration_label, new_iteration_label),
                    Write(ll_text),
                    run_time=1
                )
                self.wait(0.5)
                self.play(FadeOut(m_step_label))
                prev_ll_text = ll_text
            else:
                new_ll_text = MathTex(
                    f"\\log L = {log_likelihood:.2f}",
                    font_size=24,
                    color=GREEN
                )
                new_ll_text.next_to(new_param_text, DOWN, buff=0.3, aligned_edge=LEFT)
                
                # Remove old labels and add new ones (avoids OpenGL triangulation bugs)
                self.remove(iteration_label, prev_ll_text)
                self.add(new_iteration_label, new_ll_text)
                iteration_label = new_iteration_label
                prev_ll_text = new_ll_text
                
                self.play(
                    Transform(curves, new_curves),
                    Transform(param_text, new_param_text),
                    run_time=0.6
                )
            
            self.wait(0.5)
        
        # Final message
        converged_text = Text("Converged!", font_size=32, color=GREEN)
        converged_text.to_edge(DOWN)
        self.play(Write(converged_text))
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class EMStepByStep(Scene):
    """Step-by-step visualization of E-step and M-step."""
    
    def construct(self):
        # Title
        title = Text("EM Algorithm: E-Step and M-Step", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Generate data
        np.random.seed(123)
        data = np.concatenate([
            np.random.normal(-2, 0.7, 15),
            np.random.normal(2, 0.9, 20)
        ])
        n = len(data)
        
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 0.6, 0.1],
            x_length=10,
            y_length=4,
            axis_config={"color": WHITE},
            tips=False
        )
        axes.shift(DOWN * 1)
        
        self.play(Create(axes))
        
        # Plot data
        data_dots = VGroup()
        for x in data:
            dot = Dot(axes.c2p(x, 0.02), color=WHITE, radius=0.06)
            data_dots.add(dot)
        
        self.play(Create(data_dots))
        self.wait(1)
        
        # Initialize
        mu = np.array([-1.0, 1.0])
        sigma = np.array([1.5, 1.5])
        pi = np.array([0.5, 0.5])
        K = 2
        colors = [BLUE, RED]
        
        # Show initial Gaussians
        def plot_components(mu, sigma, pi, axes):
            curves = VGroup()
            for k in range(K):
                def pdf(x, mu_k=mu[k], sigma_k=sigma[k], pi_k=pi[k]):
                    return pi_k * stats.norm.pdf(x, mu_k, sigma_k)
                curve = axes.plot(pdf, color=colors[k], x_range=[-5, 5], stroke_width=3, use_smoothing=False)
                curve.set_fill(opacity=0)
                curves.add(curve)
            return curves
        
        curves = plot_components(mu, sigma, pi, axes)
        
        mu_lines = VGroup()
        for k in range(K):
            line = DashedLine(
                axes.c2p(mu[k], 0),
                axes.c2p(mu[k], 0.5),
                color=colors[k],
                stroke_width=2
            )
            mu_lines.add(line)
        
        self.play(Create(curves), Create(mu_lines))
        self.wait(1)
        
        # E-step explanation
        e_step_title = Text("E-Step: Compute Responsibilities", font_size=28, color=GREEN)
        e_step_title.next_to(title, DOWN, buff=0.3)
        
        e_step_eq = MathTex(
            r"\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i|\mu_j, \sigma_j)}",
            font_size=28
        )
        e_step_eq.to_corner(UL).shift(DOWN * 1.5)
        
        self.play(Write(e_step_title), Write(e_step_eq))
        self.wait(1)
        
        # Compute and show responsibilities
        gamma = np.zeros((n, K))
        for i in range(n):
            for k in range(K):
                gamma[i, k] = pi[k] * stats.norm.pdf(data[i], mu[k], sigma[k])
            gamma[i, :] /= gamma[i, :].sum()
        
        # Animate points changing color
        colored_dots = VGroup()
        for i, x in enumerate(data):
            color = interpolate_color(colors[0], colors[1], gamma[i, 1])
            dot = Dot(axes.c2p(x, 0.02), color=color, radius=0.06)
            colored_dots.add(dot)
        
        self.play(Transform(data_dots, colored_dots), run_time=1.5)
        self.wait(1)
        
        # M-step
        self.play(FadeOut(e_step_title), FadeOut(e_step_eq))
        
        m_step_title = Text("M-Step: Update Parameters", font_size=28, color=BLUE)
        m_step_title.next_to(title, DOWN, buff=0.3)
        
        m_step_eqs = MathTex(
            r"\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad "
            r"\sigma_k^2 = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)^2}{\sum_i \gamma_{ik}}",
            font_size=24
        )
        m_step_eqs.to_corner(UL).shift(DOWN * 1.5)
        
        self.play(Write(m_step_title), Write(m_step_eqs))
        self.wait(1)
        
        # Update parameters
        N_k = gamma.sum(axis=0)
        new_mu = np.array([np.sum(gamma[:, k] * data) / N_k[k] for k in range(K)])
        new_sigma = np.array([
            np.sqrt(np.sum(gamma[:, k] * (data - new_mu[k])**2) / N_k[k])
            for k in range(K)
        ])
        new_pi = N_k / n
        
        mu, sigma, pi = new_mu, new_sigma, new_pi
        
        # Animate update
        new_curves = plot_components(mu, sigma, pi, axes)
        new_mu_lines = VGroup()
        for k in range(K):
            line = DashedLine(
                axes.c2p(mu[k], 0),
                axes.c2p(mu[k], 0.5),
                color=colors[k],
                stroke_width=2
            )
            new_mu_lines.add(line)
        
        self.play(
            Transform(curves, new_curves),
            Transform(mu_lines, new_mu_lines),
            run_time=1.5
        )
        self.wait(1)
        
        # Repeat indication
        repeat_text = Text("Repeat until convergence...", font_size=28, color=YELLOW)
        repeat_text.to_edge(DOWN)
        self.play(Write(repeat_text))
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)

if __name__ == "__main__":
    from manim import config
    config.preview = True
    
    # Run the main EM visualization
    scene = EMVisualization1D()
    scene.render()
