from manim import *
import numpy as np
from scipy import stats


class MonteCarloConvergence(Scene):
    """Visualize Monte Carlo estimation converging to true expectation."""
    
    def construct(self):
        # Title
        title = Text("Monte Carlo Estimation", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Problem statement
        problem = MathTex(
            r"\mathbb{E}_{p(x)}[f(x)] = \int p(x) f(x) \, dx \approx \frac{1}{L} \sum_{l=1}^{L} f(x^{(l)})",
            font_size=28
        )
        problem.next_to(title, DOWN, buff=0.3)
        self.play(Write(problem))
        self.wait(1)
        
        # Create axes for the distribution
        axes_dist = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=2.5,
            axis_config={"color": WHITE},
            tips=False
        )
        axes_dist.to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        
        x_label = MathTex("x", font_size=24).next_to(axes_dist.x_axis, RIGHT)
        
        # Plot the distribution p(x) = N(0, 1)
        def gaussian_pdf(x):
            return stats.norm.pdf(x, 0, 1)
        
        dist_curve = axes_dist.plot(
            gaussian_pdf,
            color=BLUE,
            x_range=[-4, 4],
            stroke_width=3,
            use_smoothing=False
        )
        dist_curve.set_fill(BLUE, opacity=0.3)
        
        dist_label = MathTex(r"p(x) = \mathcal{N}(0, 1)", font_size=24, color=BLUE)
        dist_label.next_to(axes_dist, UP, buff=0.1)
        
        self.play(Create(axes_dist), Write(x_label), Create(dist_curve), Write(dist_label))
        self.wait(0.5)
        
        # Define the function f(x) = x^2 (so E[f(x)] = Var(x) + E[x]^2 = 1 for N(0,1))
        func_label = MathTex(r"f(x) = x^2", font_size=24, color=ORANGE)
        func_label.next_to(dist_label, RIGHT, buff=1)
        
        true_expectation = 1.0  # E[x^2] for N(0,1)
        true_label = MathTex(r"\mathbb{E}[x^2] = 1", font_size=24, color=GREEN)
        true_label.next_to(func_label, RIGHT, buff=1)
        
        self.play(Write(func_label), Write(true_label))
        self.wait(1)
        
        # Create axes for the convergence plot
        axes_conv = Axes(
            x_range=[0, 200, 50],
            y_range=[0, 2, 0.5],
            x_length=6,
            y_length=3,
            axis_config={"color": WHITE},
            tips=False
        )
        axes_conv.to_corner(DR).shift(UP * 0.5 + LEFT * 0.5)
        
        conv_x_label = MathTex("L", font_size=24).next_to(axes_conv.x_axis, RIGHT)
        conv_y_label = MathTex(r"\hat{\mu}_L", font_size=24).next_to(axes_conv.y_axis, UP)
        
        # True value line
        true_line = DashedLine(
            axes_conv.c2p(0, true_expectation),
            axes_conv.c2p(200, true_expectation),
            color=GREEN,
            stroke_width=2
        )
        true_line_label = MathTex(r"\mathbb{E}[f(x)] = 1", font_size=20, color=GREEN)
        true_line_label.next_to(true_line, UP, buff=0.1).shift(LEFT * 2)
        
        self.play(
            Create(axes_conv), 
            Write(conv_x_label), 
            Write(conv_y_label),
            Create(true_line),
            Write(true_line_label)
        )
        self.wait(0.5)
        
        # Monte Carlo estimation animation
        np.random.seed(42)
        max_samples = 200
        
        # Pre-generate all samples
        samples = np.random.normal(0, 1, max_samples)
        f_samples = samples ** 2  # f(x) = x^2
        
        # Running estimate
        running_sum = 0
        estimates = []
        
        for l in range(1, max_samples + 1):
            running_sum += f_samples[l - 1]
            estimates.append(running_sum / l)
        
        # Create sample dots on distribution
        sample_dots = VGroup()
        
        # Estimate display
        estimate_display = MathTex(
            r"\hat{\mu}_L = 0.00",
            font_size=28,
            color=YELLOW
        )
        estimate_display.next_to(axes_conv, DOWN, buff=0.3)
        
        sample_count_display = MathTex(r"L = 0", font_size=28, color=WHITE)
        sample_count_display.next_to(estimate_display, LEFT, buff=0.5)
        
        self.play(Write(estimate_display), Write(sample_count_display))
        
        # Animate sampling process - use plot instead of VMobject for compatibility
        # Phase 1: Slow animation for first 20 samples
        convergence_dots = VGroup()
        
        for l in range(1, 21):
            x_sample = samples[l - 1]
            
            # Add sample dot on distribution
            sample_dot = Dot(
                axes_dist.c2p(x_sample, 0.0),
                color=YELLOW,
                radius=0.04
            )
            
            # Add convergence point
            conv_dot = Dot(
                axes_conv.c2p(l, estimates[l - 1]),
                color=YELLOW,
                radius=0.03
            )
            
            # Update displays
            new_estimate = MathTex(
                f"\\hat{{\\mu}}_L = {estimates[l - 1]:.2f}",
                font_size=28,
                color=YELLOW
            )
            new_estimate.next_to(axes_conv, DOWN, buff=0.3)
            
            new_count = MathTex(f"L = {l}", font_size=28, color=WHITE)
            new_count.next_to(new_estimate, LEFT, buff=0.5)
            
            # Draw line segment to previous point
            if l > 1:
                line_seg = Line(
                    axes_conv.c2p(l - 1, estimates[l - 2]),
                    axes_conv.c2p(l, estimates[l - 1]),
                    color=YELLOW,
                    stroke_width=2
                )
                self.play(
                    FadeIn(sample_dot, scale=0.5),
                    FadeIn(conv_dot),
                    Create(line_seg),
                    Transform(estimate_display, new_estimate),
                    Transform(sample_count_display, new_count),
                    run_time=0.3
                )
            else:
                self.play(
                    FadeIn(sample_dot, scale=0.5),
                    FadeIn(conv_dot),
                    Transform(estimate_display, new_estimate),
                    Transform(sample_count_display, new_count),
                    run_time=0.3
                )
            
            sample_dots.add(sample_dot)
            convergence_dots.add(conv_dot)
        
        self.wait(0.5)
        
        # Phase 2: Faster animation for samples 21-50
        for l in range(21, 51):
            x_sample = samples[l - 1]
            
            sample_dot = Dot(
                axes_dist.c2p(x_sample, 0.0),
                color=YELLOW,
                radius=0.04,
                fill_opacity=0.5
            )
            
            conv_dot = Dot(
                axes_conv.c2p(l, estimates[l - 1]),
                color=YELLOW,
                radius=0.03
            )
            
            line_seg = Line(
                axes_conv.c2p(l - 1, estimates[l - 2]),
                axes_conv.c2p(l, estimates[l - 1]),
                color=YELLOW,
                stroke_width=2
            )
            
            new_estimate = MathTex(
                f"\\hat{{\\mu}}_L = {estimates[l - 1]:.2f}",
                font_size=28,
                color=YELLOW
            )
            new_estimate.next_to(axes_conv, DOWN, buff=0.3)
            
            new_count = MathTex(f"L = {l}", font_size=28, color=WHITE)
            new_count.next_to(new_estimate, LEFT, buff=0.5)
            
            self.play(
                FadeIn(sample_dot, scale=0.5),
                FadeIn(conv_dot),
                Create(line_seg),
                Transform(estimate_display, new_estimate),
                Transform(sample_count_display, new_count),
                run_time=0.1
            )
            sample_dots.add(sample_dot)
            convergence_dots.add(conv_dot)
        
        self.wait(0.3)
        
        # Phase 3: Batch the rest (51-200)
        batch_size = 10
        for start in range(51, max_samples + 1, batch_size):
            end = min(start + batch_size, max_samples + 1)
            l = end - 1
            
            # Create all line segments for this batch
            batch_lines = VGroup()
            for i in range(start, end):
                line_seg = Line(
                    axes_conv.c2p(i - 1, estimates[i - 2]),
                    axes_conv.c2p(i, estimates[i - 1]),
                    color=YELLOW,
                    stroke_width=2
                )
                batch_lines.add(line_seg)
            
            new_estimate = MathTex(
                f"\\hat{{\\mu}}_L = {estimates[l - 1]:.2f}",
                font_size=28,
                color=YELLOW
            )
            new_estimate.next_to(axes_conv, DOWN, buff=0.3)
            
            new_count = MathTex(f"L = {l}", font_size=28, color=WHITE)
            new_count.next_to(new_estimate, LEFT, buff=0.5)
            
            self.play(
                Create(batch_lines),
                Transform(estimate_display, new_estimate),
                Transform(sample_count_display, new_count),
                run_time=0.15
            )
        
        self.wait(1)
        
        # Show convergence message
        converged_text = Text("Converged to true value!", font_size=28, color=GREEN)
        converged_text.to_edge(DOWN)
        
        self.play(Write(converged_text))
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


class VAEArchitecture(Scene):
    """Visualize the full VAE architecture."""
    
    def construct(self):
        # Title
        title = Text("Variational Autoencoder Architecture", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Input image placeholder
        input_box = Rectangle(width=1.5, height=1.5, color=WHITE, fill_opacity=0.2)
        input_box.move_to(LEFT * 5)
        input_label = MathTex(r"\mathbf{x}", font_size=28)
        input_label.next_to(input_box, DOWN, buff=0.2)
        
        # Encoder
        encoder = VGroup()
        encoder_rect = Rectangle(width=1.5, height=3, color=BLUE, fill_opacity=0.3)
        encoder_label = Text("Encoder", font_size=20, color=BLUE)
        encoder_label.rotate(PI/2)
        encoder.add(encoder_rect, encoder_label)
        encoder.move_to(LEFT * 2.5)
        
        phi_label = MathTex(r"\boldsymbol{\phi}", font_size=24, color=BLUE)
        phi_label.next_to(encoder, DOWN, buff=0.2)
        
        # Latent parameters
        mu_box = Rectangle(width=1, height=0.8, color=GREEN, fill_opacity=0.3)
        mu_text = MathTex(r"\boldsymbol{\mu}", font_size=24, color=GREEN)
        mu_group = VGroup(mu_box, mu_text).move_to(UP * 0.8)
        
        sigma_box = Rectangle(width=1, height=0.8, color=GREEN, fill_opacity=0.3)
        sigma_text = MathTex(r"\boldsymbol{\sigma}", font_size=24, color=GREEN)
        sigma_group = VGroup(sigma_box, sigma_text).move_to(DOWN * 0.8)
        
        params = VGroup(mu_group, sigma_group)
        
        # Reparameterization
        reparam_circle = Circle(radius=0.5, color=YELLOW, fill_opacity=0.2)
        reparam_text = MathTex(r"+", font_size=32, color=YELLOW)
        reparam = VGroup(reparam_circle, reparam_text).move_to(RIGHT * 1.5)
        
        epsilon_box = Rectangle(width=1, height=0.8, color=ORANGE, fill_opacity=0.3)
        epsilon_text = MathTex(r"\boldsymbol{\epsilon}", font_size=24, color=ORANGE)
        epsilon_group = VGroup(epsilon_box, epsilon_text).move_to(RIGHT * 1.5 + DOWN * 2)
        epsilon_label = MathTex(r"\sim \mathcal{N}(\mathbf{0}, \mathbf{I})", font_size=18, color=ORANGE)
        epsilon_label.next_to(epsilon_group, DOWN, buff=0.1)
        
        # Latent z
        z_box = Rectangle(width=1, height=0.8, color=PURPLE, fill_opacity=0.3)
        z_text = MathTex(r"\mathbf{z}", font_size=24, color=PURPLE)
        z_group = VGroup(z_box, z_text).move_to(RIGHT * 3)
        
        # Decoder
        decoder = VGroup()
        decoder_rect = Rectangle(width=1.5, height=3, color=RED, fill_opacity=0.3)
        decoder_label = Text("Decoder", font_size=20, color=RED)
        decoder_label.rotate(PI/2)
        decoder.add(decoder_rect, decoder_label)
        decoder.move_to(RIGHT * 5)
        
        theta_label = MathTex(r"\boldsymbol{\theta}", font_size=24, color=RED)
        theta_label.next_to(decoder, DOWN, buff=0.2)
        
        # Output
        output_box = Rectangle(width=1.5, height=1.5, color=WHITE, fill_opacity=0.2)
        output_box.move_to(RIGHT * 7.5)
        output_label = MathTex(r"\hat{\mathbf{x}}", font_size=28)
        output_label.next_to(output_box, DOWN, buff=0.2)
        
        # Arrows
        arrow_in = Arrow(input_box.get_right(), encoder.get_left(), buff=0.1)
        arrow_enc_mu = Arrow(encoder.get_right(), mu_group.get_left(), buff=0.1)
        arrow_enc_sigma = Arrow(encoder.get_right(), sigma_group.get_left(), buff=0.1)
        arrow_mu_reparam = Arrow(mu_group.get_right(), reparam.get_left(), buff=0.1)
        arrow_sigma_reparam = Arrow(sigma_group.get_right(), reparam.get_left(), buff=0.1)
        arrow_eps_reparam = Arrow(epsilon_group.get_top(), reparam.get_bottom(), buff=0.1, color=ORANGE)
        arrow_reparam_z = Arrow(reparam.get_right(), z_group.get_left(), buff=0.1)
        arrow_z_dec = Arrow(z_group.get_right(), decoder.get_left(), buff=0.1)
        arrow_out = Arrow(decoder.get_right(), output_box.get_left(), buff=0.1)
        
        # Build scene
        all_elements = VGroup(
            input_box, input_label,
            encoder, phi_label,
            params,
            reparam, epsilon_group, epsilon_label,
            z_group,
            decoder, theta_label,
            output_box, output_label,
            arrow_in, arrow_enc_mu, arrow_enc_sigma,
            arrow_mu_reparam, arrow_sigma_reparam,
            arrow_eps_reparam, arrow_reparam_z,
            arrow_z_dec, arrow_out
        )
        all_elements.scale(0.85).center().shift(DOWN * 0.3)
        
        # Animate construction
        self.play(FadeIn(input_box), Write(input_label))
        self.play(Create(arrow_in))
        self.play(FadeIn(encoder), Write(phi_label))
        self.play(Create(arrow_enc_mu), Create(arrow_enc_sigma))
        self.play(FadeIn(params))
        self.play(
            FadeIn(reparam),
            FadeIn(epsilon_group),
            Write(epsilon_label),
            Create(arrow_mu_reparam),
            Create(arrow_sigma_reparam),
            Create(arrow_eps_reparam)
        )
        self.play(Create(arrow_reparam_z))
        self.play(FadeIn(z_group))
        self.play(Create(arrow_z_dec))
        self.play(FadeIn(decoder), Write(theta_label))
        self.play(Create(arrow_out))
        self.play(FadeIn(output_box), Write(output_label))
        
        self.wait(1)
        
        # Loss function with LaTeX underbraces
        loss = MathTex(
            r"\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{Reconstruction}} + \underbrace{\beta \cdot D_{\text{KL}}(q \| p)}_{\text{Regularization}}",
            font_size=28,
            color=YELLOW
        )
        loss.to_edge(DOWN).shift(UP * 0.2)
        
        self.play(Write(loss))
        self.wait(2)
        
        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.5)


if __name__ == "__main__":
    from manim import config
    config.preview = True
    
    # Run Monte Carlo convergence animation
    scene = MonteCarloConvergence()
    scene.render()
