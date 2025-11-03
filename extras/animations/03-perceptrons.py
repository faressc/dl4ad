from manim import *
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
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

class XORTanhTransformation(ThreeDScene):
    def construct(self):
        # Part 1: Show XOR problem in original space (2D view)
        self.show_original_xor()
        self.wait(2)
        
        # Part 2: Show 3D warping visualization
        self.show_3d_warping()
        self.wait(3)
        
        # Part 3: Show single layer transformation result in 2D
        self.show_hidden_space_xor()
        self.wait(3)

    def neural_network_computation(self, xor_inputs, xor_outputs):
        # Learn a neural network that solves XOR
        # Weights and biases for 2 hidden neurons
        np.random.seed(42)
        # Better initialization: Xavier/Glorot initialization
        W_1 = np.random.randn(2, 2) * 0.5
        b_1 = np.zeros(2)
        W_2 = np.random.randn(1, 2) * 0.5
        b_2 = np.zeros(1)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        def model_forward(X):
            hidden = np.tanh(X @ W_1.T + b_1)
            output = sigmoid(hidden @ W_2.T + b_2)
            return hidden, output
        
        def loss_fn(y_true, y_pred):
            # Binary cross-entropy loss
            eps = 1e-7
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )

        learning_rate = 0.5

        for i in range(20000):
            # Forward pass
            hidden, outputs = model_forward(xor_inputs)
            loss = loss_fn(xor_outputs, outputs.flatten())

            # Backpropagation
            # For sigmoid + binary cross-entropy, the gradient simplifies to:
            output_error = outputs - xor_outputs.reshape(-1, 1)

            dW2 = (output_error.T @ hidden) / xor_inputs.shape[0]
            db2 = np.mean(output_error, axis=0)

            # Backpropagate to hidden layer
            # hidden is already tanh(z), so derivative is 1 - hidden^2
            hidden_error = output_error @ W_2 * (1 - hidden**2)
            dW1 = (hidden_error.T @ xor_inputs) / xor_inputs.shape[0]
            db1 = np.mean(hidden_error, axis=0)

            # Update weights and biases
            W_2 -= learning_rate * dW2
            b_2 -= learning_rate * db2
            W_1 -= learning_rate * dW1
            b_1 -= learning_rate * db1

            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}, Max grad: {np.max(np.abs(dW2)):.4f}")
                print(f"  Outputs: {outputs.flatten()}")
                print(f"  Targets: {xor_outputs}")
                print(f"  Error: {output_error.flatten()}")

        print("Trained Weights and Biases:")
        print("W1:", W_1)
        print("b1:", b_1)
        print("W2:", W_2)
        print("b2:", b_2)
        print("Final Loss:", loss)

        return W_1, b_1, W_2, b_2

    def show_original_xor(self):
        # Switch to 2D view temporarily
        self.set_camera_orientation(phi=0 * DEGREES, theta=0 * DEGREES)
        
        subtitle = Text("Original Space: Not Linearly Separable", font_size=28)
        subtitle.to_edge(UP)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(Write(subtitle))
        
        # Create axes
        axes = Axes(
            x_range=[-0.5, 1.5, 0.5],
            y_range=[-0.5, 1.5, 0.5],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE},
        )
        
        labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        self.play(Create(axes), Write(labels))
        
        # XOR data points
        xor_data = [
            (0, 0, RED),    # Class 0
            (0, 1, GREEN),  # Class 1
            (1, 0, GREEN),  # Class 1
            (1, 1, RED),    # Class 0
        ]
        
        points = VGroup()
        for x, y, color in xor_data:
            point = Dot(axes.c2p(x, y), color=color, radius=0.15)
            label = Text(f"({x},{y})", font_size=20).next_to(point, UP, buff=0.1)
            points.add(point, label)
        
        self.play(FadeIn(points))
        
        # Try to draw a separating line (and fail)
        line1 = Line(axes.c2p(-0.5, 0.5), axes.c2p(1.5, 0.5), color=YELLOW)
        cross1 = Cross(line1)
        
        self.play(Create(line1))
        self.wait(0.5)
        self.play(Create(cross1))
        self.wait(0.5)
        self.play(FadeOut(line1), FadeOut(cross1))
        
        # Store for later
        self.original_axes = axes
        self.original_labels = labels
        self.original_points = points
        self.subtitle = subtitle
        
        # Fade out for 3D transition
        self.play(
            FadeOut(axes), FadeOut(labels), 
            FadeOut(points), FadeOut(subtitle)
        )
        self.wait(0.5)

    def show_3d_warping(self):
        # Update subtitle
        warping_subtitle = Text("3D View: Full Neural Network Transformation", font_size=28)
        warping_subtitle.to_edge(UP)
        self.add_fixed_in_frame_mobjects(warping_subtitle)
        self.play(Write(warping_subtitle))
        
        # Set 3D camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D axes - now z-axis represents h2 (second hidden neuron)
        axes_3d = ThreeDAxes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[-1, 1, 0.5],
            x_length=5,
            y_length=5,
            z_length=2.5,
        )
        
        axes_labels = axes_3d.get_axis_labels(
            x_label="x_1 (norm)", y_label="x_2 (norm)", z_label="h_1, h_2"
        )
        
        self.play(Create(axes_3d), Write(axes_labels))
        self.wait()
        
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_outputs = np.array([0, 1, 1, 0])

        # Train neural network to get weights and biases
        W_1, b_1, W_2, b_2 = self.neural_network_computation(xor_inputs, xor_outputs)

        # Create a warped grid showing the full transformation
        # We'll show how the x1-x2 plane gets warped into h1-h2 space
        grid_lines = VGroup()
        
        # Create grid in original space and warp it to hidden space
        for i in np.linspace(0, 1, 10):
            # Vertical lines (constant x1)
            points_v_h1 = []
            points_v_h2 = []
            for j in np.linspace(0, 1, 20):
                inp = np.array([i, j])
                hidden = np.tanh(W_1 @ inp + b_1)
                # Map to visualization space: (x1, x2, h1)
                points_v_h1.append(axes_3d.c2p(i * 2 - 1, j * 2 - 1, hidden[0]))
                points_v_h2.append(axes_3d.c2p(i * 2 - 1, j * 2 - 1, hidden[1]))

            line_v_h1 = OpenGLVMobject()
            line_v_h1.set_points_as_corners(points_v_h1)
            line_v_h1.set_stroke(color=BLUE, width=1, opacity=0.4)
            grid_lines.add(line_v_h1)

            line_v_h2 = OpenGLVMobject()
            line_v_h2.set_points_as_corners(points_v_h2)
            line_v_h2.set_stroke(color=GREEN, width=1, opacity=0.4)
            grid_lines.add(line_v_h2)

            # Horizontal lines (constant x2)
            points_h1 = []
            points_h2 = []
            for j in np.linspace(0, 1, 20):
                inp = np.array([j, i])
                hidden = np.tanh(W_1 @ inp + b_1)
                points_h1.append(axes_3d.c2p(j * 2 - 1, i * 2 - 1, hidden[0]))
                points_h2.append(axes_3d.c2p(j * 2 - 1, i * 2 - 1, hidden[1]))

            line_h1 = OpenGLVMobject()
            line_h1.set_points_as_corners(points_h1)
            line_h1.set_stroke(color=PURPLE, width=1, opacity=0.4)
            grid_lines.add(line_h1)

            line_h2 = OpenGLVMobject()
            line_h2.set_points_as_corners(points_h2)
            line_h2.set_stroke(color=ORANGE, width=1, opacity=0.4)
            grid_lines.add(line_h2)

        formula_text = Text(
            "Warped grid: Input space → Hidden layer activations",
            font_size=24
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(formula_text)
        
        self.play(Create(grid_lines), Write(formula_text))
        self.wait()

        # Add XOR points in the transformed 3D space
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_colors = [RED, GREEN, GREEN, RED]
        
        points_3d = VGroup()
        labels_3d = VGroup()
        
        for inp, color in zip(xor_inputs, xor_colors):
            # Transform through neural network
            hidden = np.tanh(W_1 @ inp + b_1)
            h1, h2 = hidden[0], hidden[1]
            
            # Position in 3D: (x1, x2, h2)
            x_normalized = inp * 2 - 1
            
            # Point in transformed space
            point_3d_h1 = Dot(
                point=axes_3d.c2p(x_normalized[0], x_normalized[1], h1),
                color=color,
                radius=0.15
            )
            points_3d.add(point_3d_h1)

            # Point in transformed space
            point_3d_h2 = Dot(
                point=axes_3d.c2p(x_normalized[0], x_normalized[1], h2),
                color=color,
                radius=0.15
            )
            points_3d.add(point_3d_h2)

            # Label showing transformation
            label_h1 = MathTex(
                f"({inp[0]},{inp[1]}) \\to h_1 = ({h1:.2f})",
                font_size=20,
                color=color
            )

            label_h2 = MathTex(
                f"({inp[0]},{inp[1]}) \\to h_2 = ({h2:.2f})",
                font_size=20,
                color=color
            )
            labels_3d.add(label_h1)
            labels_3d.add(label_h2)
        
        # Position labels on the side
        labels_3d.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        labels_3d.to_corner(UL).shift(DOWN * 1.5 + RIGHT * 0.2)
        self.add_fixed_in_frame_mobjects(labels_3d)


        for point, label in zip(points_3d, labels_3d):
            self.play(FadeIn(point), Write(label))
            self.wait(0.1)
        
        # Rotate to show the warping from different angles
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(18)
        self.stop_ambient_camera_rotation()
        self.wait()

        self.play(FadeOut(points_3d), FadeOut(labels_3d), FadeOut(formula_text), FadeOut(grid_lines))
        self.wait(0.5)

        # Create a warped grid showing the full transformation
        # We'll show how the x1-x2 plane gets warped into h1-h2 space
        grid_lines = VGroup()
        
        # Create grid in original space and warp it to hidden space
        for i in np.linspace(0, 1, 10):
            # Vertical lines (constant x1)
            points_v_h1 = []
            points_v_h2 = []
            for j in np.linspace(0, 1, 20):
                inp = np.array([i, j])
                hidden = W_1 @ inp + b_1
                # Map to visualization space: (x1, x2, h1)
                points_v_h1.append(axes_3d.c2p(i * 2 - 1, j * 2 - 1, hidden[0]))
                points_v_h2.append(axes_3d.c2p(i * 2 - 1, j * 2 - 1, hidden[1]))

            line_v_h1 = OpenGLVMobject()
            line_v_h1.set_points_as_corners(points_v_h1)
            line_v_h1.set_stroke(color=BLUE, width=1, opacity=0.4)
            grid_lines.add(line_v_h1)

            line_v_h2 = OpenGLVMobject()
            line_v_h2.set_points_as_corners(points_v_h2)
            line_v_h2.set_stroke(color=GREEN, width=1, opacity=0.4)
            grid_lines.add(line_v_h2)

            # Horizontal lines (constant x2)
            points_h1 = []
            points_h2 = []
            for j in np.linspace(0, 1, 20):
                inp = np.array([j, i])
                hidden = W_1 @ inp + b_1
                points_h1.append(axes_3d.c2p(j * 2 - 1, i * 2 - 1, hidden[0]))
                points_h2.append(axes_3d.c2p(j * 2 - 1, i * 2 - 1, hidden[1]))

            line_h1 = OpenGLVMobject()
            line_h1.set_points_as_corners(points_h1)
            line_h1.set_stroke(color=PURPLE, width=1, opacity=0.4)
            grid_lines.add(line_h1)

            line_h2 = OpenGLVMobject()
            line_h2.set_points_as_corners(points_h2)
            line_h2.set_stroke(color=ORANGE, width=1, opacity=0.4)
            grid_lines.add(line_h2)

        formula_text = Text(
            "Warped grid: Input space → Hidden layer with no activation",
            font_size=24
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(formula_text)
        
        self.play(Create(grid_lines), Write(formula_text))
        self.wait()

        # Add XOR points in the transformed 3D space
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_colors = [RED, GREEN, GREEN, RED]
        
        points_3d = VGroup()
        labels_3d = VGroup()
        
        for inp, color in zip(xor_inputs, xor_colors):
            # Transform through neural network
            hidden = W_1 @ inp + b_1
            h1, h2 = hidden[0], hidden[1]
            
            # Position in 3D: (x1, x2, h2)
            x_normalized = inp * 2 - 1
            
            # Point in transformed space
            point_3d_h1 = Dot(
                point=axes_3d.c2p(x_normalized[0], x_normalized[1], h1),
                color=color,
                radius=0.15
            )
            points_3d.add(point_3d_h1)

            # Point in transformed space
            point_3d_h2 = Dot(
                point=axes_3d.c2p(x_normalized[0], x_normalized[1], h2),
                color=color,
                radius=0.15
            )
            points_3d.add(point_3d_h2)

            # Label showing transformation
            label_h1 = MathTex(
                f"({inp[0]},{inp[1]}) \\to h_1 = ({h1:.2f})",
                font_size=20,
                color=color
            )

            label_h2 = MathTex(
                f"({inp[0]},{inp[1]}) \\to h_2 = ({h2:.2f})",
                font_size=20,
                color=color
            )
            labels_3d.add(label_h1)
            labels_3d.add(label_h2)
        
        # Position labels on the side
        labels_3d.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        labels_3d.to_corner(UL).shift(DOWN * 1.5 + RIGHT * 0.2)
        self.add_fixed_in_frame_mobjects(labels_3d)


        for point, label in zip(points_3d, labels_3d):
            self.play(FadeIn(point), Write(label))
            self.wait(0.1)
        
        # Rotate to show the warping from different angles
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(20)
        self.stop_ambient_camera_rotation()
        self.wait()
        
        # Clean up for next scene
        self.play(
            FadeOut(axes_3d), FadeOut(axes_labels),
            FadeOut(grid_lines), FadeOut(points_3d),
            FadeOut(warping_subtitle), FadeOut(labels_3d), FadeOut(formula_text)
        )
        self.wait(0.5)

    def show_hidden_space_xor(self):
        # Switch to 2D view temporarily
        self.set_camera_orientation(phi=0 * DEGREES, theta=0 * DEGREES)

        subtitle = Text("Hidden Layer Space: Now Linearly Separable", font_size=28)
        subtitle.to_edge(UP)
        self.add_fixed_in_frame_mobjects(subtitle)
        self.play(Write(subtitle))

        # Create axes for hidden layer space
        axes_hidden = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE},
        )

        labels = axes_hidden.get_axis_labels(x_label="h_1", y_label="h_2")
        self.play(Create(axes_hidden), Write(labels))

        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_outputs = np.array([0, 1, 1, 0])

        # Train neural network to get weights and biases
        W_1, b_1, W_2, b_2 = self.neural_network_computation(xor_inputs, xor_outputs)

        # Add XOR points in the hidden layer space
        xor_colors = [RED, GREEN, GREEN, RED]
        points_hidden = VGroup()

        for inp, color in zip(xor_inputs, xor_colors):
            # Transform through neural network
            hidden = np.tanh(W_1 @ inp + b_1)
            h1, h2 = hidden[0], hidden[1]

            point_hidden = Dot(
                axes_hidden.c2p(h1, h2),
                color=color,
                radius=0.15
            )
            points_hidden.add(point_hidden)

        self.play(FadeIn(points_hidden))
        self.wait()

        # Draw separating line from the trained network
        # The decision boundary in hidden space is: W_2 @ h + b_2 = 0
        # Solving for h2: w2_1 * h1 + w2_2 * h2 + b2 = 0
        # h2 = -(w2_1 * h1 + b2) / w2_2
        
        w2_1, w2_2 = W_2[0, 0], W_2[0, 1]
        b2_val = b_2[0]
        
        # Calculate two points on the decision boundary
        h1_range = np.array([-1.5, 1.5])
        h2_boundary = -(w2_1 * h1_range + b2_val) / w2_2
        
        # Create the decision boundary line
        decision_line = Line(
            axes_hidden.c2p(h1_range[0], h2_boundary[0]),
            axes_hidden.c2p(h1_range[1], h2_boundary[1]),
            color=YELLOW,
            stroke_width=4
        )
        
        boundary_label = Text(
            f"Decision boundary: {w2_1:.2f}h₁ + {w2_2:.2f}h₂ + {b2_val:.2f} = 0",
            font_size=20,
            color=YELLOW
        ).next_to(decision_line, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(boundary_label)
        
        self.play(Create(decision_line), Write(boundary_label))
        self.wait()
        
        # Add checkmark to indicate success
        checkmark = Text("✓", color=GREEN, font_size=48).next_to(axes_hidden, RIGHT)
        self.add_fixed_in_frame_mobjects(checkmark)
        self.play(FadeIn(checkmark))
        self.wait(2)

        self.play(
            FadeOut(points_hidden), FadeOut(decision_line), FadeOut(boundary_label), FadeOut(checkmark)
        )

        no_activation_label = Text(
            "Without Activation Function",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(no_activation_label)
        self.play(Write(no_activation_label))

        # Zoom out to show the effect of no activation function
        self.play(axes_hidden.animate.scale(0.2), FadeOut(labels))

        points_hidden_no_activation = VGroup()

        for inp, color in zip(xor_inputs, xor_colors):
            # Transform through neural network
            hidden = W_1 @ inp + b_1
            h1, h2 = hidden[0], hidden[1]

            point_hidden = Dot(
                axes_hidden.c2p(h1, h2),
                color=color,
                radius=0.15
            )
            points_hidden_no_activation.add(point_hidden)

        self.play(FadeIn(points_hidden_no_activation))
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(axes_hidden), FadeOut(points_hidden_no_activation), FadeOut(subtitle), FadeOut(no_activation_label)
        )
        self.wait(0.5)

if __name__ == "__main__":
    XORTanhTransformation().neural_network_computation(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([0, 1, 1, 0])
    )