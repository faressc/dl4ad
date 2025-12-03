from manim import *
import numpy as np
import matplotlib.pyplot as plt


def create_positional_encoding_plot(max_seq_len=100, d_model=512, save_path=None):
    """
    Create and display a plot of the vanilla (sinusoidal) positional encoding matrix.
    
    Args:
        max_seq_len: Maximum sequence length (number of positions)
        d_model: Dimension of the model (embedding dimension)
    """
    # Initialize positional encoding matrix
    pe = np.zeros((max_seq_len, d_model))
    
    # Create position indices
    position = np.arange(0, max_seq_len).reshape(-1, 1)
    
    # Create dimension indices
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Full positional encoding matrix (transposed)
    im1 = ax1.imshow(pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Embedding Dimension', fontsize=12)
    ax1.set_title('Positional Encoding Matrix (Sinusoidal)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Value')
    
    # Plot 2: Specific dimensions to show the pattern
    ax2.plot(pe[:, 0], label='Dimension 0', alpha=0.7)
    ax2.plot(pe[:, 1], label='Dimension 1', alpha=0.7)
    ax2.plot(pe[:, 50], label='Dimension 50', alpha=0.7)
    ax2.plot(pe[:, 100], label='Dimension 100', alpha=0.7)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Encoding Value', fontsize=12)
    ax2.set_title('Positional Encodings for Selected Dimensions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    return pe

if __name__ == "__main__":
    # Create and display the positional encoding plot
    print("Generating positional encoding visualization...")
    pe_matrix = create_positional_encoding_plot(max_seq_len=100, d_model=128, save_path="./slides/assets/images/06-attention_transformer/positional_encoding.png")
    print(f"Positional encoding matrix shape: {pe_matrix.shape}")
    print("Plot saved as 'positional_encoding.png'")

