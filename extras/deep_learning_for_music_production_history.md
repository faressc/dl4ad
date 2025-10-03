# Deep Learning for Music Production: History

## Foundations for Deep Learning in Music Production

I. Feature Extraction & Music Analysis

    1987: Waibel et al. - "Phoneme Recognition Using Time-Delay Neural Networks". First neural processing of raw audio spectrograms.
    1995-1998: Kohonen Self-Organizing Maps for music (Use of audio features like MFCCs, spectral features):
        1995: Kohonen. "Self-Organizing Maps". Foundation for timbre analysis.
        1998: Toiviainen et al. "Timbre Similarity" - SOMs match human perception and neural responses.
    1999-2002: Matija Marolt's transcription breakthroughs (uses raw spectrograms):
        1999: Compares neural architectures (MLPs, RBF, SVM, TDNNs) for polyphonic piano transcription.
        2002: "Neural Networks for Note Onset Detection in Piano Music".
    2005-2006: Hinton's Deep Belief Networks (DBNs):
        "A Fast Learning Algorithm for Deep Belief Nets" - makes deep networks trainable.
        2006: "Reducing the Dimensionality of Data with Neural Networks" in Science.
        Immediately applied to music transcription and genre classification.
    2006-2010: MIR community transition - Shift from hand-crafted features (MFCCs, chroma) to learned representations. MIREX competitions drive adoption.
    2009: Lee et al. - "Unsupervised feature learning for audio classification using convolutional deep belief networks". First CNNs on audio spectrograms.
    2009: Han and Wang - "Multi-pitch detection using neural networks". Outperforms traditional DSP.
    2010: Hamel and Eck - "Learning Features from Music Audio with Deep Belief Networks". First deep learning specifically for MIR. Autoencoder like architectures for unsupervised feature learning from raw audio. While it was not cross-modal, it it laid groundwork for later work that did combine modalities (e.g. shared latent spaces for audio, MIDI, tags).

II. Symbolic Music Generation

    1951: John Cage experiments with stochastic processes (I Ching). "Music of Changes".
    1953: Iannis Xenakis uses probability theory and stochastic processes, formalized in "Formalized Music" (1963).
    1957: Hiller and Isaacson's "Illiac Suite" - First computer-composed score using Markov Chains.
    1981: David Cope starts EMI project - rule-based systems for style imitation.
    1989: Todd - "A Connectionist Approach to Algorithmic Composition". First RNN for music composition.
    1994: Mozer's CONCERT system - "Neural Network Music Composition by Prediction". First successful neural melodic generation.
    2002: Eck and Schmidhuber - "Finding Temporal Structure in Music: Blues Improvisation with LSTM". First neural network to learn entire musical form (12-bar blues).
    2011: Boulanger-Lewandowski, Bengio, Vincent - "Modeling Temporal Dependencies in High-Dimensional Sequences" (RNN-RBM). Scaled polyphonic generation with deep generative models.

III. Differentiable Audio Processing & Effects

    1960: Widrow and Hoff - LMS algorithm. "Adaptive Switching Circuits". Foundation for gradient-based audio processing, later used in echo cancellation, noise reduction.
    1989: Shynk and Moorer - "A Gradient-Based Approach to IIR Filter Design for Musical Applications". First gradient descent for musical DSP.
    1997: Zhang and Duhamel - "Neural Networks for Musical Effects Processing". First attempt at modeling analog effects with MLPs.
    2007 – David Yeh et al. (CCRMA, Stanford): “Automated physical modeling of nonlinear audio circuits for real-time audio effects—Part I: Theoretical development” (IEEE TCAS). MLPs for Guitar Amp Modeling. Bridged physical modeling (SPICE-like) and data-driven approaches.
    2009: Paksoy & Gunel - "Neural Networks for Emulation of Tube Amplifiers". Demonstrated small feedforward networks could learn transfer functions of guitar preamps and tubes, paving way for real-time applications.

IV. The Convergence (2012)

    2012: AlexNet (Krizhevsky, Sutskever, Hinton) wins ImageNet. While not music-specific, this breakthrough:
        Catalyzed treating audio spectrograms as images
        Enabled CNN revolution in music
        Unified feature extraction, generation, and processing approaches
        Marked the beginning of modern deep learning era in music production

## Milestones in Deep Learning for Music Production

Universal:

    2014: Kingma and Welling - Variational Autoencoders (VAEs). Enables controlled generation with latent space manipulation.
    2014: Goodfellow et al. - Generative Adversarial Networks (GANs). While not music-specific, becomes crucial for audio synthesis later.
    2016: Oord et al. - "WaveNet: A Generative Model for Raw Audio". First successful raw audio generative model, high-quality speech and music synthesis.
    2017: Attention is All You Need - Vaswani et al. Foundation for transformer models, later adapted for music (e.g. Music Transformer).
    2020: Denoising Diffusion Probabilistic Models (DDPMs) - Sohl-Dickstein et al. and Ho et al. Foundation for diffusion models, later applied to music generation.
    2021: CLIP "Contrastive Language-Image Pre-Training" (2021). Foundation for cross-modal models, later adapted for audio (e.g. CLAP).

I: Feature Extraction & Music Analysis

    2013: Sigtia and Dixon - "Improved Music Feature Learning with Deep Neural Networks". CNNs outperform hand-crafted features on all MIREX tasks.
    2014: Dieleman & Schrauwen – “End-to-end learning for music audio” (raw waveform CNN for music tagging). Often overlooked precursor to WaveNet and raw audio approaches.
    2015: Roberts, Engel, Raffel - "Hierarchical Variational Autoencoders for Music". First VAE specifically for music, learns hierarchical structure.
    2018: CREPE: "A Convolutional Representation for Pitch Estimation". State-of-the-art pitch tracking with CNNs on raw audio.
    2019: Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms. New evaluation metric for generative models, analogous to FID in images. Uses Neural Network computed embeddings to assess quality.
    2021: BeatNet by M.J. Hydri: "CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking" First system to jointly track beat, downbeat, tempo, AND meter in real-time
    2022: PASST: Efficient Training of Audio Transformers with Patchout (2022). Transformer models for audio tasks, outperforming CNNs in many scenarios.
    2022: CLAP: "CLAP: Learning Audio Concepts from Natural Language Supervision" (2022). Foundation model for audio, enabling zero-shot tasks.

II. Neural Audio Synthesis

    2017: Engel et al. - "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders". First to use WaveNet for musical timbre synthesis.
    2018: Hawthorne et al. - "Adversarial Audio Synthesis". First GAN for raw audio synthesis, high-quality piano notes.
    2019: DDSP: "Differentiable Digital Signal Processing". Combines DSP and deep learning for interpretable synthesis.
    2019: Universal audio synthesizer control with normalizing flows (2019). Flow-based models for controllable synthesis.
    2019: Musenet demonstrated transformer architecture effectiveness for music by generating 4-minute compositions SoundCloud
    2020: Jukebox: "Jukebox: A Generative Model for Music". High-fidelity music generation with lyrics using VQ-VAE and transformers.
    2021: Rave: "A variational Autoencoder for fast and high-quality neural audio synthesis". Real-time synthesis with VAE architecture.
    2022: SoundStream: An End-to-End Neural Audio Codec
    2022: AudioLM (Google) – large language model for audio tokens, including music. Precursor to MusicLM.
    2022: MuLan: A Joint Embedding of Music Audio and Natural Language
    2023: GanStrument: "Adversarial Instrument Sound Synthesis with Pitch-invariant Instance Conditioning".
    2023: MusicLM:  Generating Music From Text (2023). Text-to-music generation with high fidelity and adherence to prompts.
    2023: MusicGen (Meta) - Text-to-music model with open weights, enabling wider access to music generation technology.
    2023: MusicGen (Meta) - You have MusicLM but MusicGen was equally important
    2023: Suno AI.
    2024: Udio
    2024: Music2Latent: Consistency Autoencoders for Latent Audio Compression (2024).
    2024: Stable Audio Open
    2025: Adding Temporal Musical Controls on Top of Pretrained Generative Models (2025).
    2025: Live Music Models: Lyria 2
    2025: SpectroStream: A Versatile Neural Codec for General Audio

    Challenges: High computational cost, long-range structure, and controllability.

III. Audio Processing & Effects

    2014: LANDR: AI-driven mastering service using machine learning on large datasets.
    2016: iZotope Neutronintroduced the first industry AI mixing assistant with Track Assistant that automatically detects instruments and recommends processing chains.
    2016: iZotope Ozone - AI-assisted mastering with Master Assistant.
    2018: Ramirez and Reiss - "End-to-End Equalization With Convolutional Neural Networks". CNNs for EQ matching.
    2019: Spleeter and Demucs - Deep learning for source separation. Real-time, high-quality stems from mixed audio.
    2020: Real-Time Guitar Amplifier Emulation with Deep Learning (Alec Wright et al., 2020). Real-time amp modeling with low-latency neural networks.
    2020: Automatic multitrack mixing with a differentiable mixing console of neural audio effects (2020). End-to-end mixing with neural effects.
    2021: HT-Demucs - "Hybrid Transformer Demucs" (2021). State-of-the-art source separation with hybrid CNN-transformer architecture.
    2021: Steerable discovery of neural audio effects (2021). Steinmetz et al. - Creating neural audio effects with unseen new user controls and timbres.
    2022: Neural DSP Quad Cortex - First hardware unit with neural amp modeling
    2022: Deepfilternet: "Deepfilternet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based On Deep Filtering" (2022). Real-time speech enhancement with low-latency deep adaptive spectral filtering.
    2023: Pruning Deep Neural Network Models of Guitar Distortion Effects (2023). Techniques for reducing model size while maintaining audio quality.
    2023: Logic Pro AI Session Players - AI musicians in a major DAW.
    2024: Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing (2024). RNNs that can operate at different sample rates without retraining.
    
    Challenges: Modeling long range dependencies, real-time constraints, interpretability, and user control.

IV. Advances in Real-Time & Edge Deployment

    2019: TFLite and ONNX - Frameworks for deploying models on mobile and embedded devices.
    2020: Streaming keyword spotting on mobile devices.
    2021: Apple Core ML updates - Enhanced support for audio models on iOS devices.
    2021: RT-Neural: "FAST NEURAL INFERENCING FOR REAL-TIME SYSTEMS". C++ library for low-latency neural network inference in audio applications.
    2022: Streamable Neural Audio Synthesis with Non-Causal Convolutions Caillon, Esling
    2025: S-Edge: Efficient and interpretable raw audio classification with diagonal state space models (2025). Can change sample rate at inference time.

V. Symbolic/MIDI Generation:

    2016: DeepBach - Bach chorales in the style of Bach
    2018: Music Transformer (Huang et al.) - Long-term structure in symbolic music
    2019: MuseNet (OpenAI) - Multi-instrument, multi-style symbolic generation
    2019: Pop Music Transformer - Pop piano generation with relative attention
    2023: Anticipatory Music Transformer - Real-time accompaniment

Our latest Research Contributions:
    2024: Anira: "An Architecture for Real-Time Neural Audio Effects" (2024).
    2025: PGESAM: "Pitch-Conditioned Instrument Sound Synthesis from an Interactive Timbre Latent Space" (2025).
    2025: Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations (2025).
