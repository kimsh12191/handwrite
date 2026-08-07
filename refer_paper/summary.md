# Handwriting Dynamics Modeling: Research Paper Summary

A comprehensive collection of relevant papers on handwriting recognition, generation, and dynamics modeling from 2018 onwards. Organized by research category with key contributions and relevance to dynamics-based handwriting modeling.

---

## 1. Online Handwriting Recognition Using Dynamics & Trajectory

### 1.1 Online Handwriting Trajectory Reconstruction from Kinematic Sensors using Temporal Convolutional Network
- **Authors:** Not specified (IJDAR/ICDAR 2023)
- **Year:** 2023
- **ArXiv:** https://arxiv.org/abs/2607.26733
- **DOI/Journal:** International Journal on Document Analysis and Recognition (IJDAR)
- **Key Contribution:** Introduces a Temporal Convolutional Network (TCN) architecture for reconstructing online handwriting trajectory from digital pen sensors (IMU). Uses Dynamic Time Warping for signal alignment and Fréchet distance for evaluation.
- **Relevance:** Directly addresses trajectory reconstruction from pen dynamics. Essential for understanding how to extract and model pen kinematics using modern neural architectures.

### 1.2 A Transformer Based Handwriting Recognition System Jointly Using Online and Offline Features
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/abs/2506.20255
- **Key Contribution:** Early fusion of offline grayscale images and online stroke (x, y, pen) sequences within a shared latent space using patch encoders and transformers.
- **Relevance:** Shows modern fusion approaches for combining dynamic trajectory data with static features. Relevant for understanding multi-modal handwriting representations.

### 1.3 Handwriting Trajectory Recovery with Diffusion Models
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/pdf/2607.03422
- **Key Contribution:** Applies diffusion models for trajectory recovery and reconstruction.
- **Relevance:** Explores generative models for trajectory synthesis and reconstruction tasks.

### 1.4 Benchmarking Online Sequence-to-Sequence and Character-based Handwriting Recognition from IMU-Enhanced Pens
- **Authors:** Not specified
- **Year:** 2022
- **ArXiv:** https://arxiv.org/pdf/2202.07036
- **Key Contribution:** Comparative study of seq2seq vs. character-based models on OnHW dataset (IMU pen dataset) with baseline comparisons.
- **Relevance:** Provides benchmarks and dataset information for handwriting recognition from pen dynamics. Establishes baselines for IMU-based approaches.

### 1.5 Mixture-of-experts for handwriting trajectory reconstruction from IMU sensors
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/html/2607.26708v1
- **Key Contribution:** Uses mixture-of-experts approach with separate expert networks for touching and hovering trajectory segments.
- **Relevance:** Advanced architecture for handling different trajectory modes. Relevant for understanding specialized modeling of pen dynamics.

### 1.6 Rotation-free Online Handwritten Character Recognition Using Linear Recurrent Units
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/html/2602.01533
- **Key Contribution:** Applies Linear Recurrent Units (state-space models) for efficient parallel training while maintaining temporal dynamics modeling.
- **Relevance:** State-space model approach to handwriting recognition. **CRITICAL for ODE-based approaches** - shows how state-space models compare with RNNs.

### 1.7 Representing Online Handwriting for Recognition in Large Vision-Language Models
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/html/2402.15307v1
- **Key Contribution:** Methods for representing online handwriting trajectories for use in large vision-language models.
- **Relevance:** Shows how to bridge trajectory representations with modern foundation models. Important for LLM integration.

### 1.8 Fast Multi-language LSTM-based Online Handwriting Recognition
- **Authors:** Not specified
- **Year:** 2019
- **ArXiv:** https://arxiv.org/abs/1902.10525
- **Key Contribution:** LSTM architecture for multi-lingual online handwriting recognition supporting 102+ languages.
- **Relevance:** Foundational LSTM work for online handwriting. Demonstrates scalability across languages using temporal neural models.

---

## 2. Handwriting Generation Models with Dynamics

### 2.1 Generating Sequences With Recurrent Neural Networks [SEMINAL WORK]
- **Authors:** Alex Graves
- **Year:** 2013 (but critically important foundation)
- **ArXiv:** https://arxiv.org/pdf/1308.0850
- **Key Contribution:** Introduces LSTM networks with mixture density networks (MDN) for text-conditioned handwriting generation. Generates realistic pen trajectories with (x, y, pen-up/down) outputs using attention mechanisms.
- **Relevance:** **FOUNDATIONAL PAPER** for dynamics-based handwriting generation. Must-read for understanding mixture density approaches to trajectory modeling and the attention mechanism for pen movement.
- **Key Insight:** Shows how to generate continuous trajectories (x, y, pen pressure) conditioned on text using MDN + attention.

### 2.2 Neural Trajectory Analysis of Recurrent Neural Network In Handwriting Synthesis
- **Authors:** Not specified
- **Year:** 2018
- **ArXiv:** https://arxiv.org/pdf/1804.04890
- **Key Contribution:** Analyzes learned representations in RNNs by extracting low-dimensional neural trajectories. Shows that writing styles are encoded in different subspaces with character-specific dynamics.
- **Relevance:** Deep analysis of how neural networks learn to represent handwriting dynamics. Essential for understanding latent dynamics in generative models.

### 2.3 WriteViT: Handwritten Text Generation with Vision Transformer
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2505.13235
- **Key Contribution:** One-shot handwritten text synthesis using Vision Transformers (ViT) to capture both fine-grained stroke details and higher-level style information.
- **Relevance:** Modern transformer-based approach to handwriting generation. Shows evolution from RNNs to attention-based architectures for dynamics modeling.

### 2.4 ScriptViT: Vision Transformer-Based Personalized Handwriting Generation
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2511.18307
- **Key Contribution:** Vision Transformer system addressing capture of writer-specific global stylistic patterns that span long-range spatial dependencies.
- **Relevance:** Addresses personalization in handwriting generation. Relevant for understanding how to model individual writing style dynamics.

### 2.5 TrInk: Ink Generation with Transformer Network
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2508.21098
- **Key Contribution:** Transformer architecture specifically designed for ink trajectory generation.
- **Relevance:** Modern approach to trajectory-based generation using transformers instead of RNNs.

### 2.6 Handwriting Transformers
- **Authors:** Ankanbhunia et al.
- **Year:** 2021
- **Conference:** ICCV 2021
- **ArXiv:** https://arxiv.org/pdf/2104.03964
- **Key Contribution:** Self-attention mechanisms for capturing local and global style patterns in few-shot handwritten text generation.
- **Relevance:** Key paper on transformer-based handwriting synthesis. Shows advantages of attention mechanisms for style modeling.

### 2.7 DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/pdf/2509.23624
- **Key Contribution:** Latent diffusion transformer with glyph and style awareness for online handwriting generation. Combines diffusion models with transformer architecture.
- **Relevance:** **State-of-the-art diffusion-based approach** for online handwriting synthesis. Important for understanding modern generative approaches to dynamics.

### 2.8 HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/html/2606.18788
- **Key Contribution:** Vector-space based scalable handwriting synthesis with language-driven control. Enables arbitrary resolution and style customization.
- **Relevance:** Demonstrates scalable vector-based representation of handwriting dynamics. **Relevant for LLM-based synthesis**.

### 2.9 One-Shot Diffusion Mimicker for Handwritten Text Generation
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/pdf/2409.04004
- **Key Contribution:** One-shot diffusion model for handwritten text generation. Reduces style conditioning to a single reference sample.
- **Relevance:** Shows how diffusion models can learn from minimal dynamics examples.

### 2.10 Disentangling Writer and Character Styles for Handwriting Generation
- **Authors:** Not specified
- **Year:** 2023
- **ArXiv:** https://arxiv.org/pdf/2303.14736
- **Key Contribution:** Separates writer style (global dynamics) from character style for improved generation control and interpretability.
- **Relevance:** Addresses fundamental question of decomposing handwriting dynamics into writer and character factors. **Important for parameterization**.

### 2.11 DeepCalliFont: Few-shot Chinese Calligraphy Font Synthesis by Integrating Dual-modality Generative Models
- **Authors:** Not specified
- **Year:** 2023
- **ArXiv:** https://arxiv.org/pdf/2312.10314
- **Key Contribution:** Dual-modality generative models for few-shot synthesis of stylized handwriting.
- **Relevance:** Demonstrates few-shot learning approaches for handwriting dynamics in non-Latin scripts.

---

## 3. Neural ODE & Differential Equation Models for Handwriting ⭐ CRITICAL FOR PROJECT

### 3.1 Modeling Trajectories with Neural Ordinary Differential Equations
- **Authors:** Not specified
- **Year:** 2021
- **Conference:** IJCAI 2021
- **Paper:** https://www.ijcai.org/proceedings/2021/0207.pdf
- **Key Contribution:** Framework for trajectory modeling using Neural ODEs. Enables continuous-time modeling of trajectories with neural network-parameterized dynamics.
- **Relevance:** **ESSENTIAL FOR ODE-BASED HANDWRITING MODELS**. Shows how to apply Neural ODEs to trajectory problems. Foundational for continuous trajectory representation.
- **Key Insight:** Demonstrates how to learn dynamics directly from trajectory data using ODE solvers.

### 3.2 Trainability, Expressivity and Interpretability in Gated Neural ODEs
- **Authors:** Not specified
- **Year:** 2023
- **ArXiv:** https://arxiv.org/pdf/2307.06398
- **Key Contribution:** Analysis of gated Neural ODE architectures for improved trainability and expressivity.
- **Relevance:** Technical analysis of Neural ODE variants for improved dynamics modeling.

### 3.3 Neural ODE Processes
- **Authors:** Not specified
- **Year:** 2021
- **ArXiv:** https://arxiv.org/pdf/2103.12413
- **Key Contribution:** Combines Neural ODEs with Gaussian process framework for uncertainty quantification in trajectory modeling.
- **Relevance:** Probabilistic approach to ODE-based trajectory modeling with uncertainty.

### 3.4 Learning Spatio-Temporal Dynamics for Trajectory Recovery via Time-Aware Transformer
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2505.13857
- **Key Contribution:** Time-aware transformer architecture for learning spatio-temporal dynamics in trajectory recovery tasks.
- **Relevance:** Modern approach combining transformers with spatio-temporal dynamics for trajectory modeling.

### 3.5 Path-minimizing Latent ODEs for improved extrapolation and inference
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/pdf/2410.08923
- **Key Contribution:** Improved ODE formulation with path-minimizing constraints for better trajectory extrapolation.
- **Relevance:** Advanced ODE technique relevant for trajectory prediction and synthesis.

### 3.6 Comprehensive Review of Neural Differential Equations for Time Series Analysis
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/html/2502.09885v1
- **Key Contribution:** Comprehensive review of Neural Differential Equations and their applications to time series.
- **Relevance:** Recent survey covering ODE approaches applicable to handwriting as time series.

### 3.7 Deep Neural Networks Inspired by Differential Equations
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/html/2510.09685v1
- **Key Contribution:** Overview of neural networks designed with differential equation principles.
- **Relevance:** Theoretical foundations for applying differential equations to neural network design.

---

## 4. Signature Verification Using Pen Dynamics

### 4.1 An Automated Method for Biometric Handwritten Signature Authentication Employing Neural Networks
- **Authors:** Not specified
- **Year:** 2021
- **Journal:** Electronics (MDPI)
- **URL:** https://www.mdpi.com/2079-9292/10/4/456
- **Key Contribution:** Neural network-based signature authentication extracting both dynamic features (velocity, pressure, pen-up/down) and static features.
- **Relevance:** Demonstrates practical application of pen dynamics for biometric authentication. Shows which dynamic features are most discriminative.

### 4.2 Auxiliary Cross-Modal Representation Learning with Triplet Loss Functions for Online Handwriting Recognition
- **Authors:** Not specified
- **Year:** 2022
- **ArXiv:** https://arxiv.org/pdf/2202.07901
- **Key Contribution:** Cross-modal learning between online trajectory and offline image representations using triplet loss.
- **Relevance:** Shows how to combine online dynamics with offline representations for improved recognition.

---

## 5. Synthetic Handwriting & GAN-based Generation

### 5.1 Adversarial Generation of Handwritten Text Images Conditioned on Sequences
- **Authors:** Not specified
- **Year:** 2019
- **ArXiv:** https://arxiv.org/pdf/1903.00277
- **Key Contribution:** GAN-based approach using BiLSTM for text-conditioned handwritten text image generation. Uses adversarial loss and CTC loss for textual control.
- **Relevance:** Key work on GAN-based handwriting synthesis. Shows how to incorporate text conditioning with dynamics.

### 5.2 Generative Adversarial Networks for Handwriting Image Generation: A Review
- **Authors:** Not specified
- **Year:** 2024
- **Journal:** The Visual Computer
- **URL:** https://dl.acm.org/doi/abs/10.1007/s00371-024-03534-9
- **Key Contribution:** Comprehensive review of GAN approaches for handwriting synthesis, covering various architectures and applications.
- **Relevance:** Recent comprehensive review of GAN-based methods in handwriting generation.

### 5.3 A Survey of Handwriting Synthesis from 2019 to 2024: A Comprehensive Review
- **Authors:** Not specified
- **Year:** 2025
- **Journal:** Pattern Recognition
- **URL:** https://www.sciencedirect.com/science/article/pii/S0031320325000172
- **Key Contribution:** State-of-the-art survey covering GAN, Transformer, VAE, and diffusion-based approaches for handwriting synthesis.
- **Relevance:** **EXCELLENT COMPREHENSIVE REFERENCE** covering all modern approaches to handwriting generation up to 2024.

---

## 6. Additional Deep Learning Approaches

### 6.1 Fully Convolutional Networks for Handwriting Recognition
- **Authors:** Not specified
- **Year:** 2019
- **ArXiv:** https://arxiv.org/pdf/1907.04888
- **Key Contribution:** FCN architecture for handwriting recognition without recurrent components.
- **Relevance:** Alternative to RNN approaches for handwriting recognition.

### 6.2 Boosting CNN-based Handwriting Recognition Systems
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/pdf/2409.05699
- **Key Contribution:** Techniques for improving CNN-based recognition systems.
- **Relevance:** Modern enhancements to convolutional approaches.

### 6.3 An Inclusive Review on Deep Learning Techniques and Their Scope in Handwriting Recognition
- **Authors:** Not specified
- **Year:** 2024
- **ArXiv:** https://arxiv.org/html/2404.08011v1
- **Key Contribution:** Comprehensive review of deep learning methods in handwriting recognition.
- **Relevance:** Recent survey covering state-of-the-art approaches and their applications.

### 6.4 Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2502.20954
- **Key Contribution:** IMU sensor-based handwriting recognition with writer-independence and efficiency improvements.
- **Relevance:** Latest work on IMU/kinematic sensor-based recognition with modern techniques.

### 6.5 Handwriting Recognition using Cohort of LSTM and Lexicon Verification with Extremely Large Lexicon
- **Authors:** Not specified
- **Year:** 2016
- **ArXiv:** https://arxiv.org/abs/1612.07528
- **Key Contribution:** Ensemble LSTM approach with lexicon verification for handwriting recognition.
- **Relevance:** Earlier LSTM-based work showing ensemble approaches.

### 6.6 Handwriting Classification using a Convolutional Recurrent Network
- **Authors:** Not specified
- **Year:** 2020
- **ArXiv:** https://arxiv.org/pdf/2008.01078
- **Key Contribution:** Hybrid CNN-RNN architecture for handwriting classification.
- **Relevance:** Shows combination of convolutional and recurrent approaches.

### 6.7 The Cursive Transformer
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2504.00051
- **Key Contribution:** Transformer-based model specifically designed for cursive handwriting.
- **Relevance:** Latest transformer work on handwriting-specific tasks.

### 6.8 Prediction of Grade, Gender, and Academic Performance of Children and Teenagers from Handwriting Using the Sigma-Lognormal Model
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/pdf/2603.11519
- **Key Contribution:** Application of sigma-lognormal model to handwriting analysis and biometric prediction.
- **Relevance:** Shows mathematical modeling of handwriting dynamics for personality/performance prediction. **Alternative mathematical model**.

### 6.9 A Benchmark of State-Space Models vs. Transformers and BiLSTM-based Models for Historical Newspaper OCR
- **Authors:** Not specified
- **Year:** 2025
- **ArXiv:** https://arxiv.org/html/2604.00725
- **Key Contribution:** Comparative study of state-space models, transformers, and BiLSTM for sequential tasks.
- **Relevance:** Benchmark comparing state-space models with other temporal architectures, relevant for trajectory modeling.

---

## Summary by Research Direction

### For Online Handwriting Recognition with Dynamics:
1. Start with: **Online Handwriting Trajectory Reconstruction from Kinematic Sensors using TCN** (2023)
2. Review: **Rotation-free Online Character Recognition Using Linear Recurrent Units** (2025)
3. Benchmark: **Benchmarking Online Sequence-to-Sequence and Character-based Handwriting** (2022)

### For Handwriting Generation with Trajectory Modeling:
1. Foundational: **Generating Sequences With Recurrent Neural Networks** (Graves, 2013)
2. Analysis: **Neural Trajectory Analysis of RNN in Handwriting Synthesis** (2018)
3. Modern: **DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer** (2024)
4. Review: **A Survey of Handwriting Synthesis from 2019 to 2024** (2025)

### For Neural ODE Approaches (⭐ PRIMARY FOCUS):
1. **Foundation:** **Modeling Trajectories with Neural Ordinary Differential Equations** (2021) - START HERE
2. **Advanced:** **Learning Spatio-Temporal Dynamics for Trajectory Recovery via Time-Aware Transformer** (2025)
3. **Theory:** **Comprehensive Review of Neural Differential Equations for Time Series** (2025)

### For Signature Verification & Biometrics:
1. Main work: **An Automated Method for Biometric Handwritten Signature Authentication** (2021)
2. Multi-modal: **Auxiliary Cross-Modal Representation Learning with Triplet Loss** (2022)

### For Synthetic Generation & GANs:
1. Key work: **Adversarial Generation of Handwritten Text Images** (2019)
2. Comprehensive: **Survey of Handwriting Synthesis 2019-2024** (2025)
3. Modern approaches: **DiffInk** and **HandwritingAgent** (2024-2025)

---

## Key Datasets & Resources Mentioned

- **OnHW Dataset:** IMU-enhanced pen dataset for online handwriting recognition
- **IAMOn-DB:** Standard benchmark for online handwriting
- **VNOn-DB:** Vietnamese online handwriting dataset
- **CharacterTrajectories:** Single-character trajectory dataset
- **IAM Dataset:** Large-scale handwriting database

---

## Recommended Reading Order for This Project

### Priority 1: CRITICAL FOR ODE-BASED APPROACH (Week 1)
1. **Modeling Trajectories with Neural ODEs** (IJCAI 2021) - Essential foundation
2. **Generating Sequences With Recurrent Neural Networks** (Graves 2013) - Understand trajectory generation paradigm
3. **Neural Trajectory Analysis** (2018) - How neural networks learn handwriting dynamics

### Priority 2: STATE-OF-THE-ART & MODERN METHODS (Week 2)
1. **A Survey of Handwriting Synthesis 2019-2024** (2025) - Comprehensive overview
2. **DiffInk: Latent Diffusion Transformer** (2024) - Modern generation approach
3. **Rotation-free Character Recognition Using LRU** (2025) - State-space model comparison
4. **Online Handwriting Trajectory Reconstruction using TCN** (2023) - Latest trajectory reconstruction

### Priority 3: IMPLEMENTATION DETAILS (Week 3)
1. **Online Handwriting Recognition with Transformers** (2025) - Multi-modal approach
2. **Disentangling Writer and Character Styles** (2023) - Parameter decomposition
3. **HandwritingAgent** (2025) - LLM-based synthesis approach

### Priority 4: SPECIALIZED TOPICS (Week 4)
1. **Comprehensive Review of Neural Differential Equations** (2025) - Deeper ODE theory
2. **Sigma-Lognormal Model** (2025) - Alternative mathematical approaches
3. **Signature Verification with Dynamics** (2021) - Biometric application

---

## Key Insights for Project Design

### ODE Model Design:
- State-space models (e.g., LRU) are competitive with RNNs - consider as alternative
- Mixture density networks (MDN) are established for continuous trajectory generation
- Gated Neural ODEs provide better expressivity
- Path-minimizing constraints can improve trajectory extrapolation

### Trajectory Representation:
- Standard: (x, y, pen-up/down) - from Graves
- Extended: (x, y, vx, vy, pressure, angle) - for fine-grained control
- Diffusion-based approaches are emerging as state-of-the-art
- Writer and character styles should be disentangled for controllability

### LLM Integration (for your auto-research loop):
- **HandwritingAgent** (2025) shows vector-space synthesis with language control
- Vision-language models can understand trajectory representations
- In-context learning for trajectory parameter inference is promising
- Multi-modal fusion (online trajectory + offline image) improves robustness

### Data Generation:
- Synthetic generation via diffusion/neural ODEs is proven effective
- GANs remain viable but transformers/diffusion are newer state-of-the-art
- Few-shot learning (1-shot examples) is achievable
- Cross-modal learning (online→offline) aids robustness

---

## Last Updated
2026-08-07

**Total Papers Found:** 40+  
**Focus Period:** 2018-2026 (with foundational 2013 work included)  
**Project Phase:** Research & Model Selection
