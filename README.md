# Real-Time Personalization with Hybrid Neural-Linear Bandits

> **A research-driven recommender system** combining deep contrastive representation learning with Bayesian bandits for adaptive, low-latency personalization.

---

## Overview

This project implements a **hybrid contextual bandit** architecture designed for **real-time personalization** in recommendation systems.  

It combines:
- **Neural representation learning (TinyMLP)** – warmed up offline with contrastive/BPR loss.
- **Bayesian linear bandit heads (NeuralLinear)** – stable Thompson sampling with adaptive regularization.

This setup addresses two critical challenges in contextual bandits:
1. **Cold-start representation problem** – MLP starts as random noise.  
2. **Numerical instability** – Posterior sampling breaks with ill-conditioned covariance matrices.  

Our solution achieves **stable learning** and **higher CTR** compared to vanilla LinUCB or cold-start NeuralLinear.

---

## Installation

git clone https://github.com/<your-username>/realtime-personalization.git
cd realtime-personalization
python -m venv .venv
source .venv/bin/activate
pip install -e .



## Results (Offline Replay)
Model	CTR (%)
LinUCB (baseline)	~5.1
NeuralLinear (cold)	~5.3
NeuralLinear (warm)	5.6+

## Research Contribution

Adaptive Posterior Sampling with Dynamic Regularization (APS-DR)
Ensures stable Thompson sampling under ill-conditioned covariance.

Warm-started NeuralLinear Bandit
MLP pre-trained with contrastive/BPR loss → faster convergence and higher CTR.

Offline Evaluation Framework
Includes Doubly Robust (DR) estimator for unbiased policy evaluation.

## Deployment

Convert offline_bandit.py → AWS Lambda for serverless inference.

Store mlp_weights.npz and bandit_heads.json in S3.

Serve recommendations through a low-latency API.

## References

Riquelme et al. (2018). Deep Bayesian Bandits Showdown

Rendle et al. (2009). Bayesian Personalized Ranking (BPR)

Agarwal et al. (2019). Doubly Robust Off-Policy Evaluation
