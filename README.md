# 🛍️ E-Commerce A/B Test Causal Impact Incrementality of Recommendations

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📌 Executive Summary
Causal evaluation of e-commerce recommendation systems. Measuring true incremental revenue (iRPU) vs. cannibalization using A/B testing, CUPED variance reduction, and counterfactual modeling. 

This project explores the **true incremental impact (iRPU)** of a new Fashion Recommendation System. By building a custom Causal Simulation Engine and applying advanced statistical methods (CUPED, Bootstrap CI, Mediation Analysis), this research demonstrates how to isolate genuine revenue growth from demand redistribution and infrastructure latency.

## 🎯 The Business Problem
A new ranking policy in the recommendation block showed significant increases in engagement. However, the business needed to answer:
1. **Incrementality:** Is the system generating *new* money, or simply cannibalizing Organic Search?
2. **Infrastructure Tax:** How much does the algorithm's inference latency cost us in lost conversions?
3. **Supply Chain:** Does the new policy lead to out-of-stock cancellations?

## 🧪 Methodology & Experiment Design
To establish a verifiable "Ground Truth", I developed a Python-based **Causal Data Generator** simulating 150,000 users. This allowed me to benchmark standard A/B test estimations against actual counterfactual data.

* **Split:** Randomized 50/50 A/B Split (User-level).
* **Metrics:** ARPU (Average Revenue Per User), iRPU (Incremental RPU), Session Depth, Latency-adjusted CTR.
* **Techniques Used:** Welch's T-test, Bootstrap Confidence Intervals, CUPED (Variance Reduction), Causal Deep Dives.

