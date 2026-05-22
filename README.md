# Module 4: Machine Learning

**Python for Business | FinHub | Rotman School of Management**

---

## Overview

This module introduces two machine learning techniques: K-Means clustering (unsupervised) and Random Forest classification (supervised). Both use the same scikit-learn patterns from Module 3.

## What You'll Learn

| Skill | Example |
|-------|--------|
| Discover groups in data | "These 20 stocks cluster into 2 natural groups" |
| Predict categories | "Will the stock go UP or DOWN tomorrow?" |
| Interpret feature importance | "Momentum matters more than volume" |
| Evaluate out-of-sample | Train on 4 years, test on 1 year of unseen data |

## Part 1: K-Means Clustering

**The question:** Can we find meaningful investment groups based only on beta and dividend yield?

**The reveal:** K-Means discovered Utilities vs Tech — with no labels! It found that:
- Utilities: Low beta, high dividend yield
- Tech: High beta, low dividend yield

### Key Steps
1. **Standardize features** (K-Means uses distance, so scale matters)
2. **Fit the model:** `KMeans(n_clusters=2).fit(X_scaled)`
3. **Check the clusters:** Compare to actual sector labels

## Part 2: Random Forest Classification

**The question:** Can we predict if a stock will go UP or DOWN tomorrow?

**The approach:** Instead of predicting exact returns (hard), classify direction (easier to interpret).

### Features Used
- Basic: today's return, volume change, price range, gap
- Technical: 20-day volatility, 5-day momentum
- Market: SPY return, VIX level

### The Honest Result
Training accuracy: 100% (overfitting!)
Test accuracy: ~50% (basically random guessing)

**Takeaway:** Even small edges matter in finance, but don't expect miracles from simple models.

## Files

| File | Description |
|------|-------------|
| `module4_machine_learning.ipynb` | Main notebook with clustering and classification |

## Quick Reference

```python
# K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2).fit(X_scaled)
labels = kmeans.labels_

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
importance = rf.feature_importances_
```

## Exercises

1. **Exercise 4.1:** Use the elbow method to choose K for K-Means
2. **Exercise 4.2:** Try Random Forest on a different stock (NVDA, MSFT)
3. **Exercise 4.3:** Commit your work to GitHub

---

**Next:** Module 5 — Responsible Coding with AI
