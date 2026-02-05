# rapidds — What This Library Intentionally Does NOT Do

This document clarifies the boundaries of rapidds.

rapidds is designed to be a **thinking companion and guardrail**, not an automated decision-maker.
The exclusions below are intentional and enforced by design.

---

## rapidds does NOT train models

rapidds will never automatically train machine learning models.

**Why:**
- Model choice depends on domain context
- Automatic training risks data leakage
- Evaluation metrics vary by problem

**What rapidds does instead:**
- Detects modeling risks (imbalance, leakage signals)
- Suggests preparation steps
- Hands off cleanly to sklearn or other frameworks

---

## rapidds does NOT automatically encode or scale features

rapidds will not silently:
- encode categorical variables
- scale numeric features
- engineer features

**Why:**
- These steps are context-dependent
- Silent transformations reduce trust
- Analysts must remain in control

**What rapidds does instead:**
- Flags unencoded categoricals
- Highlights missing-value risks
- Provides explicit preparation helpers

---

## rapidds does NOT replace pandas or NumPy

rapidds is not a replacement for foundational data libraries.

**Why:**
- pandas and NumPy are optimized, battle-tested engines
- Replacing them would increase risk and reduce adoption

**What rapidds does instead:**
- Builds on top of pandas
- Uses NumPy under the hood
- Focuses on reasoning, not computation

---

## rapidds does NOT make irreversible decisions

rapidds will never:
- drop columns without instruction
- mutate data on inspection
- apply transformations implicitly

**Why:**
- Silent changes break reproducibility
- Analysts must approve every action

---

## Design principle

> rapidds accelerates judgment — it does not replace it.

If a decision is controversial among experienced analysts,
rapidds will **suggest**, not **execute**.

---

## Why this matters

Clear boundaries are what make rapidds trustworthy in real-world environments.
This document exists to prevent scope creep and preserve long-term credibility.
