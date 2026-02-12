# 🚀 rapidds

[![PyPI
version](https://img.shields.io/pypi/v/rapidds.svg)](https://pypi.org/project/rapidds/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**rapidds makes data science simpler.**

Instead of spending time figuring out *what* to analyze, `rapidds`
guides you.

Whether you're a student exploring your first dataset or a developer
prototyping quickly, `rapidds` helps you understand your data without
overthinking the process.

------------------------------------------------------------------------

## ✨ What rapidds Does

`rapidds` provides a guided dataset companion that helps you:

-   📊 Automatically analyze your data\
-   🧠 Get intelligent suggestions on what to explore next\
-   📝 Receive explanations in clear, plain language\
-   🧹 Identify missing values and mixed data types\
-   🛠 Clean datasets using guided actions

It reduces analysis paralysis and gives you direction when you're unsure
where to begin.

------------------------------------------------------------------------

## ⚡ 60-Second Example

``` python
from rapidds import Dataset

data = Dataset("students.csv")

data.analyze()
data.suggest()
data.explain()
```

No complex setup.\
No guessing what to do next.\
Just guided insights.

------------------------------------------------------------------------

## 🛠 Installation

``` bash
pip install rapidds
```

Requires: - pandas - numpy - scikit-learn

------------------------------------------------------------------------

## 🎯 Philosophy

Most data science tools assume you already know what you're looking for.

`rapidds` is built on a different idea:

> You shouldn't need to know what to analyze before you start.

It helps you discover patterns, issues, and next steps --- especially
when you're new to data science or exploring an unfamiliar dataset.

------------------------------------------------------------------------

## 📦 Current Version

`v0.1.2` --- Stable foundation release.

Includes: - Missing value detection - Mixed-type column detection -
Guided suggestions - Beginner-friendly analysis summaries - File path
and DataFrame support

------------------------------------------------------------------------

## 🛣 Roadmap

Future versions aim to include:

-   Structured return objects (beyond print statements)
-   Smarter suggestion heuristics
-   Improved type inference
-   Optional verbosity levels
-   Expanded cleaning utilities

------------------------------------------------------------------------

## 🤝 Who Is rapidds For?

-   Students learning data science
-   Developers prototyping ideas quickly
-   Anyone who wants guidance before deep analysis
