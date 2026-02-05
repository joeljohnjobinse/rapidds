# rapidds – Design Manifesto

rapidds exists to accelerate data understanding, not replace analyst judgment.

## Core Principles

### 1. Explicit over implicit
rapidds never makes irreversible decisions without user consent.

### 2. Detect → Suggest → Execute
- Detect facts
- Suggest actions
- Execute only when explicitly instructed

### 3. Conservative defaults
Defaults are safe, reversible, and transparent.

### 4. No silent data leakage
rapidds avoids automatic modeling or feature engineering that could introduce leakage.

### 5. Composable by design
rapidds integrates with pandas, sklearn, and existing workflows.

## What rapidds is NOT
- An AutoML tool
- A replacement for pandas or sklearn
- A black-box decision system
