# Breadth Analysis Dashboard

Automated Streamlit dashboard for tracking KOSPI breadth indicators, A/D line, breadth thrust, and price-breadth divergence.

This project converts market breadth concepts into a reproducible data dashboard: daily breadth data collection, indicator calculation, visualization, and dashboard-based interpretation.

The main focus is software engineering practice: data pipeline construction, time-series indicator calculation, dashboard implementation, and reproducible market breadth analysis.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Live Demo

- **Streamlit Dashboard**: https://t8xsdcmpnny73qbgm2zyrl.streamlit.app/

---

## TL;DR

- Built a Streamlit-based breadth analysis dashboard for KOSPI market breadth data.
- Calculated and visualized advance-decline breadth, A/D line, breadth thrust, and price-breadth divergence.
- Structured daily breadth data into a reproducible CSV-based time-series dataset.
- Implemented a dashboard workflow for monitoring internal market participation beyond index-level price movement.
- Reframed a book-inspired market interpretation concept into a software dashboard and data visualization project.

---

## Project Overview

This repository provides a breadth analysis dashboard and dataset for tracking internal KOSPI market participation.

Instead of looking only at index price movement, the dashboard focuses on breadth indicators such as:

- how many components advanced or declined
- whether the A/D line confirms or diverges from index movement
- whether breadth thrust appears after weak market conditions
- whether internal participation supports or contradicts visible index trends

The project is inspired by Stan Weinstein-style breadth interpretation, but it is implemented as a data dashboard and time-series analysis workflow rather than as an investment recommendation tool.

---

## Key Features

### 1. Breadth Data Tracking

The repository stores daily breadth data with advance / decline statistics.

Core fields include:

- `date`
- `advances`
- `declines`
- `unchanged`
- `ad_diff`
- `ad_line`
- `breadth_thrust_ema10`

### 2. A/D Line Calculation

The A/D line is calculated as a cumulative advance-decline line.

```text
ad_diff = advances - declines
ad_line = cumulative sum of ad_diff
```

This helps compare index-level price movement with internal market participation.

### 3. Breadth Thrust Indicator

The dashboard includes a breadth thrust-style indicator using an EMA-based breadth signal.

This is used as a reference indicator for identifying changes in internal market strength.

### 4. Price vs Breadth Confirmation / Divergence

The dashboard is designed to support interpretation of whether market price movement is confirmed by internal breadth.

Example cases:

```text
Index rises + A/D line rises
→ broad participation confirmation

Index rises + A/D line weakens
→ possible internal divergence

Index falls + A/D line improves
→ possible early breadth recovery signal
```

These interpretations are used as analytical references only.

---

## Data

### Main Data File

```text
kospi_breadth.csv
```

### Columns

| Column | Description |
|---|---|
| `date` | Observation date |
| `advances` | Number of advancing components |
| `declines` | Number of declining components |
| `unchanged` | Number of unchanged components |
| `ad_diff` | Advances minus declines |
| `ad_line` | Cumulative advance-decline line |
| `breadth_thrust_ema10` | EMA-based breadth thrust reference signal |

---

## Dashboard Workflow

```text
Daily KOSPI Breadth Data
    ↓
Advance / Decline Calculation
    ↓
A/D Line Construction
    ↓
Breadth Thrust Indicator
    ↓
Price-Breadth Confirmation / Divergence View
    ↓
Streamlit Dashboard Visualization
```

---

## Methodological Background

This project is inspired by Stan Weinstein-style market breadth interpretation, especially the idea of comparing index movement with internal breadth indicators.

The implementation focuses on the software engineering side:

- converting domain concepts into structured data
- calculating time-series indicators
- visualizing interpretation layers through a dashboard
- maintaining a reproducible dataset for repeated analysis

The goal is not to reproduce or republish the original book content, but to implement a data-driven dashboard inspired by the general concept of breadth-based market analysis.

---

## Local Usage

If you want to inspect or extend the dataset locally:

```bash
git clone https://github.com/onekindalpha/breadth-analysis-dashboard.git
cd breadth-analysis-dashboard
```

If the Streamlit app source is included in the repository:

```bash
pip install -r requirements.txt
streamlit run app.py
```

If your dashboard file uses a different name, replace `app.py` with the actual Streamlit entrypoint.

---

## Repository Structure

```text
repository/
├── kospi_breadth.csv      # Daily KOSPI breadth data
├── README.md
└── [Streamlit dashboard files]
```

---

## Limitations

- Breadth indicators are analytical references and should not be interpreted as standalone decision rules.
- KOSPI breadth behavior can differ across market regimes and should be interpreted with macro, sector, and liquidity context.
- A/D line and breadth thrust signals may generate false positives during volatile or low-liquidity periods.
- Data quality depends on the consistency of the underlying daily breadth dataset.
- This project focuses on dashboard implementation and time-series visualization, not investment advice.

---

## Disclaimer

This project is for educational, research, and software engineering portfolio purposes only.

It is not financial advice, investment recommendation, or a trading signal service. All investment decisions are the responsibility of the user.
