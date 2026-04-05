# SR 11-7 Compliant VaR Model Validation Engine

A quantitative model validation project built against the Federal Reserve's SR 11-7 supervisory guidance. Implements and stress-tests three Value-at-Risk methodologies — Historical Simulation, Parametric, and EWMA — on a multi-asset portfolio covering 17 years of market data (2007–2025), including two major crisis regimes.

## Portfolio

| Asset | Ticker | Role |
|-------|--------|------|
| SPDR S&P 500 ETF | SPY | Equity beta |
| iShares 20+ Year Treasury | TLT | Duration / flight-to-quality |
| iShares iBoxx High Yield | HYG | Credit spread risk |
| iShares iBoxx Investment Grade | LQD | Investment-grade credit |
| SPDR Gold Shares | GLD | Inflation / tail hedge |

**Dataset:** 4,458 trading days · April 2007 – December 2025 · Equal-weighted daily returns


## Models Implemented

### 1. Historical Simulation VaR
Rolling 252-day empirical 1st percentile. No distributional assumption.

### 2. Parametric VaR
Rolling 252-day mean and standard deviation with a fixed z-score multiplier.  
Two calibrations tested: Normal (−2.33σ) and Student's t (−2.81σ).

### 3. EWMA VaR
Exponentially weighted volatility with α = 0.06 (λ = 0.94, consistent with RiskMetrics). 

**Note**: For all three models, the VaR threshold at *t* time is set using only the information set available at time *t - 1*

## Key Findings

### Distributional Assumption Testing

The normality assumption underlying both Parametric and EWMA VaR was tested via Jarque-Bera on rolling 252-day windows.

| Metric | Value |
|--------|-------|
| Excess kurtosis (full sample) | 12.03 |
| Skewness | −0.004 |
| Jarque-Bera statistic | 26,861.78 |
| Jarque-Bera p-value | 0.0 |
| Parametric VaR — normality rejection rate | **76.3%** of windows |
| EWMA residuals — normality rejection rate | **67.7%** of windows |

EWMA's dynamic volatility absorption reduces excess kurtosis from 12.03 in raw returns to 2.04 in standardized residuals, indicating a partial improvement, but far from resolving the fat-tail problem.

Student's t was fitted on standardized residuals with: **dof = 9.11 → 1% threshold = −2.8144σ** (vs −2.33σ under normality)

### Breach Count Summary

Expected breaches at 1%: **44** (= 4,458 × 0.01)

| Model | Threshold | Breaches | vs Expected |
|-------|-----------|----------|-------------|
| Historical VaR | Empirical | 69 | +25 (+57%) |
| Parametric VaR (Normal) | −2.33σ | 84 | +40 (+91%) |
| **Parametric VaR (t-dist)** | **−2.81σ** | **44** | **0 (0%)** |
| EWMA VaR (Normal) | −2.33σ | 102 | +58 (+132%) |
| EWMA VaR (t-dist) | −2.81σ | 53 | +9 (+20%) |

Switching from Normal to t-dist reduces EWMA excess breaches by 48% (102 → 53). Parametric VaR achieves exact target calibration under t-dist because a slow-moving rolling window pairs cleanly with a fixed threshold multiplier. EWMA's reactive volatility estimates create a moving target that a single multiplier cannot perfectly track.


### Kupiec Test (Unconditional Coverage)

Tests whether the total breach rate is statistically consistent with the model's promised 1%.  
Null hypothesis: true breach probability = 1%. Critical value: χ²(1) = 3.841 at α = 5%.

| Model | Breaches | Observed Rate | LR Statistic | p-value | Result |
|-------|----------|---------------|--------------|---------|--------|
| Parametric VaR (Normal) | 84 | 1.88% | 27.946 | < 0.001 | **FAIL** |
| Parametric VaR (t-dist) | 44 | 0.99% | 0.008 | 0.930 | **PASS** |
| EWMA VaR (Normal) | 102 | 2.29% | 54.785 | < 0.001 | **FAIL** |
| EWMA VaR (t-dist) | 53 | 1.19% | 1.519 | 0.218 | **PASS** |


### Christoffersen Test (Independence + Conditional Coverage)

Tests whether breaches are independently distributed across time, or cluster in stress regimes.  
Critical values: χ²(1) = 3.841 (LR_uc, LR_ind) · χ²(2) = 5.991 (LR_cc)

| Model | LR_uc | LR_ind | LR_cc | Unc. Coverage | Independence | Combined |
|-------|-------|--------|-------|---------------|--------------|---------|
| Historical VaR | 11.588 | 11.585 | 23.173 | FAIL | FAIL | FAIL |
| Parametric VaR (Normal) | 27.964 | 10.712 | 38.677 | FAIL | FAIL | FAIL |
| Parametric VaR (t-dist) | 0.007 | 16.299 | 16.306 | PASS | FAIL | FAIL |
| EWMA VaR (Normal) | 54.785 | 4.273 | 59.058 | FAIL | FAIL | FAIL |
| EWMA VaR (t-dist) | 1.519 | 4.841 | 6.360 | PASS | FAIL | FAIL |

**Every model fails the independence test.** Breach clustering is structural, and isn't fixable by distributional recalibration alone.

#### Clustering Detail

| Model | P(breach \| calm yesterday) | P(breach \| breach yesterday) | Consecutive breaches (n₁₁) |
|-------|----------------------------|-------------------------------|----------------------------|
| Historical VaR | 1.44% | 8.70% | 6 |
| Parametric VaR (Normal) | 1.76% | 8.33% | 7 |
| Parametric VaR (t-dist) | 0.88% | 11.36% | 5 |
| EWMA VaR (Normal) | 2.20% | 5.88% | 6 |
| EWMA VaR (t-dist) | 1.14% | 5.66% | 3 |

After a breach day, every model shows a conditional breach probability 4–13x the unconditional 1% target. The cause: EWMA's α = 0.06 means 94% of yesterday's volatility underestimate carries into today's threshold. During sudden regime shifts (2008, 2020), the effective breach threshold collapses — a return only needs to cross −0.93σ rather than −2.33σ, inflating breach probability from 1% to ~17%.

---

## Repository Structure

```
├── ModelValidation_Engine.ipynb   # Full pipeline: data → models → backtesting
├── ValidationReport.md         # SR 11-7 structured validation writeup
└── README.md
```

---

## How to Run

```python
# Requirements
pip install yfinance pandas-datareader scipy seaborn matplotlib numpy

# Data
# yfinance: SPY, TLT, HYG, LQD, GLD | 2007-04-12 to 2025-12-31
# FRED: VIXCLS, DGS10, BAMLH0A0HYM2 (optional macro overlays)

# Run the notebook top to bottom — all outputs are self-contained
```

---

## Limitations and Next Steps

| Limitation | Root Cause | Recommended Fix |
|------------|------------|-----------------|
| All models fail independence test | Volatility persistence leaks into breach sequence | GARCH(1,1) or regime-switching volatility |
| EWMA t-dist: 53 breaches vs target 44 | Fixed multiplier can't track dynamic vol | Fit t-dist on rolling EWMA residuals per window |
| Crisis breaches irreducible | 2008/2020 are structural regime breaks | Expected Shortfall (CVaR) as complement |
| Normality violated in 67–76% of windows | Fat tails are persistent, not episodic | t-dist or skewed-t as base distribution |

---

*Disclaimer: This is my first independent quant risk project. Feedback and suggestions are welcome as I continue to improve and expand the framework*
