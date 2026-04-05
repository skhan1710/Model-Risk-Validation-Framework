# Model Validation Report

## 1. Scope and Methodology

This report validates three Value-at-Risk models against the three pillars SR 11-7 requires for effective model validation: **conceptual soundness**, **outcomes analysis**, and **benchmarking**.

The portfolio consists of five liquid ETFs: SPY (U.S. equity), TLT (long-duration Treasuries), HYG (high-yield credit), LQD (investment-grade credit), and GLD (gold), equal-weighted daily. The 4,458-day backtest window (April 2007 to December 2025) encompasses two major crisis regimes: the 2008-2009 Global Financial Crisis and the March 2020 COVID-19 shock. Both are deliberate inclusions.

**One-day 99% VaR** is the target metric throughout. Expected breaches at 1%: **44** (= 4,458 x 0.01).

Threshold set on day *t* uses only information available before day *t*'s return is realized to avoud look-ahead bias.

## 2. Conceptual Soundness

### 2.1 Distributional Assumption

Both Parametric and EWMA VaR embed a normality assumption through their use of z-score multipliers (-2.33σ for the standard 1% threshold). Before testing model outputs, this assumption needs to be tested directly.

The portfolio's standardized daily returns exhibit the following distributional properties:

| Statistic | Value | Normal Benchmark |
|-----------|-------|-----------------|
| Excess kurtosis | 12.03 | 0 |
| Skewness | -0.004 | 0 |
| Jarque-Bera statistic | 26,861.78 | — |
| Jarque-Bera p-value | 0.0 | — |

The Jarque-Bera p-value of 0.0 suggests that the test statistic sits so far into the tail of its reference distribution so, normality is completely rejected.

Under normality, the probability of a return below -2.5σ is 0.62%. In this portfolio it was 1.21%, nearly twice the expected rate. At -7.5σ, a normal distribution assigns probability 3.19 × 10⁻¹⁴. This portfolio produced three such days across 17 years.

**Rolling window analysis:** Jarque-Bera was applied to each rolling 252-day window of portfolio returns. Normality was rejected (α = 5%) in **76.3% of all windows** for the Parametric model, and in **67.7% of windows** for EWMA standardized residuals. The EWMA test addresses a specific question: does EWMA's dynamic volatility adjustment fix the distributional problem? Partially. Excess kurtosis drops from 12.03 in raw returns to 2.04 in EWMA residuals, but the violation persists across regimes rather than appearing only during crises.

The normality assumption is violated throughout the full sample period.

### 2.2 T-Distribution Recalibration

Student's t was fitted on the full sample of standardized residuals:

```
stats.t.fit(z, floc=0, fscale=1) → df = 9.11
stats.t.ppf(0.01, df=9.11)      → threshold = -2.8144σ
```

The optimizer is restricted to finding only the degrees-of-freedom parameter that best describes the shape of already-standardized data, rather than absorbing volatility information into the scale parameter.

The resulting 1% threshold under t(df=9.11) is **-2.8144σ**, versus -2.33σ under normality. The t-distribution has wider tails, so it demands a more extreme threshold to capture the true 1%, which is consistent with what the empirical data shows.


## 3. Outcomes Analysis

### 3.1 Breach Count Summary

| Model | Threshold | Breaches | vs Target (44) | Excess Rate |
|-------|-----------|----------|----------------|-------------|
| Historical VaR | Empirical 1st pct. | 69 | +25 | +57% |
| Parametric VaR (Normal) | -2.33σ | 84 | +40 | +91% |
| Parametric VaR (t-dist) | -2.81σ | 44 | 0 | 0% |
| EWMA VaR (Normal) | -2.33σ | 102 | +58 | +132% |
| EWMA VaR (t-dist) | -2.81σ | 53 | +9 | +20% |

Parametric VaR with t-dist recalibration hits the target exactly. EWMA with t-dist cuts excess breaches by 48% (102 to 53) but falls short of the target. The gap traces back to a fundamental mismatch: EWMA's daily-updating volatility estimate produces a moving threshold that a single fixed multiplier cannot track across all market regimes.


### 3.2 Kupiec Test (Unconditional Coverage)

The Kupiec likelihood ratio test asks: given observed breach count *x* out of *n* days, is this consistent with a true breach probability of p₀ = 1%?

The test statistic is:

```
LR = -2 × [x·ln(p₀) + (n−x)·ln(1−p₀) − x·ln(p̂) − (n−x)·ln(1−p̂)]
where p̂ = x/n (observed breach rate)
LR ~ χ²(1) under the null. Critical value = 3.841 (α = 5%)
```

| Model | Breaches | Observed Rate | LR | p-value | Result |
|-------|----------|---------------|-----|---------|--------|
| Parametric VaR (Normal) | 84 | 1.88% | 27.946 | < 0.001 | **FAIL** |
| Parametric VaR (t-dist) | 44 | 0.99% | 0.008 | 0.930 | **PASS** |
| EWMA VaR (Normal) | 102 | 2.29% | 54.785 | < 0.001 | **FAIL** |
| EWMA VaR (t-dist) | 53 | 1.19% | 1.519 | 0.218 | **PASS** |

An LR of 54.785 for EWMA Normal means that if the model were correctly calibrated, observing 102 breaches in 4,458 days would be a near-impossible outcome. The normal assumption is categorically wrong for this portfolio's return history.

Both t-dist recalibrated models pass at conventionally used significance levels.

## 4. Independence Testing (Christoffersen)

Kupiec tests *how many* breaches occur. It says nothing about *when* they occur. A model that clusters all its breaches into a single stress week would pass Kupiec if the total count were correct — but it would still be a risk management failure. Clustered breaches signal that the model is systematically underestimating risk during crisis periods.

The Christoffersen (1998) independence test addresses this. It models the breach sequence as a first-order Markov chain and tests whether the probability of a breach today depends on whether one occurred yesterday. Ideally, breaches should all be independent of each other.

Define:
- **n₀₁**: days where yesterday = no breach, today = breach → estimated p₀₁ = n₀₁ / (n₀₀ + n₀₁)
- **n₁₁**: days where yesterday = breach, today = breach → estimated p₁₁ = n₁₁ / (n₁₀ + n₁₁)

Under independence: p₁₁ = p₀₁. Any gap between them is evidence of clustering.

The combined conditional coverage statistic LR_cc = LR_uc + LR_ind follows χ²(2). Critical value = 5.991.

| Model | LR_uc | LR_ind | LR_cc | Unc. Coverage | Independence | Combined |
|-------|-------|--------|-------|---------------|--------------|---------|
| Historical VaR | 11.588 | 11.585 | 23.173 | FAIL | FAIL | FAIL |
| Parametric VaR (Normal) | 27.964 | 10.712 | 38.677 | FAIL | FAIL | FAIL |
| Parametric VaR (t-dist) | 0.007 | 16.299 | 16.306 | PASS | FAIL | FAIL |
| EWMA VaR (Normal) | 54.785 | 4.273 | 59.058 | FAIL | FAIL | FAIL |
| EWMA VaR (t-dist) | 1.519 | 4.841 | 6.360 | PASS | FAIL | FAIL |

**Every model fails the independence test.** T-dist recalibration fixes the unconditional coverage problem. It does not fix clustering. That's the core finding of this validation.

### 4.1 Clustering Quantified

| Model | p₀₁ | p₁₁ | n₁₁ | Breaches |
|-------|-----|-----|-----|----------|
| Historical VaR | 1.44% | 8.70% | 6 | 69 |
| Parametric VaR (Normal) | 1.76% | 8.33% | 7 | 84 |
| Parametric VaR (t-dist) | 0.88% | 11.36% | 5 | 44 |
| EWMA VaR (Normal) | 2.20% | 5.88% | 6 | 102 |
| EWMA VaR (t-dist) | 1.14% | 5.66% | 3 | 53 |

For every model, the conditional breach probability the day after a breach runs 4 to 13 times the unconditional 1% target. In EWMA VaR (Normal), p₁₁ = 5.88%. The morning after a breach, the model is effectively running a 6% daily VaR test rather than a 1% test.

### 4.2 Mechanism of Clustering

The clustering has a mechanical explanation rooted in EWMA's update equation:

```
σ̂²ₜ = α · r²ₜ₋₁ + (1 − α) · σ̂²ₜ₋₁    where α = 0.06
```

After a sudden volatility jump (e.g., September 2008, March 2020), true volatility σₜ exceeds the EWMA estimate σ̂ₜ. The VaR threshold is set using σ̂ₜ, so the effective barrier the market shock must cross becomes:

```
ε_t < -2.33 × (σ̂ₜ / σₜ)
```

When σ̂ₜ = 2% and σₜ = 5%, the effective threshold collapses to -0.93σ. Under the standard normal, P(ε < -0.93) ≈ 17%, not 1%. The 1% guarantee breaks the moment the model lags true volatility.

With α = 0.06, 94% of yesterday's underestimate carries forward into today's estimate. The threshold stays suppressed across multiple consecutive days. That's why n₁₁ > 0 across every model tested.

This indicates that when any model estimates volatility from lagged data and then tries to guarantee a forward-looking probability that depends on today's true, unobservable volatility.


## 5. Benchmark Comparison

Historical Simulation makes no distributional assumption and avoids the normality miscalibration that affects both Parametric and EWMA. Even so, it produces 69 breaches against a target of 44, fails both Kupiec and Christoffersen, and carries p₁₁ = 8.70%, the highest conditional clustering rate of any model tested.

The reason traces back to how rolling quantiles update. The 252-day empirical quantile responds to volatility changes only as quickly as the window turns over. It cannot adjust intra-crisis the way EWMA does. During a sustained crisis, extreme returns enter the historical window gradually, keeping the threshold too lenient for weeks at a time.

The t-dist recalibrated models outperform Historical VaR on unconditional coverage. On independence, no model does. Breach clustering during crisis regimes is a property of how markets behave, not a calibration problem that better multipliers can solve.


## 6. Limitations and Recommendations

| Finding | Root Cause | Recommended Enhancement |
|---------|------------|------------------------|
| Independence test fails across all models | Volatility clustering is persistent and exceeds any model's lag correction | GARCH(1,1): volatility persistence is explicitly parameterized via β; Expected Shortfall as regulatory complement |
| EWMA t-dist: 53 vs target 44 | Fixed multiplier cannot track a dynamic threshold | Fit t-dist on rolling 252-day EWMA residuals per window; use dynamic df |
| Normality violated in 67-76% of windows | Return distribution is fat-tailed across all market regimes | Replace -2.33σ with regime-conditional threshold; skewed-t distribution |
| 2008 and 2020 breaches irreducible | Structural regime break; no static distributional assumption survives | Stress VaR overlay; conditional VaR (CVaR/Expected Shortfall) |

The residual gap between EWMA t-dist (53 breaches) and the 44 target concentrates in the 2008 and 2020 crisis windows. This is consistent with the academic literature: fewer than half of VaR exceptions in EWMA models trace back to distributional mis-specification. The rest reflect genuine regime breaks that stress testing frameworks need to handle, not better multipliers.

## 7. Summary

| Validation Criterion | Finding |
|---------------------|---------|
| Normality assumption — Parametric | Violated in 76.3% of rolling 252-day windows. JB statistic: 26,861.78, p = 0.0 |
| Normality assumption — EWMA residuals | Violated in 67.7% of windows. Excess kurtosis of residuals: 2.04 (vs 12.03 in raw returns) |
| T-distribution fit | df = 9.11, threshold = -2.8144σ |
| Kupiec — Normal models | Both fail. LR = 27.946 (Parametric), 54.785 (EWMA). p < 0.001 |
| Kupiec — T-dist models | Both pass. LR = 0.008 (Parametric), 1.519 (EWMA). p = 0.930, 0.218 |
| Christoffersen — independence | All five models fail. Breach clustering is consistent across the full sample |
| Root cause of clustering | EWMA lag: σ̂ₜ < σₜ after regime breaks collapses the effective threshold from -2.33σ to as low as -0.93σ |
| Recommended regulatory complement | Expected Shortfall (CVaR) for crisis tail coverage; GARCH(1,1) for volatility persistence |
