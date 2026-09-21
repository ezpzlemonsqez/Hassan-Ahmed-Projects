# Hassan Ahmed — Quantitative Finance & Analytics Portfolio

This repository contains my portfolio of quantitative finance, financial mathematics, actuarial risk and analytics projects. The projects focus on data cleaning, modelling, validation, backtesting, risk analysis and clear reporting.

## Live Portfolio

View the portfolio here:

https://ezpzlemonsqez.github.io/Hassan-Ahmed-Projects/

## Project Areas

### Quantitative Finance Research

* **Rough Volatility Estimation**
  Implemented roughness and Hurst estimators, with validation through Monte Carlo simulation and intraday volatility experiments.

* **CAPM-Informed Mean-Variance Portfolio Optimisation**
  Built a sector ETF portfolio strategy using rolling CAPM betas, covariance shrinkage, constrained Markowitz optimisation, walk-forward testing and portfolio risk metrics.

* **Volatility Forecasting & Risk Signal Analysis**
  Forecasted log-VIX using ARIMA and regression models, evaluated rolling one-day-ahead forecasts and classified volatility regimes using rolling thresholds.

* **Derivatives Pricing & Greeks Engine**
  Built an options pricing workflow with pricing models, Greeks, hedge ratios, scenario testing and validation checks.

### Financial Mathematics & Risk

* **Annuity Liability Valuation & Sensitivity**
  Modelled survival-weighted annuity cashflows using mortality data and gilt-based discounting, with sensitivities to rates, longevity, mortality, inflation and expenses.

* **General Insurance Reserving & Claims Triangle Analysis**
  Analysed paid and incurred claims triangles using Chain Ladder and Bornhuetter-Ferguson methods, with reserve estimates and uncertainty checks.

* **KPI Forecasting & Monitoring Diagnostics**
  Built ARIMA and regression models for synthetic insurance and operational KPIs, with rolling evaluation, residual checks and anomaly monitoring.

* **Commercial Analysis & Strategic Insight**
  Analysed 7,032 IBM Telco customers using Python and SQL, identified churn concentrations by contract type, tenure, payment method and monthly charge band, and built logistic regression risk bands to support retention prioritisation.

## Tools and Skills

* **Programming:** Python, R, SQL, C++, HTML/CSS
* **Data analysis:** Pandas, NumPy, DuckDB, SQL, tidyverse
* **Modelling:** ARIMA, regression, logistic regression, Monte Carlo, optimisation
* **Finance and risk:** derivatives pricing, portfolio optimisation, reserving, liability valuation, churn segmentation, risk monitoring
* **Reporting:** LaTeX reports, embedded PDF viewers, code previews and dashboard-style outputs

## Website structure

The site is static HTML and CSS, published from the repository root on GitHub Pages. No build step or package installation is required.

| File or folder | Purpose |
| --- | --- |
| `index.html` | Recruiter-facing introduction, qualifications, About section, eight projects in two subject groups, and contact links. |
| `assets/site.css` | Shared typography, colours, layout, native-link hit areas, responsive breakpoints, focus states, reduced-motion support and print rules. |
| `assets/normal-density.svg` | Mathematical illustration of the bivariate standard normal density; not an empirical project result. |
| `assets/fonts/` | Locally hosted DM Sans and Newsreader fonts, with their open-source licences. |
| `assets/code-preview.css` | Shared dark, monospace source-preview styling and syntax palette. |
| `assets/code-preview.js` | Loads source files as text; optionally highlights and copies them. A highlighting failure leaves readable plain text. |
| `assets/vendor/` | Local Highlight.js 11.9.0 and Python, R and C++ language definitions, with the licence. |
| `projects/<project>/*.html` | Existing report, code-overview and source-preview URLs. |
| `projects/<project>/*.pdf` | Original project reports, displayed in each visitor’s native PDF reader. |
| `projects/<project>/code/` and `data/` | Original research implementations and inputs. |
| `projects/MSc Project/FM50FinalCode.html` | Original exported notebook, embedded in the MSc code reader. |
| `DESIGN_NOTES.md` | Design rationale, original behaviour, preservation choices and maintenance notes. |

Project directories: `MSc Project`, `Portfolio Optimisation`, `Volatility Forecasting Risk Signal Analysis`, `Option Pricing Project (MC v CN)`, `Annuity Liability Valuation`, `General Insurance Reserving`, `KPI Forecasting Monitoring Diagnostics`, and `Commercial Analysis`.

### Local preview

From the repository folder, run `python -m http.server 8000`, then visit `http://localhost:8000`. A local HTTP server is needed for source-code fetches; opening `index.html` directly is sufficient to inspect the homepage, but browsers usually block local file fetches.

### Maintenance

- Edit homepage content in `index.html`. Project rows use native links; the title link covers the row, while the report and code links remain separately accessible.
- Update shared appearance in `assets/site.css`, or the code palette in `assets/code-preview.css`.
- Update a research implementation in its original source file. Its preview fetches that file directly, so the source does not need to be copied into HTML. The exported MSc notebook remains a separate export and should be regenerated if its notebook changes.
- When adding a project, follow an existing report/code/preview page. Keep relative links so the site works under `/Hassan-Ahmed-Projects/` and on a local server.
- The original notebook export retains its own MathJax/RequireJS references. The homepage, project reader styling and standalone source previews use local assets.

## Notes

AI tools may have been used to support drafting, formatting, debugging and wording refinement. All modelling choices, code review, interpretation, validation and final project outputs were checked and owned by me.

## Contact

* LinkedIn: https://www.linkedin.com/in/hassan-ahmed-72a840235/
* GitHub: https://github.com/ezpzlemonsqez
