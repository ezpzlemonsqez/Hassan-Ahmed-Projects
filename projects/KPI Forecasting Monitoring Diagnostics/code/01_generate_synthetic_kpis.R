# Synthetic insurance and operational KPI data.
# The dataset is deliberately simple enough to audit, but includes trend,
# seasonality, shocks and linked business drivers.

set.seed(42)

check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required. Install it with install.packages('", pkg, "').")
  }
}

invisible(lapply(c("dplyr", "readr", "lubridate"), check_package))

library(dplyr)
library(readr)
library(lubridate)

out_dir <- file.path(project_root, "outputs", "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

months <- seq.Date(as.Date("2019-01-01"), as.Date("2025-12-01"), by = "month")
n <- length(months)
t <- seq_len(n)
month_no <- month(months)
season <- sin(2 * pi * month_no / 12)
season_2 <- cos(2 * pi * month_no / 12)

# A stress driver used by several KPIs. It rises around 2022-2023 and is noisy,
# mimicking a weaker operating environment.
stress_base <- 0.25 + 0.18 * sin(2 * pi * (t - 4) / 36)
stress_event <- ifelse(months >= as.Date("2022-04-01") & months <= as.Date("2023-04-01"), 0.35, 0)
economic_stress <- pmax(0, pmin(1, stress_base + stress_event + rnorm(n, 0, 0.05)))

inflation_index <- 100 + cumsum(0.22 + 0.12 * economic_stress + rnorm(n, 0, 0.04))

policy_count <- round(
  11800 + 20 * t + 260 * season_2 - 220 * economic_stress + rnorm(n, 0, 120)
)
policy_count <- pmax(policy_count, 9000)

new_policies <- round(520 + 35 * season + 0.025 * policy_count - 90 * economic_stress + rnorm(n, 0, 35))
new_policies <- pmax(new_policies, 50)

lapse_rate <- 0.037 + 0.004 * season_2 + 0.013 * economic_stress + rnorm(n, 0, 0.0022)
lapse_rate[months == as.Date("2024-03-01")] <- lapse_rate[months == as.Date("2024-03-01")] + 0.014
lapse_rate <- pmax(0.025, pmin(0.075, lapse_rate))

lapsed_policies <- round(policy_count * lapse_rate)

claims_lambda <- 365 + 0.0065 * policy_count + 32 * season + 58 * economic_stress
claims_count <- rpois(n, lambda = pmax(60, claims_lambda))
claims_count[months == as.Date("2023-02-01")] <- claims_count[months == as.Date("2023-02-01")] + 105
claims_count[months == as.Date("2023-10-01")] <- claims_count[months == as.Date("2023-10-01")] - 70
claims_count <- pmax(claims_count, 40)

average_claim_cost <- 1120 + 5.4 * (inflation_index - 100) + 55 * season_2 + 85 * economic_stress + rnorm(n, 0, 45)
average_claim_cost <- pmax(average_claim_cost, 800)
total_claims <- claims_count * average_claim_cost

premium_per_policy <- 91 + 0.16 * t + 0.19 * (inflation_index - 100) + 2.8 * season_2 + rnorm(n, 0, 1.6)
earned_premium <- policy_count * premium_per_policy
earned_premium[months == as.Date("2024-10-01")] <- earned_premium[months == as.Date("2024-10-01")] * 0.93

aquisition_expense_ratio <- 0.105 + 0.01 * season + rnorm(n, 0, 0.003)
admin_expense_ratio <- 0.118 + 0.012 * economic_stress + rnorm(n, 0, 0.0025)
expense_ratio <- pmax(0.16, pmin(0.29, aquisition_expense_ratio + admin_expense_ratio))
operating_expense <- earned_premium * expense_ratio

loss_ratio <- total_claims / earned_premium
loss_ratio[months == as.Date("2023-02-01")] <- loss_ratio[months == as.Date("2023-02-01")] + 0.11
loss_ratio[months == as.Date("2023-11-01")] <- loss_ratio[months == as.Date("2023-11-01")] - 0.09
loss_ratio <- pmax(0.35, pmin(0.95, loss_ratio))

# Operating margin is derived from claims and expenses, then given a small noise
# term so later regressions do not reproduce it mechanically.
operating_margin <- 1 - loss_ratio - expense_ratio + rnorm(n, 0, 0.006)
retention_rate <- 1 - lapse_rate

kpi_data <- tibble(
  date = months,
  year = year(months),
  month = month_no,
  month_index = t,
  policy_count = policy_count,
  new_policies = new_policies,
  lapsed_policies = lapsed_policies,
  retention_rate = round(retention_rate, 5),
  economic_stress = round(economic_stress, 5),
  inflation_index = round(inflation_index, 5),
  claims_count = claims_count,
  average_claim_cost = round(average_claim_cost, 2),
  total_claims = round(total_claims, 2),
  earned_premium = round(earned_premium, 2),
  expense_ratio = round(expense_ratio, 5),
  operating_expense = round(operating_expense, 2),
  loss_ratio = round(loss_ratio, 5),
  lapse_rate = round(lapse_rate, 5),
  operating_margin = round(operating_margin, 5)
)

out_file <- file.path(out_dir, "synthetic_insurance_operational_kpis.csv")
write_csv(kpi_data, out_file)
cat("Synthetic KPI data written to:", out_file, "\n")
