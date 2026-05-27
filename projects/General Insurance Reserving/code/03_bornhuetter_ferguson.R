# 03_bornhuetter_ferguson.R
# Bornhuetter-Ferguson estimates and method comparison.

project_root <- "C:/Users/hassa/Documents/Code/R/General-Insurance-Reserving"
setwd(project_root)

packages <- c("readr", "dplyr", "tidyr", "ggplot2", "scales", "tibble")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

invisible(lapply(packages, install_if_missing))

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(tibble)

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

file.remove(list.files("outputs/tables", pattern = "^03_.*\\.csv$", full.names = TRUE))
file.remove(list.files("outputs/figures", pattern = "^03_.*\\.png$", full.names = TRUE))

step1_file <- "data/processed/01_triangle_objects.rds"
step2_file <- "data/processed/02_chain_ladder_results.rds"

if (!file.exists(step1_file)) {
  stop("Missing Step 1 object file. Run scripts/01_prepare_triangles.R first.")
}

if (!file.exists(step2_file)) {
  stop("Missing Step 2 object file. Run scripts/02_chain_ladder.R first.")
}

obj1 <- readRDS(step1_file)
obj2 <- readRDS(step2_file)

valuation_year <- obj1$valuation_year
latest_diagonal <- obj1$latest_diagonal
earned_premium_by_ay <- obj1$earned_premium_by_ay
actual_full_ultimate_by_ay <- obj1$actual_full_ultimate_by_ay
cdf_table <- obj2$cdf_table
chain_ladder_by_ay <- obj2$chain_ladder_by_ay

# Use older observed incurred ratios, not future full outcomes, for the BF prior.
expected_lr_selection <- latest_diagonal %>%
  left_join(earned_premium_by_ay, by = "accident_year") %>%
  filter(accident_year <= 2002) %>%
  transmute(
    accident_year,
    latest_incurred_m = latest_observed_incurred / 1e6,
    earned_premium_net_m = earned_premium_net / 1e6,
    observed_incurred_lr = latest_observed_incurred / earned_premium_net
  )

selected_expected_loss_ratio <- sum(expected_lr_selection$latest_incurred_m, na.rm = TRUE) /
  sum(expected_lr_selection$earned_premium_net_m, na.rm = TRUE)

latest_for_bf <- latest_diagonal %>%
  select(
    accident_year,
    latest_observed_development_lag,
    latest_observed_paid,
    latest_observed_incurred
  ) %>%
  left_join(
    earned_premium_by_ay %>% select(accident_year, earned_premium_net),
    by = "accident_year"
  ) %>%
  mutate(expected_ultimate_loss = earned_premium_net * selected_expected_loss_ratio)

calculate_bf <- function(latest_data, basis_name, latest_col, actual_col_name) {
  cdf_basis <- cdf_table %>%
    filter(basis == basis_name) %>%
    select(
      latest_observed_development_lag = development_lag,
      factor_to_ultimate
    )

  actual_comparison <- actual_full_ultimate_by_ay %>%
    select(accident_year, actual_10yr = all_of(actual_col_name))

  latest_data %>%
    select(
      accident_year,
      latest_observed_development_lag,
      earned_premium_net,
      expected_ultimate_loss,
      latest_observed_loss = all_of(latest_col)
    ) %>%
    left_join(cdf_basis, by = "latest_observed_development_lag") %>%
    mutate(
      basis = basis_name,
      reported_proportion = 1 / factor_to_ultimate,
      unreported_proportion = 1 - reported_proportion,
      bf_reserve_estimate = expected_ultimate_loss * unreported_proportion,
      bf_ultimate_estimate = latest_observed_loss + bf_reserve_estimate
    ) %>%
    left_join(actual_comparison, by = "accident_year") %>%
    mutate(
      bf_error = bf_ultimate_estimate - actual_10yr,
      bf_error_pct = bf_error / actual_10yr
    ) %>%
    select(
      basis,
      accident_year,
      latest_observed_development_lag,
      latest_observed_loss,
      earned_premium_net,
      expected_ultimate_loss,
      factor_to_ultimate,
      reported_proportion,
      unreported_proportion,
      bf_ultimate_estimate,
      bf_reserve_estimate,
      actual_10yr,
      bf_error,
      bf_error_pct
    )
}

bf_by_ay <- bind_rows(
  calculate_bf(latest_for_bf, "Paid", "latest_observed_paid", "actual_paid_10yr"),
  calculate_bf(latest_for_bf, "Incurred", "latest_observed_incurred", "actual_incurred_10yr")
)

method_comparison_by_ay <- chain_ladder_by_ay %>%
  select(
    basis,
    accident_year,
    cl_ultimate_estimate = ultimate_estimate,
    cl_reserve_estimate = reserve_estimate,
    actual_10yr
  ) %>%
  left_join(
    bf_by_ay %>%
      select(basis, accident_year, bf_ultimate_estimate, bf_reserve_estimate),
    by = c("basis", "accident_year")
  ) %>%
  mutate(
    cl_error = cl_ultimate_estimate - actual_10yr,
    bf_error = bf_ultimate_estimate - actual_10yr,
    cl_abs_error = abs(cl_error),
    bf_abs_error = abs(bf_error)
  )

method_comparison_summary <- method_comparison_by_ay %>%
  group_by(basis) %>%
  summarise(
    selected_expected_loss_ratio = selected_expected_loss_ratio,
    cl_ultimate_m = sum(cl_ultimate_estimate, na.rm = TRUE) / 1e6,
    bf_ultimate_m = sum(bf_ultimate_estimate, na.rm = TRUE) / 1e6,
    actual_ultimate_m = sum(actual_10yr, na.rm = TRUE) / 1e6,
    cl_reserve_m = sum(cl_reserve_estimate, na.rm = TRUE) / 1e6,
    bf_reserve_m = sum(bf_reserve_estimate, na.rm = TRUE) / 1e6,
    cl_error_m = (sum(cl_ultimate_estimate, na.rm = TRUE) - sum(actual_10yr, na.rm = TRUE)) / 1e6,
    bf_error_m = (sum(bf_ultimate_estimate, na.rm = TRUE) - sum(actual_10yr, na.rm = TRUE)) / 1e6,
    cl_mae_m = mean(cl_abs_error, na.rm = TRUE) / 1e6,
    bf_mae_m = mean(bf_abs_error, na.rm = TRUE) / 1e6,
    .groups = "drop"
  )

readr::write_csv(expected_lr_selection, "outputs/tables/03_expected_loss_ratio_selection_report.csv")
readr::write_csv(method_comparison_summary, "outputs/tables/03_method_comparison_summary_report.csv")

method_reserve_comparison <- method_comparison_by_ay %>%
  select(basis, accident_year, cl_reserve_estimate, bf_reserve_estimate) %>%
  pivot_longer(
    cols = c(cl_reserve_estimate, bf_reserve_estimate),
    names_to = "method",
    values_to = "reserve_estimate"
  ) %>%
  mutate(
    method = recode(
      method,
      cl_reserve_estimate = "Chain Ladder",
      bf_reserve_estimate = "Bornhuetter-Ferguson"
    )
  ) %>%
  ggplot(aes(x = factor(accident_year), y = reserve_estimate / 1e6, fill = method)) +
  geom_col(position = "dodge") +
  facet_wrap(~ basis, scales = "free_y") +
  labs(
    title = "Reserve Estimates by Method and Accident Year",
    x = "Accident year",
    y = "Reserve estimate (m)",
    fill = "Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  "outputs/figures/03_method_reserve_comparison.png",
  method_reserve_comparison,
  width = 9,
  height = 6,
  dpi = 300
)

ultimate_comparison <- method_comparison_by_ay %>%
  select(basis, accident_year, cl_ultimate_estimate, bf_ultimate_estimate, actual_10yr) %>%
  pivot_longer(
    cols = c(cl_ultimate_estimate, bf_ultimate_estimate, actual_10yr),
    names_to = "series",
    values_to = "ultimate"
  ) %>%
  mutate(
    series = recode(
      series,
      cl_ultimate_estimate = "Chain Ladder",
      bf_ultimate_estimate = "Bornhuetter-Ferguson",
      actual_10yr = "Actual 10-year outcome"
    )
  ) %>%
  ggplot(aes(x = accident_year, y = ultimate / 1e6, group = series, linetype = series)) +
  geom_line() +
  geom_point(size = 1.5) +
  facet_wrap(~ basis, scales = "free_y") +
  scale_x_continuous(breaks = sort(unique(method_comparison_by_ay$accident_year))) +
  labs(
    title = "Ultimate Loss Estimate by Method vs Actual Outcome",
    x = "Accident year",
    y = "Ultimate loss (m)",
    linetype = ""
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  "outputs/figures/03_ultimate_method_comparison.png",
  ultimate_comparison,
  width = 9,
  height = 6,
  dpi = 300
)

saveRDS(
  list(
    valuation_year = valuation_year,
    selected_expected_loss_ratio = selected_expected_loss_ratio,
    expected_lr_selection = expected_lr_selection,
    bf_by_ay = bf_by_ay,
    method_comparison_by_ay = method_comparison_by_ay,
    method_comparison_summary = method_comparison_summary
  ),
  "data/processed/03_bornhuetter_ferguson_results.rds"
)

message("")
message("Step 3 complete.")
message("Report outputs:")
message(" - outputs/tables/03_expected_loss_ratio_selection_report.csv")
message(" - outputs/tables/03_method_comparison_summary_report.csv")
message(" - outputs/figures/03_method_reserve_comparison.png")
message(" - outputs/figures/03_ultimate_method_comparison.png")
message("Pipeline object:")
message(" - data/processed/03_bornhuetter_ferguson_results.rds")
message("")
message("Selected expected loss ratio:")
print(selected_expected_loss_ratio)
message("")
print(method_comparison_summary)
