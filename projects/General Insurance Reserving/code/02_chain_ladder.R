# 02_chain_ladder.R
# Deterministic Chain Ladder estimates for paid and incurred triangles.

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

file.remove(list.files("outputs/tables", pattern = "^02_.*\\.csv$", full.names = TRUE))
file.remove(list.files("outputs/figures", pattern = "^02_.*\\.png$", full.names = TRUE))

step1_file <- "data/processed/01_triangle_objects.rds"
if (!file.exists(step1_file)) {
  stop("Missing Step 1 object file. Run scripts/01_prepare_triangles.R first.")
}

obj <- readRDS(step1_file)

observed_triangle_long <- obj$observed_triangle_long
actual_full_ultimate_by_ay <- obj$actual_full_ultimate_by_ay
valuation_year <- obj$valuation_year

calculate_chain_ladder <- function(data, value_col, basis_name, actual_col_name) {
  triangle_long <- data %>%
    transmute(
      accident_year = as.integer(accident_year),
      development_lag = as.integer(development_lag),
      cumulative_loss = as.numeric(.data[[value_col]])
    ) %>%
    arrange(accident_year, development_lag)

  lags <- sort(unique(triangle_long$development_lag))
  max_lag <- max(lags)

  development_factors <- lapply(lags[lags < max_lag], function(lag_now) {
    paired <- triangle_long %>%
      filter(development_lag == lag_now) %>%
      select(accident_year, current_loss = cumulative_loss) %>%
      inner_join(
        triangle_long %>%
          filter(development_lag == lag_now + 1) %>%
          select(accident_year, next_loss = cumulative_loss),
        by = "accident_year"
      ) %>%
      filter(!is.na(current_loss), !is.na(next_loss), current_loss > 0)

    tibble(
      basis = basis_name,
      from_lag = lag_now,
      to_lag = lag_now + 1,
      selected_factor = sum(paired$next_loss, na.rm = TRUE) /
        sum(paired$current_loss, na.rm = TRUE),
      n_pairs = nrow(paired)
    )
  }) %>%
    bind_rows()

  cdf_table <- lapply(lags, function(lag_now) {
    factor_to_ultimate <- if (lag_now == max_lag) {
      1
    } else {
      development_factors %>%
        filter(from_lag >= lag_now) %>%
        summarise(value = prod(selected_factor, na.rm = TRUE)) %>%
        pull(value)
    }

    tibble(
      basis = basis_name,
      development_lag = lag_now,
      factor_to_ultimate = factor_to_ultimate
    )
  }) %>%
    bind_rows()

  latest_observed <- triangle_long %>%
    group_by(accident_year) %>%
    slice_max(order_by = development_lag, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    rename(
      latest_development_lag = development_lag,
      latest_observed_loss = cumulative_loss
    )

  actual_comparison <- actual_full_ultimate_by_ay %>%
    select(accident_year, actual_10yr = all_of(actual_col_name))

  estimates <- latest_observed %>%
    left_join(
      cdf_table,
      by = c("latest_development_lag" = "development_lag")
    ) %>%
    mutate(
      basis = basis_name,
      ultimate_estimate = latest_observed_loss * factor_to_ultimate,
      reserve_estimate = ultimate_estimate - latest_observed_loss
    ) %>%
    left_join(actual_comparison, by = "accident_year") %>%
    mutate(
      ultimate_error = ultimate_estimate - actual_10yr,
      ultimate_error_pct = ultimate_error / actual_10yr
    ) %>%
    select(
      basis,
      accident_year,
      latest_development_lag,
      latest_observed_loss,
      factor_to_ultimate,
      ultimate_estimate,
      reserve_estimate,
      actual_10yr,
      ultimate_error,
      ultimate_error_pct
    )

  summary <- estimates %>%
    summarise(
      basis = basis_name,
      valuation_year = valuation_year,
      latest_observed_m = sum(latest_observed_loss, na.rm = TRUE) / 1e6,
      ultimate_estimate_m = sum(ultimate_estimate, na.rm = TRUE) / 1e6,
      reserve_estimate_m = sum(reserve_estimate, na.rm = TRUE) / 1e6,
      actual_10yr_m = sum(actual_10yr, na.rm = TRUE) / 1e6,
      ultimate_error_m = (sum(ultimate_estimate, na.rm = TRUE) - sum(actual_10yr, na.rm = TRUE)) / 1e6,
      ultimate_error_pct = ultimate_error_m / actual_10yr_m,
      accident_years = n(),
      .groups = "drop"
    )

  list(
    development_factors = development_factors,
    cdf_table = cdf_table,
    estimates = estimates,
    summary = summary
  )
}

paid_cl <- calculate_chain_ladder(
  observed_triangle_long,
  value_col = "cum_paid_loss",
  basis_name = "Paid",
  actual_col_name = "actual_paid_10yr"
)

incurred_cl <- calculate_chain_ladder(
  observed_triangle_long,
  value_col = "incurred_losses",
  basis_name = "Incurred",
  actual_col_name = "actual_incurred_10yr"
)

development_factors <- bind_rows(paid_cl$development_factors, incurred_cl$development_factors)
cdf_table <- bind_rows(paid_cl$cdf_table, incurred_cl$cdf_table)
chain_ladder_by_ay <- bind_rows(paid_cl$estimates, incurred_cl$estimates)
chain_ladder_summary <- bind_rows(paid_cl$summary, incurred_cl$summary)

readr::write_csv(chain_ladder_summary, "outputs/tables/02_chain_ladder_summary_report.csv")

factor_plot <- development_factors %>%
  ggplot(aes(x = from_lag, y = selected_factor, group = basis, linetype = basis)) +
  geom_line() +
  geom_point(size = 1.5) +
  scale_x_continuous(breaks = sort(unique(development_factors$from_lag))) +
  labs(
    title = "Selected Chain Ladder Age-to-Age Development Factors",
    x = "Development lag",
    y = "Selected age-to-age factor",
    linetype = "Basis"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  "outputs/figures/02_chain_ladder_development_factors.png",
  factor_plot,
  width = 9,
  height = 6,
  dpi = 300
)

saveRDS(
  list(
    valuation_year = valuation_year,
    development_factors = development_factors,
    cdf_table = cdf_table,
    chain_ladder_by_ay = chain_ladder_by_ay,
    chain_ladder_summary = chain_ladder_summary
  ),
  "data/processed/02_chain_ladder_results.rds"
)

message("")
message("Step 2 complete.")
message("Report outputs:")
message(" - outputs/tables/02_chain_ladder_summary_report.csv")
message(" - outputs/figures/02_chain_ladder_development_factors.png")
message("Pipeline object:")
message(" - data/processed/02_chain_ladder_results.rds")
print(chain_ladder_summary)
