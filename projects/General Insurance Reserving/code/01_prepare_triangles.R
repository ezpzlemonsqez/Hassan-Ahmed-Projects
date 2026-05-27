# 01_prepare_triangles.R
# Build the observed 2007 reserving triangle and compact data diagnostics.

project_root <- "C:/Users/hassa/Documents/Code/R/General-Insurance-Reserving"
setwd(project_root)

packages <- c("readr", "dplyr", "tidyr", "janitor", "ggplot2", "scales", "tibble", "ChainLadder")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

invisible(lapply(packages, install_if_missing))

library(readr)
library(dplyr)
library(tidyr)
library(janitor)
library(ggplot2)
library(scales)
library(tibble)
library(ChainLadder)

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Remove old Step 1 report outputs.
file.remove(list.files("outputs/tables", pattern = "^01_.*\\.csv$", full.names = TRUE))
file.remove(list.files("outputs/figures", pattern = "^01_.*\\.png$", full.names = TRUE))

input_file <- "data/raw/ppauto_pos98-07.csv"
valuation_year <- 2007

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

raw_data <- readr::read_csv(input_file, show_col_types = FALSE) %>%
  janitor::clean_names()

required_cols <- c(
  "grcode", "grname", "accident_year", "development_year", "development_lag",
  "incurred_losses", "cum_paid_loss", "bulk_loss",
  "earned_prem_dir", "earned_prem_ceded", "earned_prem_net",
  "single", "posted_reserves2007"
)

missing_cols <- setdiff(required_cols, names(raw_data))
if (length(missing_cols) > 0) {
  stop("Missing expected columns: ", paste(missing_cols, collapse = ", "))
}

portfolio_full <- raw_data %>%
  group_by(accident_year, development_lag) %>%
  summarise(
    development_year = max(development_year, na.rm = TRUE),
    incurred_losses = sum(incurred_losses, na.rm = TRUE),
    cum_paid_loss = sum(cum_paid_loss, na.rm = TRUE),
    bulk_loss = sum(bulk_loss, na.rm = TRUE),
    posted_reserves2007 = sum(posted_reserves2007, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(calendar_year = accident_year + development_lag - 1) %>%
  arrange(accident_year, development_lag)

observed_triangle_long <- portfolio_full %>%
  filter(calendar_year <= valuation_year) %>%
  arrange(accident_year, development_lag)

latest_diagonal <- observed_triangle_long %>%
  group_by(accident_year) %>%
  slice_max(order_by = development_lag, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    accident_year,
    latest_observed_development_lag = development_lag,
    latest_observed_paid = cum_paid_loss,
    latest_observed_incurred = incurred_losses,
    latest_observed_bulk_loss = bulk_loss
  ) %>%
  arrange(accident_year)

actual_full_ultimate_by_ay <- portfolio_full %>%
  group_by(accident_year) %>%
  slice_max(order_by = development_lag, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    accident_year,
    actual_paid_10yr = cum_paid_loss,
    actual_incurred_10yr = incurred_losses,
    actual_bulk_loss_10yr = bulk_loss
  ) %>%
  arrange(accident_year)

earned_premium_by_ay <- raw_data %>%
  distinct(grcode, accident_year, earned_prem_net, earned_prem_dir, earned_prem_ceded) %>%
  group_by(accident_year) %>%
  summarise(
    earned_premium_net = sum(earned_prem_net, na.rm = TRUE),
    earned_premium_direct = sum(earned_prem_dir, na.rm = TRUE),
    earned_premium_ceded = sum(earned_prem_ceded, na.rm = TRUE),
    n_companies_or_groups = n_distinct(grcode),
    .groups = "drop"
  ) %>%
  arrange(accident_year)

make_triangle_matrix <- function(data, value_col) {
  years <- sort(unique(portfolio_full$accident_year))
  lags <- sort(unique(portfolio_full$development_lag))

  wide <- data %>%
    select(accident_year, development_lag, value = {{ value_col }}) %>%
    complete(accident_year = years, development_lag = lags) %>%
    pivot_wider(names_from = development_lag, values_from = value, names_prefix = "dev_") %>%
    arrange(accident_year)

  dev_cols <- paste0("dev_", lags)
  mat <- wide %>% select(all_of(dev_cols)) %>% as.matrix()

  rownames(mat) <- wide$accident_year
  colnames(mat) <- as.character(lags)

  mat
}

observed_paid_triangle <- ChainLadder::as.triangle(
  make_triangle_matrix(observed_triangle_long, cum_paid_loss)
)

observed_incurred_triangle <- ChainLadder::as.triangle(
  make_triangle_matrix(observed_triangle_long, incurred_losses)
)

# Compact report tables.
data_summary <- tibble(
  line_of_business = "Private Passenger Auto",
  valuation_year = valuation_year,
  schedule_p_records = nrow(raw_data),
  companies_or_groups = n_distinct(raw_data$grcode),
  accident_years = paste0(min(raw_data$accident_year), "-", max(raw_data$accident_year)),
  development_lags = paste0(min(raw_data$development_lag), "-", max(raw_data$development_lag)),
  observed_triangle_cells = nrow(observed_triangle_long),
  expected_observed_cells = sum(seq_len(length(unique(raw_data$accident_year)))),
  latest_observed_paid_m = sum(latest_diagonal$latest_observed_paid, na.rm = TRUE) / 1e6,
  latest_observed_incurred_m = sum(latest_diagonal$latest_observed_incurred, na.rm = TRUE) / 1e6,
  earned_premium_net_m = sum(earned_premium_by_ay$earned_premium_net, na.rm = TRUE) / 1e6,
  actual_10yr_paid_m = sum(actual_full_ultimate_by_ay$actual_paid_10yr, na.rm = TRUE) / 1e6,
  actual_10yr_incurred_m = sum(actual_full_ultimate_by_ay$actual_incurred_10yr, na.rm = TRUE) / 1e6
)

latest_diagonal_report <- latest_diagonal %>%
  left_join(earned_premium_by_ay, by = "accident_year") %>%
  transmute(
    accident_year,
    latest_lag = latest_observed_development_lag,
    latest_paid_m = latest_observed_paid / 1e6,
    latest_incurred_m = latest_observed_incurred / 1e6,
    earned_premium_net_m = earned_premium_net / 1e6,
    paid_to_premium = latest_observed_paid / earned_premium_net,
    incurred_to_premium = latest_observed_incurred / earned_premium_net
  )

readr::write_csv(data_summary, "outputs/tables/01_data_summary_report.csv")
readr::write_csv(latest_diagonal_report, "outputs/tables/01_latest_diagonal_report.csv")

# One data figure is enough for the final report; the rest is handled by model-result figures.
years <- sort(unique(portfolio_full$accident_year))
lags <- sort(unique(portfolio_full$development_lag))

paid_triangle_plot <- observed_triangle_long %>%
  select(accident_year, development_lag, value = cum_paid_loss) %>%
  complete(accident_year = years, development_lag = lags) %>%
  ggplot(
    aes(
      x = factor(development_lag, levels = lags),
      y = factor(accident_year, levels = years),
      fill = value
    )
  ) +
  geom_tile(color = "white") +
  scale_fill_continuous(labels = scales::comma, na.value = "white") +
  labs(
    title = "Observed Cumulative Paid Loss Triangle at 2007 Valuation",
    x = "Development lag",
    y = "Accident year",
    fill = "Cumulative value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid = element_blank()
  )

ggsave(
  "outputs/figures/01_observed_paid_triangle_heatmap.png",
  paid_triangle_plot,
  width = 9,
  height = 6,
  dpi = 300
)

saveRDS(
  list(
    valuation_year = valuation_year,
    raw_data = raw_data,
    portfolio_full = portfolio_full,
    observed_triangle_long = observed_triangle_long,
    latest_diagonal = latest_diagonal,
    actual_full_ultimate_by_ay = actual_full_ultimate_by_ay,
    earned_premium_by_ay = earned_premium_by_ay,
    observed_paid_triangle = observed_paid_triangle,
    observed_incurred_triangle = observed_incurred_triangle
  ),
  "data/processed/01_triangle_objects.rds"
)

message("")
message("Step 1 complete.")
message("Report outputs:")
message(" - outputs/tables/01_data_summary_report.csv")
message(" - outputs/tables/01_latest_diagonal_report.csv")
message(" - outputs/figures/01_observed_paid_triangle_heatmap.png")
message("Pipeline object:")
message(" - data/processed/01_triangle_objects.rds")
print(data_summary)
