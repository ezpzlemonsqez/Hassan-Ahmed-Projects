# 04_mack_uncertainty.R
# Mack Chain Ladder uncertainty for the observed paid and incurred triangles.

project_root <- "C:/Users/hassa/Documents/Code/R/General-Insurance-Reserving"
setwd(project_root)

packages <- c("readr", "dplyr", "tidyr", "ggplot2", "scales", "tibble", "ChainLadder", "janitor", "stringr")

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
library(ChainLadder)
library(janitor)
library(stringr)

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

file.remove(list.files("outputs/tables", pattern = "^04_.*\\.csv$", full.names = TRUE))
file.remove(list.files("outputs/figures", pattern = "^04_.*\\.png$", full.names = TRUE))

step1_file <- "data/processed/01_triangle_objects.rds"
if (!file.exists(step1_file)) {
  stop("Missing Step 1 object file. Run scripts/01_prepare_triangles.R first.")
}

obj <- readRDS(step1_file)

valuation_year <- obj$valuation_year
paid_triangle <- obj$observed_paid_triangle
incurred_triangle <- obj$observed_incurred_triangle
actual_full_ultimate_by_ay <- obj$actual_full_ultimate_by_ay

pick_col <- function(data, possible_names, required = TRUE) {
  hits <- intersect(possible_names, names(data))

  if (length(hits) == 0 && required) {
    stop(
      "Could not find required column. Tried: ",
      paste(possible_names, collapse = ", "),
      "\nAvailable columns: ",
      paste(names(data), collapse = ", ")
    )
  }

  if (length(hits) == 0) {
    return(NA_character_)
  }

  hits[1]
}

run_mack <- function(triangle, basis_name) {
  message("Running Mack Chain Ladder for ", basis_name, " triangle...")

  ChainLadder::MackChainLadder(
    triangle,
    est.sigma = "Mack",
    alpha = 1
  )
}

extract_mack_results <- function(mack_model, basis_name, actual_col_name) {
  by_origin <- summary(mack_model)$ByOrigin %>%
    as.data.frame() %>%
    tibble::rownames_to_column("origin_label") %>%
    janitor::clean_names() %>%
    filter(!stringr::str_detect(tolower(origin_label), "total"))

  triangle_years <- suppressWarnings(as.integer(rownames(as.matrix(mack_model$Triangle))))
  origin_years <- suppressWarnings(as.integer(by_origin$origin_label))

  if (
    length(triangle_years) == nrow(by_origin) &&
      (all(is.na(origin_years)) || suppressWarnings(max(origin_years, na.rm = TRUE)) < 1900)
  ) {
    by_origin$accident_year <- triangle_years
  } else {
    by_origin$accident_year <- origin_years
  }

  latest_col <- pick_col(by_origin, c("latest", "latest_observed", "latest_loss"))
  ultimate_col <- pick_col(by_origin, c("ultimate", "ultimates", "ultimate_loss"))
  reserve_col <- pick_col(by_origin, c("ibnr", "reserve", "reserves"))
  se_col <- pick_col(by_origin, c("mack_s_e", "mack_se", "s_e", "se", "std_error"))

  actual_comparison <- actual_full_ultimate_by_ay %>%
    select(accident_year, actual_10yr = all_of(actual_col_name))

  by_origin %>%
    filter(!is.na(accident_year)) %>%
    transmute(
      basis = basis_name,
      accident_year = as.integer(accident_year),
      latest_observed_loss = as.numeric(.data[[latest_col]]),
      mack_ultimate_estimate = as.numeric(.data[[ultimate_col]]),
      mack_reserve_estimate = as.numeric(.data[[reserve_col]]),
      mack_se = as.numeric(.data[[se_col]]),
      lower_95 = mack_reserve_estimate - 1.96 * mack_se,
      upper_95 = mack_reserve_estimate + 1.96 * mack_se
    ) %>%
    left_join(actual_comparison, by = "accident_year") %>%
    mutate(
      ultimate_error = mack_ultimate_estimate - actual_10yr,
      ultimate_error_pct = ultimate_error / actual_10yr
    ) %>%
    arrange(accident_year)
}

paid_mack <- run_mack(paid_triangle, "Paid")
incurred_mack <- run_mack(incurred_triangle, "Incurred")

mack_by_ay <- bind_rows(
  extract_mack_results(paid_mack, "Paid", "actual_paid_10yr"),
  extract_mack_results(incurred_mack, "Incurred", "actual_incurred_10yr")
)

mack_summary <- mack_by_ay %>%
  group_by(basis) %>%
  summarise(
    valuation_year = valuation_year,
    latest_observed_m = sum(latest_observed_loss, na.rm = TRUE) / 1e6,
    mack_ultimate_m = sum(mack_ultimate_estimate, na.rm = TRUE) / 1e6,
    mack_reserve_m = sum(mack_reserve_estimate, na.rm = TRUE) / 1e6,
    mack_se_m = sqrt(sum(mack_se^2, na.rm = TRUE)) / 1e6,
    actual_10yr_m = sum(actual_10yr, na.rm = TRUE) / 1e6,
    ultimate_error_m = (sum(mack_ultimate_estimate, na.rm = TRUE) - sum(actual_10yr, na.rm = TRUE)) / 1e6,
    ultimate_error_pct = ultimate_error_m / actual_10yr_m,
    accident_years = n(),
    .groups = "drop"
  ) %>%
  mutate(
    lower_95_m = mack_reserve_m - 1.96 * mack_se_m,
    upper_95_m = mack_reserve_m + 1.96 * mack_se_m
  ) %>%
  select(
    basis,
    valuation_year,
    latest_observed_m,
    mack_ultimate_m,
    mack_reserve_m,
    mack_se_m,
    lower_95_m,
    upper_95_m,
    actual_10yr_m,
    ultimate_error_m,
    ultimate_error_pct,
    accident_years
  )

readr::write_csv(mack_summary, "outputs/tables/04_mack_summary_report.csv")

mack_ci_plot <- mack_by_ay %>%
  ggplot(
    aes(
      x = factor(accident_year),
      y = mack_reserve_estimate / 1e6,
      ymin = lower_95 / 1e6,
      ymax = upper_95 / 1e6
    )
  ) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_pointrange() +
  facet_wrap(~ basis, scales = "free_y") +
  labs(
    title = "Mack Reserve Estimate with Approximate 95% Interval",
    x = "Accident year",
    y = "Reserve estimate (m)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  "outputs/figures/04_mack_reserve_ci_by_accident_year.png",
  mack_ci_plot,
  width = 9,
  height = 6,
  dpi = 300
)

saveRDS(
  list(
    valuation_year = valuation_year,
    paid_mack = paid_mack,
    incurred_mack = incurred_mack,
    mack_by_ay = mack_by_ay,
    mack_summary = mack_summary
  ),
  "data/processed/04_mack_uncertainty_results.rds"
)

message("")
message("Step 4 complete.")
message("Report outputs:")
message(" - outputs/tables/04_mack_summary_report.csv")
message(" - outputs/figures/04_mack_reserve_ci_by_accident_year.png")
message("Pipeline object:")
message(" - data/processed/04_mack_uncertainty_results.rds")
print(mack_summary)
