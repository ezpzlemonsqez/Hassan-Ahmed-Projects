# Forecasting models and rolling evaluation.
# The code keeps the report compact: detailed rolling rows are written to a
# temporary work folder, while only summary tables and one combined forecast
# chart are kept in the final outputs.

check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required. Install it with install.packages('", pkg, "').")
  }
}

invisible(lapply(c("dplyr", "tidyr", "readr", "lubridate", "forecast", "ggplot2", "scales"), check_package))

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(forecast)
library(ggplot2)
library(scales)

data_file <- file.path(project_root, "outputs", "data", "synthetic_insurance_operational_kpis.csv")
plot_dir <- file.path(project_root, "outputs", "plots", "forecast")
table_dir <- file.path(project_root, "outputs", "tables")
work_dir <- file.path(project_root, "outputs", "work")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

cat("Loading synthetic KPI data...\n")
kpi_data <- read_csv(data_file, show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

kpis <- tibble::tribble(
  ~kpi,                 ~kpi_label,
  "claims_count",       "Claims Count",
  "earned_premium",     "Earned Premium / Revenue",
  "lapse_rate",         "Lapse Rate",
  "loss_ratio",         "Loss Ratio",
  "operating_margin",   "Operating Margin Proxy"
)

make_ts <- function(x) {
  ts(x, start = c(year(min(kpi_data$date)), month(min(kpi_data$date))), frequency = 12)
}

model_name <- function(fit) {
  ord <- forecast::arimaorder(fit)
  if (length(ord) >= 7) {
    paste0("ARIMA(", ord[1], ",", ord[2], ",", ord[3], ")(", ord[4], ",", ord[5], ",", ord[6], ")[", ord[7], "]")
  } else {
    paste0("ARIMA(", ord[1], ",", ord[2], ",", ord[3], ")")
  }
}

future_dates <- seq.Date(from = seq.Date(max(kpi_data$date), by = "month", length.out = 2)[2], by = "month", length.out = 12)

model_rows <- list()
forecast_rows <- list()
rolling_rows <- list()

for (i in seq_len(nrow(kpis))) {
  kpi <- kpis$kpi[i]
  label <- kpis$kpi_label[i]
  cat("Fitting ARIMA model for", label, "...\n")

  y <- kpi_data[[kpi]]
  y_ts <- make_ts(y)

  fit <- forecast::auto.arima(
    y_ts,
    seasonal = TRUE,
    stepwise = TRUE,
    approximation = FALSE,
    biasadj = FALSE
  )

  fc <- forecast::forecast(fit, h = 12, level = 95)

  model_rows[[label]] <- tibble(
    kpi = kpi,
    kpi_label = label,
    selected_model = model_name(fit),
    aic = as.numeric(AIC(fit)),
    bic = as.numeric(BIC(fit)),
    sigma2 = as.numeric(fit$sigma2)
  )

  forecast_rows[[label]] <- tibble(
    date = future_dates,
    kpi = kpi,
    kpi_label = label,
    forecast = as.numeric(fc$mean),
    lower_95 = as.numeric(fc$lower[, 1]),
    upper_95 = as.numeric(fc$upper[, 1])
  )

  cat("  Rolling evaluation for", label, "using fixed selected ARIMA form\n")
  first_test <- 49
  n_roll <- length(y) - first_test + 1

  for (j in seq(from = first_test, to = length(y))) {
    train_y <- y[1:(j - 1)]
    test_y <- y[j]

    train_ts <- ts(train_y, start = c(year(min(kpi_data$date)), month(min(kpi_data$date))), frequency = 12)
    refit <- tryCatch(
      forecast::Arima(train_ts, model = fit),
      error = function(e) forecast::auto.arima(train_ts, seasonal = TRUE, stepwise = TRUE, approximation = FALSE)
    )

    one_step <- forecast::forecast(refit, h = 1, level = 95)
    pred <- as.numeric(one_step$mean[1])
    lo <- as.numeric(one_step$lower[1, 1])
    hi <- as.numeric(one_step$upper[1, 1])

    rolling_rows[[paste(label, j, sep = "_")]] <- tibble(
      date = kpi_data$date[j],
      kpi = kpi,
      kpi_label = label,
      actual = test_y,
      forecast = pred,
      lower_95 = lo,
      upper_95 = hi,
      error = test_y - pred,
      abs_error = abs(test_y - pred),
      pct_error = ifelse(abs(test_y) > 1e-12, abs((test_y - pred) / test_y), NA_real_),
      covered_95 = test_y >= lo & test_y <= hi
    )

    done <- j - first_test + 1
    if (done %% 10 == 0 || done == n_roll) {
      cat("    completed", done, "of", n_roll, "rolling forecasts\n")
    }
  }

  cat("Completed ARIMA outputs for", label, ".\n")
}

model_selection <- bind_rows(model_rows) %>% arrange(kpi_label)
forecast_12 <- bind_rows(forecast_rows) %>% arrange(kpi_label, date)
rolling_results <- bind_rows(rolling_rows) %>% arrange(kpi_label, date)

rolling_metrics <- rolling_results %>%
  group_by(kpi, kpi_label) %>%
  summarise(
    observations = n(),
    mae = mean(abs_error, na.rm = TRUE),
    rmse = sqrt(mean(error^2, na.rm = TRUE)),
    mape = mean(pct_error, na.rm = TRUE),
    bias = mean(error, na.rm = TRUE),
    interval_95_coverage = mean(covered_95, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(kpi_label)

write_csv(model_selection, file.path(table_dir, "model_selection_summary.csv"))
write_csv(rolling_metrics, file.path(table_dir, "rolling_forecast_metrics.csv"))
write_csv(rolling_results, file.path(work_dir, "rolling_forecast_results.csv"))
write_csv(forecast_12, file.path(work_dir, "forecast_12_month.csv"))

cat("Fitting AIC-selected regression models...\n")
reg_data <- kpi_data %>%
  arrange(date) %>%
  mutate(
    month_factor = factor(month),
    lag_claims_count = lag(claims_count),
    lag_earned_premium = lag(earned_premium),
    lag_lapse_rate = lag(lapse_rate),
    lag_loss_ratio = lag(loss_ratio),
    lag_operating_margin = lag(operating_margin)
  ) %>%
  tidyr::drop_na()

reg_specs <- list(
  "Claims Count" = claims_count ~ month_factor + policy_count + economic_stress + inflation_index + lag_claims_count,
  "Earned Premium / Revenue" = earned_premium ~ month_factor + policy_count + economic_stress + inflation_index + lag_earned_premium,
  "Lapse Rate" = lapse_rate ~ month_factor + economic_stress + inflation_index + lag_lapse_rate,
  "Loss Ratio" = loss_ratio ~ month_factor + economic_stress + inflation_index + lag_claims_count + lag_loss_ratio,
  # Important: this avoids same-period loss_ratio and expense_ratio, which would
  # almost reproduce operating_margin by construction.
  "Operating Margin Proxy" = operating_margin ~ month_factor + economic_stress + inflation_index + lag_loss_ratio + lag_operating_margin
)

reg_rows <- list()
for (label in names(reg_specs)) {
  full_fit <- lm(reg_specs[[label]], data = reg_data)
  selected_fit <- suppressWarnings(step(full_fit, direction = "backward", trace = 0))
  s <- summary(selected_fit)

  reg_rows[[label]] <- tibble(
    kpi_label = label,
    selected_formula = paste(deparse(formula(selected_fit)), collapse = " "),
    aic = AIC(selected_fit),
    adjusted_r_squared = unname(s$adj.r.squared),
    residual_standard_error = unname(s$sigma)
  )
}

regression_summary <- bind_rows(reg_rows) %>% arrange(kpi_label)
write_csv(regression_summary, file.path(table_dir, "regression_model_summary.csv"))

history_long <- kpi_data %>%
  select(date, all_of(kpis$kpi)) %>%
  pivot_longer(-date, names_to = "kpi", values_to = "actual") %>%
  left_join(kpis, by = "kpi")

forecast_plot_data <- forecast_12 %>%
  select(date, kpi, kpi_label, forecast, lower_95, upper_95)

forecast_summary_plot <- ggplot() +
  geom_line(data = history_long, aes(x = date, y = actual), linewidth = 0.45, colour = "black") +
  geom_ribbon(
    data = forecast_plot_data,
    aes(x = date, ymin = lower_95, ymax = upper_95),
    fill = "grey70",
    alpha = 0.45
  ) +
  geom_line(
    data = forecast_plot_data,
    aes(x = date, y = forecast),
    linewidth = 0.45,
    linetype = "dashed",
    colour = "black"
  ) +
  facet_wrap(~ kpi_label, scales = "free_y", ncol = 2) +
  labs(
    title = "Twelve-Month Forecast Summary",
    subtitle = "Historical values with ARIMA point forecasts and 95% prediction intervals",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12),
    panel.grid.minor = element_line(linewidth = 0.2),
    strip.text = element_text(face = "bold")
  )

ggsave(
  filename = file.path(plot_dir, "forecast_summary.png"),
  plot = forecast_summary_plot,
  width = 12,
  height = 9,
  dpi = 150
)

cat("Forecasting models, rolling metrics and forecast plot created.\n")
