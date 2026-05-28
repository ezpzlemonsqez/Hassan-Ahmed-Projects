# Model diagnostics, rolling error monitoring and anomaly alerts.
# Only compact report plots are kept. Diagnostics are still calculated for every
# KPI and summarised in the model-health table.

check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required. Install it with install.packages('", pkg, "').")
  }
}

invisible(lapply(c("dplyr", "tidyr", "readr", "lubridate", "forecast", "ggplot2", "zoo", "scales"), check_package))

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(forecast)
library(ggplot2)
library(zoo)
library(scales)

data_file <- file.path(project_root, "outputs", "data", "synthetic_insurance_operational_kpis.csv")
table_dir <- file.path(project_root, "outputs", "tables")
work_dir <- file.path(project_root, "outputs", "work")
monitoring_plot_dir <- file.path(project_root, "outputs", "plots", "monitoring")
diagnostic_plot_dir <- file.path(project_root, "outputs", "plots", "diagnostics")

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(monitoring_plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(diagnostic_plot_dir, recursive = TRUE, showWarnings = FALSE)

kpi_data <- read_csv(data_file, show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

rolling_results <- read_csv(file.path(work_dir, "rolling_forecast_results.csv"), show_col_types = FALSE) %>%
  mutate(date = as.Date(date))

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

health_rows <- list()
loss_ratio_fit <- NULL

for (i in seq_len(nrow(kpis))) {
  kpi <- kpis$kpi[i]
  label <- kpis$kpi_label[i]
  cat("Running residual diagnostics for", label, "...\n")

  y_ts <- make_ts(kpi_data[[kpi]])
  fit <- forecast::auto.arima(
    y_ts,
    seasonal = TRUE,
    stepwise = TRUE,
    approximation = FALSE,
    biasadj = FALSE
  )

  resid_values <- as.numeric(residuals(fit))
  resid_values <- resid_values[is.finite(resid_values)]
  test_lag <- min(18, max(8, floor(length(resid_values) / 4)))
  fit_df <- length(coef(fit))

  lb_p <- tryCatch(
    Box.test(resid_values, lag = test_lag, type = "Ljung-Box", fitdf = fit_df)$p.value,
    error = function(e) NA_real_
  )

  health_rows[[label]] <- tibble(
    kpi = kpi,
    kpi_label = label,
    selected_model = model_name(fit),
    ljung_box_p_value = lb_p,
    residual_mean = mean(resid_values, na.rm = TRUE),
    residual_sd = sd(resid_values, na.rm = TRUE),
    health_status = case_when(
      is.na(lb_p) ~ "Review manually",
      lb_p < 0.05 ~ "Review residual autocorrelation",
      TRUE ~ "No material residual autocorrelation"
    )
  )

  if (kpi == "loss_ratio") {
    loss_ratio_fit <- fit
  }
}

model_health <- bind_rows(health_rows) %>% arrange(kpi_label)
write_csv(model_health, file.path(table_dir, "model_health_summary.csv"))

# One combined diagnostic figure for the representative KPI. The health table
# above still covers all five KPI models.
png(file.path(diagnostic_plot_dir, "loss_ratio_diagnostics.png"), width = 1100, height = 850)
par(mfrow = c(3, 1), mar = c(4, 5, 3, 2))
loss_resid <- residuals(loss_ratio_fit)
plot(loss_resid, main = "Residuals: Loss Ratio", ylab = "Residual", xlab = "", type = "l")
abline(h = 0, lty = 2)
acf(loss_resid, main = "Residual ACF: Loss Ratio")
pacf(loss_resid, main = "Residual PACF: Loss Ratio")
dev.off()

kpi_long <- kpi_data %>%
  select(date, all_of(kpis$kpi)) %>%
  pivot_longer(-date, names_to = "kpi", values_to = "actual") %>%
  left_join(kpis, by = "kpi")

overview_plot <- ggplot(kpi_long, aes(x = date, y = actual)) +
  geom_line(linewidth = 0.5, colour = "black") +
  facet_wrap(~ kpi_label, scales = "free_y", ncol = 1) +
  labs(
    title = "Synthetic KPI Time Series",
    subtitle = "Monthly insurance and operational KPIs used in the forecasting workflow",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    strip.text = element_text(face = "bold")
  )

ggsave(
  file.path(monitoring_plot_dir, "synthetic_kpi_time_series.png"),
  overview_plot,
  width = 11,
  height = 10,
  dpi = 150
)

rolling_rmse_data <- rolling_results %>%
  group_by(kpi, kpi_label) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(rolling_rmse = sqrt(zoo::rollapplyr(error^2, width = 6, FUN = mean, fill = NA, partial = FALSE))) %>%
  ungroup() %>%
  filter(!is.na(rolling_rmse))

rmse_plot <- ggplot(rolling_rmse_data, aes(x = date, y = rolling_rmse)) +
  geom_line(linewidth = 0.55, colour = "black") +
  facet_wrap(~ kpi_label, scales = "free_y", ncol = 3) +
  labs(
    title = "Rolling Forecast Error Monitoring",
    subtitle = "Six-month rolling RMSE by KPI",
    x = NULL,
    y = "Rolling RMSE"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    strip.text = element_text(face = "bold")
  )

ggsave(
  file.path(monitoring_plot_dir, "rolling_rmse_by_kpi.png"),
  rmse_plot,
  width = 12,
  height = 8,
  dpi = 150
)

# Current snapshot. Volumes/revenue use relative percentage changes. Rate and
# margin KPIs use percentage-point changes because relative changes can exaggerate
# movements in ratios.
latest_date <- max(kpi_data$date)
previous_month <- latest_date %m-% months(1)
previous_year <- latest_date %m-% months(12)

ratio_kpis <- c("lapse_rate", "loss_ratio", "operating_margin")

snapshot_rows <- lapply(seq_len(nrow(kpis)), function(i) {
  kpi <- kpis$kpi[i]
  label <- kpis$kpi_label[i]
  current <- kpi_data %>% filter(date == latest_date) %>% pull(.data[[kpi]])
  last_month <- kpi_data %>% filter(date == previous_month) %>% pull(.data[[kpi]])
  last_year <- kpi_data %>% filter(date == previous_year) %>% pull(.data[[kpi]])

  if (kpi %in% ratio_kpis) {
    mom_change <- current - last_month
    yoy_change <- current - last_year
    basis <- "Percentage-point change"
  } else {
    mom_change <- current / last_month - 1
    yoy_change <- current / last_year - 1
    basis <- "Relative percentage change"
  }

  tibble(
    kpi = kpi,
    kpi_label = label,
    date = latest_date,
    actual = current,
    mom_change = mom_change,
    yoy_change = yoy_change,
    change_basis = basis
  )
})

current_snapshot <- bind_rows(snapshot_rows) %>% arrange(kpi_label)
write_csv(current_snapshot, file.path(table_dir, "kpi_current_snapshot.csv"))

snapshot_plot_data <- current_snapshot %>%
  select(kpi_label, mom_change, yoy_change, change_basis) %>%
  pivot_longer(c(mom_change, yoy_change), names_to = "period", values_to = "change") %>%
  mutate(
    period = recode(period, mom_change = "Month-on-month", yoy_change = "Year-on-year"),
    kpi_label = factor(kpi_label, levels = rev(kpis$kpi_label))
  )

movement_plot <- ggplot(snapshot_plot_data, aes(x = change, y = kpi_label)) +
  geom_col(fill = "grey35") +
  facet_grid(period ~ change_basis, scales = "free_x", space = "free_x") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    title = "Latest KPI Movement Snapshot",
    subtitle = "Volumes/revenue use relative % change; rate and margin KPIs use percentage-point change",
    x = "Change",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1.2, "lines")
  )

ggsave(
  file.path(monitoring_plot_dir, "latest_kpi_movement.png"),
  movement_plot,
  width = 12,
  height = 7,
  dpi = 150
)

# Rolling thresholds use the previous 12 months only. This avoids using the
# current observation inside its own threshold calculation.
threshold_rows <- kpi_long %>%
  group_by(kpi, kpi_label) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    rolling_mean = lag(zoo::rollapplyr(actual, 12, mean, fill = NA, partial = FALSE)),
    rolling_sd = lag(zoo::rollapplyr(actual, 12, sd, fill = NA, partial = FALSE)),
    lower_threshold = rolling_mean - 2 * rolling_sd,
    upper_threshold = rolling_mean + 2 * rolling_sd,
    threshold_flag = case_when(
      is.na(lower_threshold) | is.na(upper_threshold) ~ NA_character_,
      actual < lower_threshold ~ "Below rolling threshold",
      actual > upper_threshold ~ "Above rolling threshold",
      TRUE ~ NA_character_
    )
  ) %>%
  ungroup()

loss_threshold <- threshold_rows %>% filter(kpi == "loss_ratio")

threshold_plot <- ggplot(loss_threshold, aes(x = date)) +
  geom_ribbon(aes(ymin = lower_threshold, ymax = upper_threshold), fill = "grey70", alpha = 0.45) +
  geom_line(aes(y = rolling_mean), linewidth = 0.45, linetype = "dashed", colour = "black") +
  geom_line(aes(y = actual), linewidth = 0.55, colour = "black") +
  geom_point(
    data = loss_threshold %>% filter(!is.na(threshold_flag)),
    aes(y = actual),
    size = 2,
    colour = "black"
  ) +
  labs(
    title = "Loss Ratio Monitoring Thresholds",
    subtitle = "Prior 12-month rolling mean with +/- 2 standard deviation bands",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 18))

ggsave(
  file.path(monitoring_plot_dir, "loss_ratio_monitoring_thresholds.png"),
  threshold_plot,
  width = 11,
  height = 6,
  dpi = 150
)

threshold_alerts <- threshold_rows %>%
  filter(!is.na(threshold_flag)) %>%
  transmute(
    date,
    kpi,
    kpi_label,
    alert_type = "Rolling threshold breach",
    alert_detail = threshold_flag,
    actual,
    reference_value = rolling_mean,
    lower_threshold,
    upper_threshold
  )

forecast_alerts <- rolling_results %>%
  filter(actual < lower_95 | actual > upper_95) %>%
  transmute(
    date,
    kpi,
    kpi_label,
    alert_type = "Forecast interval breach",
    alert_detail = "Actual fell outside the one-step-ahead 95% prediction interval",
    actual,
    reference_value = forecast,
    lower_threshold = lower_95,
    upper_threshold = upper_95
  )

alerts <- bind_rows(threshold_alerts, forecast_alerts) %>%
  arrange(desc(date), kpi_label, alert_type)

write_csv(alerts, file.path(table_dir, "anomaly_alerts.csv"))
cat("Model health diagnostics and KPI monitoring outputs created.\n")
