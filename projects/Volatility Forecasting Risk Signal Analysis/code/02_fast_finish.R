
############################################################
# Fast finishing script for VIX project
# Run after the rolling evaluation has already completed.
############################################################

library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(forecast)
library(strucchange)

dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)

vix <- read_csv("data/processed/vix_clean.csv", show_col_types = FALSE)
model_df <- read_csv("data/processed/vix_model_features.csv", show_col_types = FALSE)
rolling_results <- read_csv("outputs/tables/rolling_forecasts.csv", show_col_types = FALSE)

vix$date <- as.Date(vix$date)
model_df$date <- as.Date(model_df$date)
rolling_results$date <- as.Date(rolling_results$date)

cat("Loaded processed data.\n")

############################################################
# 1. Rolling forecast comparison plot
############################################################

plot_df <- rolling_results %>%
  filter(date >= max(date) - years(1))

p_forecasts <- ggplot(plot_df, aes(x = date)) +
  geom_line(aes(y = actual_vix), linewidth = 0.5) +
  geom_line(aes(y = arima_mean_vix), linetype = "dashed", linewidth = 0.45, na.rm = TRUE) +
  geom_line(aes(y = reg_mean_vix), linetype = "dotted", linewidth = 0.45, na.rm = TRUE) +
  labs(
    title = "Rolling One-Day-Ahead VIX Forecasts",
    subtitle = "Actual VIX vs ARIMA and AIC-selected regression forecasts",
    x = "Date",
    y = "VIX"
  ) +
  theme_minimal()

ggsave("outputs/plots/rolling_forecast_comparison.png", p_forecasts, width = 10, height = 5, dpi = 300)

cat("Saved rolling forecast plot.\n")

############################################################
# 2. Fast final ARIMA forecast
############################################################

# Use recent 5 years for final forecasting to avoid slow full-sample exhaustive search.
recent_vix <- vix %>%
  filter(date >= max(date) - years(5))

final_arima <- auto.arima(
  recent_vix$log_vix,
  seasonal = FALSE,
  ic = "aicc",
  stepwise = TRUE,
  approximation = TRUE,
  max.p = 3,
  max.q = 3,
  max.d = 1
)

saveRDS(final_arima, "outputs/models/final_arima_model_fast.rds")

final_arima_summary <- capture.output(summary(final_arima))
writeLines(final_arima_summary, "outputs/tables/final_arima_summary.txt")

next_business_days <- function(last_date, n) {
  candidate_dates <- seq.Date(last_date + days(1), by = "day", length.out = n * 3)
  business_dates <- candidate_dates[!(weekdays(candidate_dates) %in% c("Saturday", "Sunday"))]
  head(business_dates, n)
}

h <- 20
future_dates <- next_business_days(max(vix$date), h)

arima_fc <- forecast(final_arima, h = h, level = c(80, 95))

final_forecast <- tibble(
  date = future_dates,
  mean_log = as.numeric(arima_fc$mean),
  lower80_log = as.numeric(arima_fc$lower[, "80%"]),
  upper80_log = as.numeric(arima_fc$upper[, "80%"]),
  lower95_log = as.numeric(arima_fc$lower[, "95%"]),
  upper95_log = as.numeric(arima_fc$upper[, "95%"])
) %>%
  mutate(
    mean_vix = exp(mean_log),
    lower80_vix = exp(lower80_log),
    upper80_vix = exp(upper80_log),
    lower95_vix = exp(lower95_log),
    upper95_vix = exp(upper95_log)
  )

write_csv(final_forecast, "outputs/tables/final_arima_20_day_forecast.csv")

p_final_fc <- ggplot() +
  geom_line(
    data = vix %>% filter(date >= max(date) - years(2)),
    aes(x = date, y = vix),
    linewidth = 0.45
  ) +
  geom_line(
    data = final_forecast,
    aes(x = date, y = mean_vix),
    linetype = "dashed",
    linewidth = 0.55
  ) +
  geom_ribbon(
    data = final_forecast,
    aes(x = date, ymin = lower95_vix, ymax = upper95_vix),
    alpha = 0.2
  ) +
  labs(
    title = "Final 20-Trading-Day VIX Forecast",
    subtitle = "Fast ARIMA forecast using recent five-year log-VIX history with 95% prediction interval",
    x = "Date",
    y = "VIX"
  ) +
  theme_minimal()

ggsave("outputs/plots/final_arima_20_day_forecast.png", p_final_fc, width = 10, height = 5, dpi = 300)

cat("Saved final ARIMA forecast.\n")

############################################################
# 3. Final regression diagnostics
############################################################

selected_formula <- log_vix ~ log_lag1 + ma5_lag1 + ma21_lag1 + ma63_lag1 + stress_lag1 + elevated_lag1

final_reg_data <- model_df %>%
  dplyr::select(
    log_vix,
    log_lag1,
    ma5_lag1,
    ma21_lag1,
    ma63_lag1,
    stress_lag1,
    elevated_lag1
  ) %>%
  tidyr::drop_na()

final_reg <- lm(selected_formula, data = final_reg_data)

saveRDS(final_reg, "outputs/models/final_regression_model.rds")

final_reg_summary <- capture.output(summary(final_reg))
writeLines(final_reg_summary, "outputs/tables/final_regression_summary.txt")

arima_resid <- residuals(final_arima)
reg_resid <- residuals(final_reg)

ljung_arima <- Box.test(
  arima_resid,
  lag = 20,
  type = "Ljung-Box",
  fitdf = length(final_arima$coef)
)

ljung_reg <- Box.test(
  reg_resid,
  lag = 20,
  type = "Ljung-Box",
  fitdf = length(coef(final_reg)) - 1
)

diagnostics_table <- tibble(
  model = c("ARIMA", "Regression"),
  test = "Ljung-Box",
  lag = 20,
  statistic = c(as.numeric(ljung_arima$statistic), as.numeric(ljung_reg$statistic)),
  p_value = c(ljung_arima$p.value, ljung_reg$p.value),
  interpretation = case_when(
    p_value < 0.05 ~ "Residual autocorrelation remains; model may miss time-series structure.",
    TRUE ~ "No strong evidence of residual autocorrelation at 5% level."
  )
)

write_csv(diagnostics_table, "outputs/tables/ljung_box_tests.csv")

png("outputs/plots/arima_residual_acf.png", width = 1000, height = 600)
acf(arima_resid, main = "ARIMA Residual ACF")
dev.off()

png("outputs/plots/arima_residual_pacf.png", width = 1000, height = 600)
pacf(arima_resid, main = "ARIMA Residual PACF")
dev.off()

png("outputs/plots/regression_residual_acf.png", width = 1000, height = 600)
acf(reg_resid, main = "Regression Residual ACF")
dev.off()

png("outputs/plots/regression_residual_pacf.png", width = 1000, height = 600)
pacf(reg_resid, main = "Regression Residual PACF")
dev.off()

cat("Saved residual diagnostics.\n")

############################################################
# 4. Fast regime-shift analysis using monthly log-VIX
############################################################

regime_input <- vix %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarise(
    log_vix = mean(log_vix, na.rm = TRUE),
    vix = mean(vix, na.rm = TRUE),
    .groups = "drop"
  )

bp_full <- breakpoints(log_vix ~ 1, data = regime_input, h = 0.10)
bic_values <- BIC(bp_full)
optimal_breaks <- which.min(bic_values) - 1

if (optimal_breaks > 0) {
  bp_opt <- breakpoints(bp_full, breaks = optimal_breaks)
  break_indices <- bp_opt$breakpoints

  regime_breaks <- tibble(
    break_number = seq_along(break_indices),
    break_index = break_indices,
    break_date = regime_input$month[break_indices],
    break_vix = regime_input$vix[break_indices]
  )
} else {
  regime_breaks <- tibble(
    break_number = integer(),
    break_index = integer(),
    break_date = as.Date(character()),
    break_vix = numeric()
  )
}

write_csv(regime_breaks, "outputs/tables/regime_breaks.csv")

p_breaks <- ggplot(regime_input, aes(x = month, y = log_vix)) +
  geom_line(linewidth = 0.35) +
  labs(
    title = "Structural Break Diagnostic for Monthly Log-VIX",
    subtitle = "Vertical lines indicate estimated regime breaks selected by BIC",
    x = "Date",
    y = "Monthly Average Log VIX"
  ) +
  theme_minimal()

if (nrow(regime_breaks) > 0) {
  p_breaks <- p_breaks +
    geom_vline(
      data = regime_breaks,
      aes(xintercept = break_date),
      linetype = "dashed"
    )
}

ggsave("outputs/plots/structural_breaks_log_vix.png", p_breaks, width = 10, height = 5, dpi = 300)

cat("Saved regime-shift analysis.\n")

############################################################
# 5. Final summary
############################################################

summary_lines <- c(
  "Volatility Forecasting & Risk Signal Analysis | VIX Index | R",
  "",
  paste("Data start:", min(vix$date)),
  paste("Data end:", max(vix$date)),
  paste("Clean observations:", nrow(vix)),
  "",
  "Models:",
  "- Rolling ARIMA and AIC-selected regression models for one-day-ahead log-VIX forecasting.",
  "- Final ARIMA forecast uses recent five-year data for fast, stable portfolio output generation.",
  "",
  "Evaluation:",
  "- Rolling one-day-ahead forecasts already completed in the main script.",
  "- Model comparison metrics saved in outputs/tables/model_comparison.csv.",
  "",
  "Diagnostics:",
  "- Residual ACF/PACF plots.",
  "- Ljung-Box residual tests.",
  "- Monthly log-VIX structural break diagnostics.",
  "",
  "Important interpretation:",
  "This project forecasts the VIX implied volatility index, not realised volatility directly."
)

writeLines(summary_lines, "outputs/project_summary.txt")

cat("\nFast finishing script complete.\n")

