
############################################################
# Final report outputs
# VIX forecasting and risk signal project
#
# Run this after:
# source("scripts/fast_finish.R")
#
# This creates the figures and tables used in the report.
############################################################

library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(forecast)

############################################################
# 1. Set up folders
############################################################

unlink("outputs/final_report", recursive = TRUE, force = TRUE)

dir.create("outputs/final_report", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/final_report/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/final_report/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/final_report/text", recursive = TRUE, showWarnings = FALSE)

############################################################
# 2. Load data
############################################################

required_files <- c(
  "data/processed/vix_clean.csv",
  "outputs/tables/rolling_forecasts.csv",
  "outputs/tables/model_comparison.csv",
  "outputs/tables/vix_risk_signals.csv",
  "outputs/tables/ljung_box_tests.csv",
  "outputs/tables/final_arima_20_day_forecast.csv"
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    paste(
      "Some files are missing. Run source(\"scripts/fast_finish.R\") first. Missing:",
      paste(missing_files, collapse = ", ")
    )
  )
}

vix <- read_csv("data/processed/vix_clean.csv", show_col_types = FALSE)
rolling_results <- read_csv("outputs/tables/rolling_forecasts.csv", show_col_types = FALSE)
model_comparison <- read_csv("outputs/tables/model_comparison.csv", show_col_types = FALSE)
signals <- read_csv("outputs/tables/vix_risk_signals.csv", show_col_types = FALSE)
ljung <- read_csv("outputs/tables/ljung_box_tests.csv", show_col_types = FALSE)
final_forecast <- read_csv("outputs/tables/final_arima_20_day_forecast.csv", show_col_types = FALSE)

vix$date <- as.Date(vix$date)
rolling_results$date <- as.Date(rolling_results$date)
signals$date <- as.Date(signals$date)
final_forecast$date <- as.Date(final_forecast$date)

# Keep only rows where the rolling thresholds are available.
signals_clean <- signals %>%
  filter(
    !is.na(vix_q75_252_lag1),
    !is.na(vix_q90_252_lag1)
  )

############################################################
# 3. Tables
############################################################

data_summary <- tibble(
  metric = c(
    "Data start",
    "Data end",
    "Clean observations",
    "Rolling forecasts",
    "Risk signal observations"
  ),
  value = c(
    as.character(min(vix$date)),
    as.character(max(vix$date)),
    as.character(nrow(vix)),
    as.character(nrow(rolling_results)),
    as.character(nrow(signals_clean))
  )
)

write_csv(data_summary, "outputs/final_report/tables/table_1_data_summary.csv")

model_comparison_report <- model_comparison %>%
  mutate(
    model = recode(model, arima = "ARIMA", reg = "Regression"),
    rmse_log = round(rmse_log, 4),
    mae_log = round(mae_log, 4),
    rmse_vix = round(rmse_vix, 4),
    mae_vix = round(mae_vix, 4),
    mape_vix = round(mape_vix, 2),
    coverage_95_log_pct = round(100 * coverage_95_log, 2),
    coverage_95_vix_pct = round(100 * coverage_95_vix, 2)
  ) %>%
  dplyr::select(
    model,
    rmse_log,
    mae_log,
    rmse_vix,
    mae_vix,
    mape_vix,
    coverage_95_log_pct,
    coverage_95_vix_pct,
    n_forecasts
  )

write_csv(model_comparison_report, "outputs/final_report/tables/table_2_model_comparison.csv")

regime_summary <- signals_clean %>%
  group_by(risk_signal) %>%
  summarise(
    observations = n(),
    pct_of_days = round(100 * n() / nrow(signals_clean), 2),
    avg_vix = round(mean(vix, na.rm = TRUE), 2),
    median_vix = round(median(vix, na.rm = TRUE), 2),
    max_vix = round(max(vix, na.rm = TRUE), 2),
    avg_z_score = round(mean(z_252, na.rm = TRUE), 2),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_vix))

write_csv(regime_summary, "outputs/final_report/tables/table_3_regime_summary.csv")

regimes_by_year <- signals_clean %>%
  filter(year(date) >= 2022, year(date) <= 2025) %>%
  group_by(year = year(date), risk_signal) %>%
  summarise(
    days = n(),
    avg_vix = round(mean(vix, na.rm = TRUE), 2),
    max_vix = round(max(vix, na.rm = TRUE), 2),
    .groups = "drop"
  )

write_csv(regimes_by_year, "outputs/final_report/tables/table_4_risk_regimes_by_year.csv")

ljung_report <- ljung %>%
  mutate(
    statistic = round(statistic, 4),
    p_value = round(p_value, 4)
  )

write_csv(ljung_report, "outputs/final_report/tables/table_5_ljung_box_tests.csv")

forecast_report <- final_forecast %>%
  dplyr::select(
    date,
    mean_vix,
    lower80_vix,
    upper80_vix,
    lower95_vix,
    upper95_vix
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))

write_csv(forecast_report, "outputs/final_report/tables/table_6_final_20_day_forecast.csv")

latest_signal <- signals_clean %>%
  slice_tail(n = 1) %>%
  dplyr::select(
    date,
    vix,
    risk_signal,
    z_252,
    vix_q75_252_lag1,
    vix_q90_252_lag1
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))

write_csv(latest_signal, "outputs/final_report/tables/table_7_latest_risk_signal.csv")

############################################################
# 4. Figure 1: VIX and rolling thresholds
############################################################

fig1 <- ggplot(signals_clean, aes(x = date)) +
  geom_line(aes(y = vix, linetype = "VIX"), linewidth = 0.35) +
  geom_line(
    aes(y = vix_q75_252_lag1, linetype = "Elevated threshold"),
    linewidth = 0.35,
    na.rm = TRUE
  ) +
  geom_line(
    aes(y = vix_q90_252_lag1, linetype = "Stress threshold"),
    linewidth = 0.35,
    na.rm = TRUE
  ) +
  scale_linetype_manual(
    values = c(
      "VIX" = "solid",
      "Elevated threshold" = "dashed",
      "Stress threshold" = "dotted"
    )
  ) +
  labs(
    title = "VIX with Rolling Risk Thresholds",
    subtitle = "Thresholds are based on the previous year of VIX data",
    x = "Date",
    y = "VIX",
    linetype = "Series"
  ) +
  theme_minimal()

ggsave(
  "outputs/final_report/figures/figure_1_vix_with_rolling_thresholds.png",
  fig1,
  width = 10,
  height = 5,
  dpi = 300
)

############################################################
# 5. Figure 2: Rolling forecast comparison
############################################################

forecast_long <- rolling_results %>%
  filter(date >= max(date) - years(1)) %>%
  dplyr::select(date, actual_vix, arima_mean_vix, reg_mean_vix) %>%
  pivot_longer(
    cols = c(actual_vix, arima_mean_vix, reg_mean_vix),
    names_to = "series",
    values_to = "vix"
  ) %>%
  mutate(
    series = recode(
      series,
      actual_vix = "Actual VIX",
      arima_mean_vix = "ARIMA forecast",
      reg_mean_vix = "Regression forecast"
    )
  )

fig2 <- ggplot(forecast_long, aes(x = date, y = vix, linetype = series)) +
  geom_line(linewidth = 0.5, na.rm = TRUE) +
  scale_linetype_manual(
    values = c(
      "Actual VIX" = "solid",
      "ARIMA forecast" = "dashed",
      "Regression forecast" = "dotted"
    )
  ) +
  labs(
    title = "One-Day-Ahead VIX Forecasts",
    subtitle = "Actual VIX compared with ARIMA and regression forecasts",
    x = "Date",
    y = "VIX",
    linetype = "Series"
  ) +
  theme_minimal()

ggsave(
  "outputs/final_report/figures/figure_2_rolling_forecast_comparison.png",
  fig2,
  width = 10,
  height = 5,
  dpi = 300
)

############################################################
# 6. Figure 3: 20-day forecast
############################################################

fig3 <- ggplot() +
  geom_line(
    data = vix %>% filter(date >= max(date) - years(2)),
    aes(x = date, y = vix),
    linewidth = 0.45
  ) +
  geom_ribbon(
    data = final_forecast,
    aes(x = date, ymin = lower95_vix, ymax = upper95_vix),
    alpha = 0.2
  ) +
  geom_line(
    data = final_forecast,
    aes(x = date, y = mean_vix),
    linetype = "dashed",
    linewidth = 0.55
  ) +
  labs(
    title = "20-Day VIX Forecast",
    subtitle = "ARIMA forecast with a 95% prediction range",
    x = "Date",
    y = "VIX"
  ) +
  theme_minimal()

ggsave(
  "outputs/final_report/figures/figure_3_final_20_day_arima_forecast.png",
  fig3,
  width = 10,
  height = 5,
  dpi = 300
)

############################################################
# 7. Figure 4: Risk regimes by year
############################################################

fig4 <- regimes_by_year %>%
  ggplot(aes(x = factor(year), y = days, fill = risk_signal)) +
  geom_col() +
  labs(
    title = "VIX Risk Regimes by Year",
    subtitle = "Full calendar years from 2022 to 2025",
    x = "Year",
    y = "Number of trading days",
    fill = "Risk signal"
  ) +
  theme_minimal()

ggsave(
  "outputs/final_report/figures/figure_4_risk_regime_frequency_by_year.png",
  fig4,
  width = 10,
  height = 5,
  dpi = 300
)

############################################################
# 8. Figure 5: Overall regime distribution
############################################################

fig5 <- regime_summary %>%
  mutate(
    risk_signal = factor(risk_signal, levels = c("Normal", "Elevated", "Stress"))
  ) %>%
  ggplot(aes(x = risk_signal, y = pct_of_days, fill = risk_signal)) +
  geom_col() +
  labs(
    title = "Overall VIX Risk Regime Split",
    subtitle = "Share of days classified as Normal, Elevated or Stress",
    x = "Risk regime",
    y = "Share of days (%)",
    fill = "Risk regime"
  ) +
  theme_minimal()

ggsave(
  "outputs/final_report/figures/figure_5_risk_regime_distribution.png",
  fig5,
  width = 8,
  height = 5,
  dpi = 300
)

############################################################
# 9. Figure 6: ARIMA residual checks
############################################################

if (file.exists("outputs/models/final_arima_model_fast.rds")) {
  final_arima <- readRDS("outputs/models/final_arima_model_fast.rds")
} else if (file.exists("outputs/models/final_arima_model.rds")) {
  final_arima <- readRDS("outputs/models/final_arima_model.rds")
} else {
  stop("No saved ARIMA model found. Run source(\"scripts/fast_finish.R\") first.")
}

arima_resid <- residuals(final_arima)

png(
  "outputs/final_report/figures/figure_6_arima_residual_acf_pacf.png",
  width = 1400,
  height = 600
)

par(mfrow = c(1, 2), mar = c(4, 4, 4, 2))

acf(
  arima_resid,
  lag.max = 40,
  main = "ACF of ARIMA Residuals",
  xlab = "Lag",
  ylab = "ACF"
)

pacf(
  arima_resid,
  lag.max = 40,
  main = "PACF of ARIMA Residuals",
  xlab = "Lag",
  ylab = "PACF"
)

dev.off()

############################################################
# 10. Plain results summary
############################################################

best_rmse_model <- model_comparison_report %>%
  arrange(rmse_vix) %>%
  slice(1)

arima_ljung <- ljung_report %>% filter(model == "ARIMA")
reg_ljung <- ljung_report %>% filter(model == "Regression")

summary_text <- c(
  "Volatility Forecasting and Risk Signal Analysis",
  "",
  "Data",
  paste0("The project uses daily VIX data from ", min(vix$date), " to ", max(vix$date), "."),
  paste0("After cleaning, the sample contains ", nrow(vix), " observations."),
  paste0("The rolling backtest contains ", nrow(rolling_results), " one-day-ahead forecasts."),
  "",
  "Model results",
  paste0("ARIMA RMSE on VIX levels: ", model_comparison_report$rmse_vix[model_comparison_report$model == "ARIMA"], "."),
  paste0("Regression RMSE on VIX levels: ", model_comparison_report$rmse_vix[model_comparison_report$model == "Regression"], "."),
  paste0("The lowest RMSE came from the ", best_rmse_model$model, " model."),
  paste0("ARIMA 95% interval coverage: ", model_comparison_report$coverage_95_vix_pct[model_comparison_report$model == "ARIMA"], "%."),
  paste0("Regression 95% interval coverage: ", model_comparison_report$coverage_95_vix_pct[model_comparison_report$model == "Regression"], "%."),
  "",
  "Model checks",
  paste0("ARIMA Ljung-Box p-value: ", arima_ljung$p_value, "."),
  paste0("Regression Ljung-Box p-value: ", reg_ljung$p_value, "."),
  "The regression model was slightly more accurate, but the ARIMA residuals looked cleaner.",
  "",
  "Risk signals",
  paste0(
    "Normal days: ",
    regime_summary$observations[regime_summary$risk_signal == "Normal"],
    " observations, ",
    regime_summary$pct_of_days[regime_summary$risk_signal == "Normal"],
    "% of the sample, average VIX ",
    regime_summary$avg_vix[regime_summary$risk_signal == "Normal"],
    "."
  ),
  paste0(
    "Elevated days: ",
    regime_summary$observations[regime_summary$risk_signal == "Elevated"],
    " observations, ",
    regime_summary$pct_of_days[regime_summary$risk_signal == "Elevated"],
    "% of the sample, average VIX ",
    regime_summary$avg_vix[regime_summary$risk_signal == "Elevated"],
    "."
  ),
  paste0(
    "Stress days: ",
    regime_summary$observations[regime_summary$risk_signal == "Stress"],
    " observations, ",
    regime_summary$pct_of_days[regime_summary$risk_signal == "Stress"],
    "% of the sample, average VIX ",
    regime_summary$avg_vix[regime_summary$risk_signal == "Stress"],
    ", maximum VIX ",
    regime_summary$max_vix[regime_summary$risk_signal == "Stress"],
    "."
  ),
  "",
  "Note",
  "This project forecasts the VIX index itself. VIX is an implied volatility measure, so the project is not directly forecasting realised volatility.",
  "The risk signals are based only on past data, so the thresholds avoid look-ahead bias."
)

writeLines(summary_text, "outputs/final_report/text/final_results_summary.txt")

############################################################
# 11. Manifest
############################################################

manifest <- tibble(
  item = c(
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 4",
    "Figure 5",
    "Figure 6",
    "Table 1",
    "Table 2",
    "Table 3",
    "Table 4",
    "Table 5",
    "Table 6",
    "Table 7",
    "Text summary"
  ),
  description = c(
    "VIX with rolling risk thresholds",
    "One-day-ahead forecast comparison",
    "20-day ARIMA forecast",
    "Risk regimes by year",
    "Overall regime split",
    "ARIMA residual ACF and PACF",
    "Data summary",
    "Model comparison",
    "Risk regime summary",
    "Risk regimes by year",
    "Ljung-Box tests",
    "20-day forecast table",
    "Latest risk signal",
    "Plain results summary"
  ),
  path = c(
    "outputs/final_report/figures/figure_1_vix_with_rolling_thresholds.png",
    "outputs/final_report/figures/figure_2_rolling_forecast_comparison.png",
    "outputs/final_report/figures/figure_3_final_20_day_arima_forecast.png",
    "outputs/final_report/figures/figure_4_risk_regime_frequency_by_year.png",
    "outputs/final_report/figures/figure_5_risk_regime_distribution.png",
    "outputs/final_report/figures/figure_6_arima_residual_acf_pacf.png",
    "outputs/final_report/tables/table_1_data_summary.csv",
    "outputs/final_report/tables/table_2_model_comparison.csv",
    "outputs/final_report/tables/table_3_regime_summary.csv",
    "outputs/final_report/tables/table_4_risk_regimes_by_year.csv",
    "outputs/final_report/tables/table_5_ljung_box_tests.csv",
    "outputs/final_report/tables/table_6_final_20_day_forecast.csv",
    "outputs/final_report/tables/table_7_latest_risk_signal.csv",
    "outputs/final_report/text/final_results_summary.txt"
  )
)

write_csv(manifest, "outputs/final_report/final_report_manifest.csv")

cat("\nFinal report outputs created.\n")
cat("Figures: outputs/final_report/figures\n")
cat("Tables: outputs/final_report/tables\n")
cat("Summary: outputs/final_report/text/final_results_summary.txt\n")
cat("Manifest: outputs/final_report/final_report_manifest.csv\n")

