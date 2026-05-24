############################################################
# Volatility Forecasting & Risk Signal Analysis | VIX Index
# Language: R
#
# Purpose:
# - Download and clean daily VIX data
# - Forecast log-VIX using ARIMA and regression models
# - Perform rolling out-of-sample evaluation
# - Generate prediction intervals
# - Run residual diagnostics: ACF, PACF, Ljung-Box
# - Build risk signals and analyse regime shifts
############################################################

############################
# 1. Setup
############################

required_packages <- c(
  "readr", "dplyr", "tidyr", "lubridate", "ggplot2",
  "zoo", "forecast", "broom", "strucchange"
)

installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)

if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(zoo)
library(forecast)
#library(MASS)
library(broom)
library(strucchange)

set.seed(123)

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)

fred_url <- "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
raw_file <- "data/raw/vix_fred.csv"

if (!file.exists(raw_file)) {
  download.file(fred_url, raw_file, mode = "wb")
}

############################
# 2. Load and clean VIX data
############################

raw_vix <- read_csv(raw_file, show_col_types = FALSE)

vix <- raw_vix %>%
  rename(
    date = observation_date,
    vix_raw = VIXCLS
  ) %>%
  mutate(
    vix = as.numeric(na_if(as.character(vix_raw), ".")),
    log_vix = log(vix)
  ) %>%
  dplyr::select(date, vix, log_vix) %>%
  filter(!is.na(vix), vix > 0) %>%
  arrange(date) %>%
  mutate(
    row_id = row_number(),
    dlog_vix = log_vix - lag(log_vix)
  )

write_csv(vix, "data/processed/vix_clean.csv")

cat("Clean VIX observations:", nrow(vix), "\n")
cat("Start date:", as.character(min(vix$date)), "\n")
cat("End date:", as.character(max(vix$date)), "\n")

############################
# 3. Feature engineering
############################

roll_mean <- function(x, n) {
  zoo::rollapply(x, width = n, FUN = mean, fill = NA, align = "right", na.rm = TRUE)
}

roll_sd <- function(x, n) {
  zoo::rollapply(x, width = n, FUN = sd, fill = NA, align = "right", na.rm = TRUE)
}

roll_quantile <- function(x, n, p) {
  zoo::rollapply(
    x,
    width = n,
    FUN = function(z) quantile(z, probs = p, na.rm = TRUE),
    fill = NA,
    align = "right"
  )
}

model_df <- vix %>%
  mutate(
    log_lag1 = lag(log_vix),
    dlog_lag1 = lag(dlog_vix),

    ma5_lag1 = lag(roll_mean(log_vix, 5)),
    ma21_lag1 = lag(roll_mean(log_vix, 21)),
    ma63_lag1 = lag(roll_mean(log_vix, 63)),

    sd21_lag1 = lag(roll_sd(log_vix, 21)),
    sd63_lag1 = lag(roll_sd(log_vix, 63)),

    vix_q75_252_lag1 = lag(roll_quantile(vix, 252, 0.75)),
    vix_q90_252_lag1 = lag(roll_quantile(vix, 252, 0.90)),

    stress_lag1 = if_else(lag(vix) >= vix_q90_252_lag1, 1, 0, missing = 0),
    elevated_lag1 = if_else(lag(vix) >= vix_q75_252_lag1, 1, 0, missing = 0),

    trend = row_number()
  )

write_csv(model_df, "data/processed/vix_model_features.csv")

############################
# 4. Risk signal construction
############################

signal_df <- model_df %>%
  mutate(
    rolling_mean_252 = roll_mean(vix, 252),
    rolling_sd_252 = roll_sd(vix, 252),
    z_252 = (vix - rolling_mean_252) / rolling_sd_252,

    risk_signal = case_when(
      !is.na(vix_q90_252_lag1) & vix >= vix_q90_252_lag1 ~ "Stress",
      !is.na(vix_q75_252_lag1) & vix >= vix_q75_252_lag1 ~ "Elevated",
      TRUE ~ "Normal"
    )
  )

write_csv(signal_df, "outputs/tables/vix_risk_signals.csv")

latest_signal <- signal_df %>%
  filter(!is.na(risk_signal)) %>%
  slice_tail(n = 1) %>%
  dplyr::select(date, vix, risk_signal, z_252, vix_q75_252_lag1, vix_q90_252_lag1)

write_csv(latest_signal, "outputs/tables/latest_risk_signal.csv")

############################
# 5. Plots: VIX and signals
############################

p_vix <- ggplot(signal_df, aes(x = date, y = vix)) +
  geom_line(linewidth = 0.35) +
  geom_line(aes(y = vix_q75_252_lag1), linetype = "dashed", linewidth = 0.3, na.rm = TRUE) +
  geom_line(aes(y = vix_q90_252_lag1), linetype = "dotted", linewidth = 0.3, na.rm = TRUE) +
  labs(
    title = "VIX Index with Rolling Risk Thresholds",
    subtitle = "Dashed = rolling 75th percentile, dotted = rolling 90th percentile",
    x = "Date",
    y = "VIX"
  ) +
  theme_minimal()

ggsave("outputs/plots/vix_with_risk_thresholds.png", p_vix, width = 10, height = 5, dpi = 300)

p_signal <- signal_df %>%
  filter(date >= max(date) - years(5)) %>%
  ggplot(aes(x = date, y = vix, colour = risk_signal)) +
  geom_line(linewidth = 0.5) +
  labs(
    title = "Recent VIX Risk Signal Classification",
    subtitle = "Normal, Elevated and Stress regimes based on rolling thresholds",
    x = "Date",
    y = "VIX",
    colour = "Signal"
  ) +
  theme_minimal()

ggsave("outputs/plots/recent_vix_risk_signals.png", p_signal, width = 10, height = 5, dpi = 300)

############################
# 6. Rolling evaluation setup
############################

# Default: evaluate over last 3 years using a 5-year rolling training window.
# Increase evaluation_years to 5 if you want a heavier, more complete backtest.

evaluation_years <- 3
rolling_window_years <- 5
trading_days <- 252

rolling_window <- rolling_window_years * trading_days
eval_start_date <- max(vix$date) - years(evaluation_years)

eval_indices <- which(vix$date >= eval_start_date & vix$row_id > rolling_window + 100)

cat("Rolling evaluation observations:", length(eval_indices), "\n")

############################
# 7. AIC-selected regression specification
############################

feature_cols <- c(
  "log_lag1", "dlog_lag1",
  "ma5_lag1", "ma21_lag1", "ma63_lag1",
  "sd21_lag1", "sd63_lag1",
  "stress_lag1", "elevated_lag1",
  "trend"
)

selector_data <- model_df %>%
  filter(date < eval_start_date) %>%
  dplyr::select(log_vix, all_of(feature_cols)) %>%
  drop_na()

full_formula <- as.formula(
  paste("log_vix ~", paste(feature_cols, collapse = " + "))
)

full_lm <- lm(full_formula, data = selector_data)

# Stepwise AIC selection gives a parsimonious regression model.
selected_lm <- step(full_lm, direction = "both", trace = 0)
selected_formula <- formula(selected_lm)

cat("Selected regression formula:\n")
print(selected_formula)

saveRDS(selected_lm, "outputs/models/selected_regression_model.rds")

regression_summary <- capture.output(summary(selected_lm))
writeLines(regression_summary, "outputs/tables/selected_regression_summary.txt")

############################
# 8. Rolling ARIMA and regression forecasts
############################

rolling_results <- list()

for (k in seq_along(eval_indices)) {

  i <- eval_indices[k]

  train_start <- max(1, i - rolling_window)
  train_end <- i - 1

  train_y <- vix$log_vix[train_start:train_end]
  actual_log <- vix$log_vix[i]
  actual_vix <- vix$vix[i]
  forecast_date <- vix$date[i]

  # -------------------------
  # ARIMA model
  # -------------------------
  arima_out <- tryCatch({

    fit_arima <- auto.arima(
      train_y,
      seasonal = FALSE,
      ic = "aicc",
      stepwise = TRUE,
      approximation = TRUE
    )

    fc <- forecast(fit_arima, h = 1, level = c(80, 95))

    tibble(
      arima_order = paste0("ARIMA(", paste(arimaorder(fit_arima), collapse = ","), ")"),
      arima_mean_log = as.numeric(fc$mean[1]),
      arima_lower80_log = as.numeric(fc$lower[1, "80%"]),
      arima_upper80_log = as.numeric(fc$upper[1, "80%"]),
      arima_lower95_log = as.numeric(fc$lower[1, "95%"]),
      arima_upper95_log = as.numeric(fc$upper[1, "95%"])
    )

  }, error = function(e) {
    tibble(
      arima_order = NA_character_,
      arima_mean_log = NA_real_,
      arima_lower80_log = NA_real_,
      arima_upper80_log = NA_real_,
      arima_lower95_log = NA_real_,
      arima_upper95_log = NA_real_
    )
  })

  # -------------------------
  # Regression model
  # -------------------------
  train_reg <- model_df %>%
    filter(row_id >= train_start, row_id <= train_end) %>%
    dplyr::select(log_vix, all_of(feature_cols)) %>%
    drop_na()

  test_reg <- model_df %>%
    filter(row_id == i) %>%
    dplyr::select(all_of(feature_cols))

  reg_out <- tryCatch({

    fit_reg <- lm(selected_formula, data = train_reg)

    pred95 <- predict(
      fit_reg,
      newdata = test_reg,
      interval = "prediction",
      level = 0.95
    )

    pred80 <- predict(
      fit_reg,
      newdata = test_reg,
      interval = "prediction",
      level = 0.80
    )

    tibble(
      reg_mean_log = as.numeric(pred95[1, "fit"]),
      reg_lower80_log = as.numeric(pred80[1, "lwr"]),
      reg_upper80_log = as.numeric(pred80[1, "upr"]),
      reg_lower95_log = as.numeric(pred95[1, "lwr"]),
      reg_upper95_log = as.numeric(pred95[1, "upr"])
    )

  }, error = function(e) {
    tibble(
      reg_mean_log = NA_real_,
      reg_lower80_log = NA_real_,
      reg_upper80_log = NA_real_,
      reg_lower95_log = NA_real_,
      reg_upper95_log = NA_real_
    )
  })

  rolling_results[[k]] <- bind_cols(
    tibble(
      date = forecast_date,
      actual_log = actual_log,
      actual_vix = actual_vix
    ),
    arima_out,
    reg_out
  )

  if (k %% 50 == 0) {
    cat("Completed rolling forecast", k, "of", length(eval_indices), "\n")
  }
}

rolling_results <- bind_rows(rolling_results) %>%
  mutate(
    arima_mean_vix = exp(arima_mean_log),
    arima_lower80_vix = exp(arima_lower80_log),
    arima_upper80_vix = exp(arima_upper80_log),
    arima_lower95_vix = exp(arima_lower95_log),
    arima_upper95_vix = exp(arima_upper95_log),

    reg_mean_vix = exp(reg_mean_log),
    reg_lower80_vix = exp(reg_lower80_log),
    reg_upper80_vix = exp(reg_upper80_log),
    reg_lower95_vix = exp(reg_lower95_log),
    reg_upper95_vix = exp(reg_upper95_log)
  )

write_csv(rolling_results, "outputs/tables/rolling_forecasts.csv")

############################
# 9. Model evaluation metrics
############################

metric_calc <- function(data, model_prefix) {

  mean_col <- paste0(model_prefix, "_mean_log")
  lower95_col <- paste0(model_prefix, "_lower95_log")
  upper95_col <- paste0(model_prefix, "_upper95_log")

  mean_vix_col <- paste0(model_prefix, "_mean_vix")
  lower95_vix_col <- paste0(model_prefix, "_lower95_vix")
  upper95_vix_col <- paste0(model_prefix, "_upper95_vix")

  data %>%
    filter(!is.na(.data[[mean_col]])) %>%
    summarise(
      model = model_prefix,

      rmse_log = sqrt(mean((actual_log - .data[[mean_col]])^2)),
      mae_log = mean(abs(actual_log - .data[[mean_col]])),

      rmse_vix = sqrt(mean((actual_vix - .data[[mean_vix_col]])^2)),
      mae_vix = mean(abs(actual_vix - .data[[mean_vix_col]])),

      mape_vix = mean(abs((actual_vix - .data[[mean_vix_col]]) / actual_vix)) * 100,

      coverage_95_log = mean(actual_log >= .data[[lower95_col]] &
                               actual_log <= .data[[upper95_col]]),

      coverage_95_vix = mean(actual_vix >= .data[[lower95_vix_col]] &
                               actual_vix <= .data[[upper95_vix_col]]),

      n_forecasts = n()
    )
}

model_comparison <- bind_rows(
  metric_calc(rolling_results, "arima"),
  metric_calc(rolling_results, "reg")
)

write_csv(model_comparison, "outputs/tables/model_comparison.csv")

print(model_comparison)


cat('\nCore rolling outputs complete.\n')
cat('Next run: source("scripts/fast_finish.R")\n')
cat('Then run: source("scripts/create_final_report_outputs.R")\n')
