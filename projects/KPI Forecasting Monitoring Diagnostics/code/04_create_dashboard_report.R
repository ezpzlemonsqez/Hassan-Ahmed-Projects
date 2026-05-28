# Standalone HTML report builder.
# This avoids a hard dependency on Pandoc/R Markdown while still producing a
# clean project report for GitHub or a portfolio page.

check_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is required. Install it with install.packages('", pkg, "').")
  }
}

invisible(lapply(c("dplyr", "readr", "lubridate", "scales"), check_package))

library(dplyr)
library(readr)
library(lubridate)
library(scales)

report_dir <- file.path(project_root, "outputs", "report")
table_dir <- file.path(project_root, "outputs", "tables")
data_file <- file.path(project_root, "outputs", "data", "synthetic_insurance_operational_kpis.csv")

dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

kpi_data <- read_csv(data_file, show_col_types = FALSE) %>% mutate(date = as.Date(date))
model_selection <- read_csv(file.path(table_dir, "model_selection_summary.csv"), show_col_types = FALSE)
regression_summary <- read_csv(file.path(table_dir, "regression_model_summary.csv"), show_col_types = FALSE)
rolling_metrics <- read_csv(file.path(table_dir, "rolling_forecast_metrics.csv"), show_col_types = FALSE)
model_health <- read_csv(file.path(table_dir, "model_health_summary.csv"), show_col_types = FALSE)
alerts <- read_csv(file.path(table_dir, "anomaly_alerts.csv"), show_col_types = FALSE) %>% mutate(date = as.Date(date))
current_snapshot <- read_csv(file.path(table_dir, "kpi_current_snapshot.csv"), show_col_types = FALSE) %>% mutate(date = as.Date(date))

html_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x <- gsub('"', "&quot;", x, fixed = TRUE)
  x
}

fmt_comma <- function(x, digits = 0) {
  ifelse(is.na(x), "", scales::comma(x, accuracy = 10^-digits))
}

fmt_pct <- function(x, digits = 1) {
  ifelse(is.na(x), "", scales::percent(x, accuracy = 10^-digits))
}

fmt_decimal <- function(x, digits = 3) {
  ifelse(is.na(x), "", formatC(x, format = "f", digits = digits))
}

ratio_kpis <- c("lapse_rate", "loss_ratio", "operating_margin")
large_kpis <- c("earned_premium")
count_kpis <- c("claims_count")

format_actual <- function(kpi, value) {
  if (kpi %in% ratio_kpis) return(fmt_pct(value, 2))
  if (kpi %in% large_kpis) return(fmt_comma(value, 0))
  if (kpi %in% count_kpis) return(fmt_comma(value, 0))
  fmt_comma(value, 2)
}

format_metric <- function(kpi, value) {
  if (kpi %in% ratio_kpis) return(fmt_pct(value, 2))
  if (kpi %in% large_kpis) return(fmt_comma(value, 0))
  if (kpi %in% count_kpis) return(fmt_comma(value, 1))
  fmt_decimal(value, 4)
}

format_change <- function(value, basis) {
  ifelse(
    is.na(value),
    "",
    ifelse(
      basis == "Percentage-point change",
      paste0(ifelse(value > 0, "+", ""), fmt_pct(value, 2), " pp"),
      paste0(ifelse(value > 0, "+", ""), fmt_pct(value, 1))
    )
  )
}

make_table <- function(df) {
  if (nrow(df) == 0) {
    return("<p class='muted'>No rows to display.</p>")
  }

  header <- paste0("<th>", html_escape(names(df)), "</th>", collapse = "")
  rows <- apply(df, 1, function(row) {
    paste0("<tr>", paste0("<td>", html_escape(row), "</td>", collapse = ""), "</tr>")
  })
  paste0("<div class='table-wrap'><table><thead><tr>", header, "</tr></thead><tbody>", paste(rows, collapse = "\n"), "</tbody></table></div>")
}

figure <- function(path, caption, alt = caption) {
  paste0(
    "<figure><img src='", path, "' alt='", html_escape(alt), "'>",
    "<figcaption>", html_escape(caption), "</figcaption></figure>"
  )
}

dataset_summary <- tibble(
  `Start date` = as.character(min(kpi_data$date)),
  `End date` = as.character(max(kpi_data$date)),
  `Months` = as.character(nrow(kpi_data)),
  `Average policy count` = fmt_comma(mean(kpi_data$policy_count), 0),
  `Average earned premium` = fmt_comma(mean(kpi_data$earned_premium), 0),
  `Average claims count` = fmt_comma(mean(kpi_data$claims_count), 0),
  `Average loss ratio` = fmt_pct(mean(kpi_data$loss_ratio), 2),
  `Average lapse rate` = fmt_pct(mean(kpi_data$lapse_rate), 2),
  `Average operating margin` = fmt_pct(mean(kpi_data$operating_margin), 2)
)

model_selection_display <- model_selection %>%
  transmute(
    `KPI` = kpi_label,
    `Selected ARIMA model` = selected_model,
    `AIC` = fmt_decimal(aic, 2),
    `BIC` = fmt_decimal(bic, 2),
    `Innovation variance` = ifelse(sigma2 >= 1000, fmt_comma(sigma2, 0), fmt_decimal(sigma2, 5))
  )

regression_display <- regression_summary %>%
  transmute(
    `KPI` = kpi_label,
    `Selected regression formula` = selected_formula,
    `AIC` = fmt_decimal(aic, 2),
    `Adjusted R²` = fmt_decimal(adjusted_r_squared, 3),
    `Residual standard error` = ifelse(residual_standard_error >= 1000, fmt_comma(residual_standard_error, 0), fmt_decimal(residual_standard_error, 5))
  )

rolling_display <- rolling_metrics %>%
  transmute(
    `KPI` = kpi_label,
    `Observations` = as.character(observations),
    `MAE` = mapply(format_metric, kpi, mae),
    `RMSE` = mapply(format_metric, kpi, rmse),
    `MAPE` = fmt_pct(mape, 1),
    `Bias` = mapply(format_metric, kpi, bias),
    `95% interval coverage` = fmt_pct(interval_95_coverage, 1)
  )

health_display <- model_health %>%
  transmute(
    `KPI` = kpi_label,
    `Selected model` = selected_model,
    `Ljung-Box p-value` = fmt_decimal(ljung_box_p_value, 4),
    `Residual mean` = mapply(format_metric, kpi, residual_mean),
    `Residual SD` = mapply(format_metric, kpi, residual_sd),
    `Health status` = health_status
  )

snapshot_display <- current_snapshot %>%
  transmute(
    `KPI` = kpi_label,
    `Date` = as.character(date),
    `Current value` = mapply(format_actual, kpi, actual),
    `MoM change` = mapply(format_change, mom_change, change_basis),
    `YoY change` = mapply(format_change, yoy_change, change_basis),
    `Change basis` = change_basis
  )

recent_alerts_display <- alerts %>%
  arrange(desc(date), kpi_label, alert_type) %>%
  slice_head(n = 10) %>%
  transmute(
    `Date` = as.character(date),
    `KPI` = kpi_label,
    `Alert type` = alert_type,
    `Detail` = alert_detail,
    `Actual` = mapply(format_actual, kpi, actual),
    `Reference` = mapply(format_actual, kpi, reference_value),
    `Lower threshold` = mapply(format_actual, kpi, lower_threshold),
    `Upper threshold` = mapply(format_actual, kpi, upper_threshold)
  )

alert_summary_display <- alerts %>%
  count(kpi_label, alert_type, name = "alerts") %>%
  arrange(kpi_label, alert_type) %>%
  transmute(
    `KPI` = kpi_label,
    `Alert type` = alert_type,
    `Alert count` = as.character(alerts)
  )

loss_ratio_health <- model_health %>% filter(kpi == "loss_ratio")
loss_ratio_note <- if (nrow(loss_ratio_health) == 1 && !is.na(loss_ratio_health$ljung_box_p_value) && loss_ratio_health$ljung_box_p_value < 0.05) {
  "The loss-ratio diagnostic is intentionally shown as a representative model-health check. Its Ljung-Box result flags remaining residual autocorrelation, which is useful in this project because the workflow is designed to surface model risks rather than hide them."
} else {
  "The loss-ratio diagnostic is shown as a representative model-health check. The same residual summary process is applied to all forecasted KPIs."
}

html <- paste0(
"<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>KPI Forecasting & Monitoring Diagnostics</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --border: #d9e2ec;
      --accent: #1f4e79;
      --accent-soft: #e8f0f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }
    .page {
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }
    header {
      background: linear-gradient(135deg, #102a43, #1f4e79);
      color: #fff;
      border-radius: 18px;
      padding: 30px 34px;
      margin-bottom: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
    }
    header h1 { margin: 0 0 8px; font-size: 2rem; }
    header p { margin: 0; color: #dbeafe; }
    section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin: 18px 0;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    h2 { margin-top: 0; color: var(--accent); }
    h3 { color: #243b53; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
    .pill { background: var(--accent-soft); color: #12395a; padding: 7px 10px; border-radius: 999px; font-size: 0.92rem; }
    .muted { color: var(--muted); }
    .note { border-left: 4px solid var(--accent); background: #f0f5fa; padding: 12px 14px; border-radius: 10px; color: #243b53; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; margin: 14px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    th { background: #f0f5fa; color: #102a43; font-weight: 700; }
    tr:last-child td { border-bottom: 0; }
    figure { margin: 20px 0; border: 1px solid var(--border); border-radius: 14px; padding: 12px; background: #fff; }
    img { max-width: 100%; height: auto; display: block; border-radius: 10px; }
    figcaption { margin-top: 8px; color: var(--muted); font-size: 0.9rem; }
  </style>
</head>
<body>
<div class='page'>
<header>
  <h1>KPI Forecasting & Monitoring Diagnostics</h1>
  <p>Synthetic insurance and operational KPIs | R project report</p>
</header>

<section>
  <h2>Executive summary</h2>
  <p>The workflow simulates monthly KPIs, fits ARIMA and regression models, evaluates forecasts through rolling backtests, checks residual behaviour and produces monitoring outputs for KPI movements and anomaly alerts.</p>
  <div class='pill-row'>
    <span class='pill'>ARIMA forecasting</span>
    <span class='pill'>AIC model selection</span>
    <span class='pill'>Regression drivers</span>
    <span class='pill'>Rolling evaluation</span>
    <span class='pill'>Residual diagnostics</span>
    <span class='pill'>Anomaly alerts</span>
  </div>
</section>

<section>
  <h2>Dataset summary</h2>
  <p>The data combines insurance KPIs with broader operational performance measures, so the same project can be discussed in actuarial or finance analytics terms.</p>
  ", make_table(dataset_summary), "
  ", figure("../plots/monitoring/synthetic_kpi_time_series.png", "Synthetic KPI time series for the five forecasted measures."), "
</section>

<section>
  <h2>ARIMA and regression model selection</h2>
  <p>Each KPI is fitted with an AIC-selected ARIMA model for forecasting. Regression models are also selected with AIC to show how KPI movements can be linked to drivers such as seasonality, exposure, inflation and economic stress.</p>
  <h3>ARIMA models</h3>
  ", make_table(model_selection_display), "
  <h3>Regression models</h3>
  <p>The operating-margin regression avoids same-period loss ratio and expense ratio inputs, because those variables define the margin mechanically. This keeps the regression closer to a predictive driver model rather than a formula reconstruction.</p>
  ", make_table(regression_display), "
</section>

<section>
  <h2>Forecasting and rolling evaluation</h2>
  <p>The ARIMA models are tested using expanding-window one-step-ahead forecasts. The compact forecast chart shows all five KPI forecasts in one figure rather than separate plots for every KPI.</p>
  ", make_table(rolling_display), "
  ", figure("../plots/forecast/forecast_summary.png", "Twelve-month ARIMA forecast summary with 95% prediction intervals."), "
  ", figure("../plots/monitoring/rolling_rmse_by_kpi.png", "Six-month rolling RMSE used to monitor forecast stability over time."), "
</section>

<section>
  <h2>Model health diagnostics</h2>
  <p>The Ljung-Box and residual summary table covers every model. To keep the report concise, the visual residual diagnostics are shown for loss ratio as a representative KPI.</p>
  ", make_table(health_display), "
  <p class='note'>", html_escape(loss_ratio_note), "</p>
  ", figure("../plots/diagnostics/loss_ratio_diagnostics.png", "Representative residual, ACF and PACF diagnostics for loss ratio."), "
</section>

<section>
  <h2>KPI monitoring and anomaly alerts</h2>
  <p>Monitoring rules flag KPI values that breach prior 12-month rolling thresholds or fall outside one-step-ahead prediction intervals. The full alert CSV is retained, while the report shows an alert count summary and the ten most recent alerts.</p>
  <h3>Current KPI snapshot</h3>
  ", make_table(snapshot_display), "
  ", figure("../plots/monitoring/latest_kpi_movement.png", "Latest KPI movements. Volumes/revenue use relative percentage change; rates and margins use percentage-point change."), "
  <h3>Alert summary</h3>
  ", make_table(alert_summary_display), "
  <h3>Ten most recent alerts</h3>
  ", make_table(recent_alerts_display), "
  ", figure("../plots/monitoring/loss_ratio_monitoring_thresholds.png", "Loss-ratio monitoring against prior 12-month rolling threshold bands."), "
</section>

<section>
  <h2>Interpretation</h2>
  <p>Claims count, loss ratio and lapse rate support insurance pricing, reserving and monitoring discussions. Earned premium, operating margin and retention support finance and business analytics discussions. The rolling errors, residual checks and alert tables show how the forecasts can be monitored rather than treated as static model outputs.</p>
</section>

</div>
</body>
</html>"
)

report_file <- file.path(report_dir, "KPI_Forecasting_Monitoring_Report.html")
writeLines(html, report_file, useBytes = TRUE)
cat("Standalone HTML report written to:", report_file, "\n")
