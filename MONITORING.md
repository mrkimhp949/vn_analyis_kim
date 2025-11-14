# 📊 Monitoring & Observability Guide

Complete guide to set up production-grade monitoring for the Vietnam Trading Bot.

---

## 🎯 **MONITORING STACK**

This bot includes:
- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **Health Checks** - System health monitoring
- **Performance Benchmarks** - Performance tracking

---

## 🚀 **QUICK START**

### **1. Enable Prometheus Metrics**

Add to your `main.py`:

```python
from prometheus_metrics import add_metrics_endpoint

# After creating FastAPI app
add_metrics_endpoint(app)
```

Access metrics at: `http://localhost:8080/metrics`

### **2. Run Health Checks**

```bash
# Manual health check
python health_check.py

# Scheduled health checks (cron)
*/5 * * * * /path/to/venv/bin/python /path/to/health_check.py
```

### **3. View Performance Metrics**

```bash
python scripts/view_metrics.py
```

### **4. Run Benchmarks**

```bash
python scripts/benchmark.py
```

---

## 📈 **PROMETHEUS SETUP**

### **Install Prometheus**

```bash
# Ubuntu/Debian
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*
```

### **Configure Prometheus**

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-bot'
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:8080']
        labels:
          instance: 'trading-bot-prod'
          environment: 'production'
```

### **Start Prometheus**

```bash
./prometheus --config.file=prometheus.yml
```

Access Prometheus UI at: `http://localhost:9090`

---

## 📊 **GRAFANA SETUP**

### **Install Grafana**

```bash
# Ubuntu/Debian
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access Grafana at: `http://localhost:3000` (default login: admin/admin)

### **Add Prometheus Data Source**

1. Go to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Set URL: `http://localhost:9090`
5. Click **Save & Test**

### **Import Dashboard**

1. Go to **Dashboards** → **Import**
2. Upload `grafana/trading_bot_dashboard.json`
3. Select Prometheus data source
4. Click **Import**

---

## 📊 **AVAILABLE METRICS**

### **Portfolio Metrics**

| Metric | Description | Type |
|--------|-------------|------|
| `portfolio_total_value` | Total portfolio value (VND) | Gauge |
| `portfolio_pnl` | Portfolio P&L (VND) | Gauge |
| `portfolio_pnl_percent` | Portfolio P&L percentage | Gauge |
| `portfolio_positions` | Number of active positions | Gauge |

### **Trading Metrics**

| Metric | Description | Type |
|--------|-------------|------|
| `signals_generated_total` | Total signals generated | Counter |
| `signal_confidence` | Signal confidence distribution | Histogram |
| `trades_executed_total` | Total trades executed | Counter |
| `trade_pnl` | Trade P&L distribution | Histogram |

### **System Metrics**

| Metric | Description | Type |
|--------|-------------|------|
| `ml_predictions_total` | Total ML predictions | Counter |
| `ml_prediction_seconds` | ML prediction latency | Histogram |
| `api_requests_total` | Total API requests | Counter |
| `api_latency_seconds` | API request latency | Histogram |
| `db_queries_total` | Total database queries | Counter |
| `cache_hits_total` | Cache hits | Counter |
| `cache_misses_total` | Cache misses | Counter |
| `errors_total` | Total errors | Counter |

### **Risk Metrics**

| Metric | Description | Type |
|--------|-------------|------|
| `risk_max_drawdown` | Maximum drawdown % | Gauge |
| `risk_sharpe_ratio` | Sharpe ratio | Gauge |
| `risk_sortino_ratio` | Sortino ratio | Gauge |
| `risk_sector_exposure` | Sector exposure % | Gauge |

---

## 🔔 **ALERTING**

### **Prometheus Alert Rules**

Create `alert.rules.yml`:

```yaml
groups:
  - name: trading_bot_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # Portfolio large loss
      - alert: PortfolioLargeLoss
        expr: portfolio_pnl_percent < -10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Portfolio loss > 10%"
          description: "Current P&L: {{ $value }}%"

      # API latency high
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(api_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "p95 latency: {{ $value }}s"

      # No trading signals
      - alert: NoTradingSignals
        expr: rate(signals_generated_total[1h]) == 0
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "No trading signals generated"
          description: "No signals in the last hour"

      # Database query slow
      - alert: SlowDatabaseQueries
        expr: histogram_quantile(0.95, rate(db_query_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow database queries"
          description: "p95 query time: {{ $value }}s"
```

### **Configure Alertmanager**

```yaml
# alertmanager.yml
route:
  receiver: 'telegram'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'telegram'
    telegram_configs:
      - bot_token: 'YOUR_TELEGRAM_BOT_TOKEN'
        chat_id: YOUR_CHAT_ID
        message: '{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'
```

---

## 🔍 **QUERY EXAMPLES**

### **Prometheus Queries**

```promql
# Average signal confidence over time
avg(signal_confidence)

# Trade success rate
sum(rate(trades_executed_total{action="SELL"}[1h])) /
sum(rate(trades_executed_total[1h]))

# Cache hit rate
rate(cache_hits_total[5m]) /
(rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Error rate by component
rate(errors_total[5m]) by (component)

# Portfolio growth
rate(portfolio_total_value[1h])
```

---

## 📱 **HEALTH CHECK INTEGRATION**

### **Setup Cron Job**

```bash
# Edit crontab
crontab -e

# Add health check every 5 minutes
*/5 * * * * /path/to/venv/bin/python /path/to/health_check.py || /usr/bin/mail -s "Trading Bot Health Check Failed" your@email.com
```

### **UptimeRobot Integration**

1. Go to [UptimeRobot](https://uptimerobot.com/)
2. Add new monitor
3. Type: HTTP(s)
4. URL: `https://your-app.com/health`
5. Interval: 5 minutes
6. Alert contacts: Email/Telegram

---

## 🎯 **PERFORMANCE TARGETS**

| Component | Target | Critical |
|-----------|--------|----------|
| ML Prediction | < 100ms | < 500ms |
| Data Loading | < 500ms | < 2s |
| Entry Logic | < 50ms | < 200ms |
| DB Query | < 10ms | < 100ms |
| API Response | < 200ms | < 1s |

Run benchmarks regularly:

```bash
python scripts/benchmark.py
```

---

## 📊 **DASHBOARD OVERVIEW**

The Grafana dashboard includes:

1. **Portfolio Overview**
   - Total value
   - P&L percentage
   - Active positions

2. **Trading Activity**
   - Signals generated
   - Signal confidence
   - Trades executed

3. **Performance Metrics**
   - ML prediction latency
   - API request rate
   - Cache hit rate

4. **Risk Metrics**
   - Sharpe ratio
   - Max drawdown
   - Sector exposure

5. **System Health**
   - Error rate
   - Database performance
   - Paper trading results

---

## 🔧 **TROUBLESHOOTING**

### **No Metrics Showing**

```bash
# Check if metrics endpoint is accessible
curl http://localhost:8080/metrics

# Check if Prometheus is scraping
# Go to Prometheus UI → Status → Targets
```

### **Grafana Dashboard Empty**

1. Check data source connection
2. Verify time range
3. Check Prometheus is collecting metrics
4. Ensure bot is running

### **High Memory Usage**

```bash
# Check memory usage
python scripts/view_metrics.py

# Clear cache if needed
rm -rf data_cache/*.pkl
```

---

## 📚 **RESOURCES**

- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **PromQL Guide**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Alert Rules**: https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/

---

**Happy Monitoring! 📊📈**
