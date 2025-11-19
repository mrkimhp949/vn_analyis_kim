# PHÂN TÍCH TOÀN BỘ LOGIC BACKTESTING - BÁO CÁO CHI TIẾT

**Ngày phân tích**: 2025-11-18  
**Người phân tích**: Code Analysis Tool  
**Phạm vi**: Toàn bộ logic backtesting (scripts/backtest.py, scripts/run_backtest.py, backtesting/engine.py)

---

## I. TÓM TẮT PHÁT HIỆN

Phân tích đã xác định **25+ vấn đề** bao gồm:

- **16 lỗi F-String formatting** (HIGH priority)
- **15+ exception handlers** chưa đặc thể (MEDIUM priority)
- **1 lỗi logic P&L calculation** nghiêm trọng (CRITICAL priority)
- **4 vấn đề về dữ liệu validation** (MEDIUM priority)
- **Các vấn đề khác về NaN/INF propagation** (MEDIUM priority)

---

## II. CHI TIẾT CÁC VẤN ĐỀ

### A. F-STRING FORMATTING ERRORS (16 instances)

#### 1. File: `/home/user/vn_analyis_kim/scripts/backtest.py`

| Line | Current Code | Issue | Fix |
|------|-------------|-------|-----|
| 94 | `print("📊 Return: {result['total_return']:.2f}%")` | Missing 'f' prefix | `print(f"📊 Return: {result['total_return']:.2f}%")` |
| 96 | `print("🎯 Win Rate: {result['win_rate']:.2f}%")` | Missing 'f' prefix | `print(f"🎯 Win Rate: {result['win_rate']:.2f}%")` |
| 160 | `print("\n🧪 So sánh các threshold cho {symbol}...\n")` | Missing 'f' prefix | `print(f"\n🧪 So sánh các threshold cho {symbol}...\n")` |
| 166 | `print("⏳ Đang test threshold {threshold}%...")` | Missing 'f' prefix | `print(f"⏳ Đang test threshold {threshold}%...")` |
| 181 | `print("  ✅ {threshold}%: Return {result['total_return']:.2f}%")` | Missing 'f' prefix | `print(f"  ✅ {threshold}%: Return {result['total_return']:.2f}%")` |
| 183 | `print("  ⚠️ Lỗi threshold {threshold}%")` | Missing 'f' prefix | `print(f"  ⚠️ Lỗi threshold {threshold}%")` |

**Impact**: Kết quả backtest sẽ hiển thị sai, ví dụ:
- Thay vì: "📊 Return: 5.25%"
- Sẽ in: "📊 Return: {result['total_return']:.2f}%"

---

#### 2. File: `/home/user/vn_analyis_kim/backtesting/engine.py`

| Line | Current Code | Issue |
|------|-------------|-------|
| 385 | `print("💰 Initial Capital: {result.initial_capital:,.0f} VND")` | Missing 'f' prefix |
| 386 | `print("💰 Final Capital: {result.final_capital:,.0f} VND")` | Missing 'f' prefix |
| 390 | `print("  Annualized Return: {result.annualized_return:.2f}%")` | Missing 'f' prefix |
| 396 | `print("  Winning Trades: {result.winning_trades} ({result.win_rate:.1f}%)")` | Missing 'f' prefix |
| 401 | `print("  Average Win: {result.avg_win:+,.0f} VND")` | Missing 'f' prefix |
| 402 | `print("  Average Loss: {result.avg_loss:+,.0f} VND")` | Missing 'f' prefix |
| 403 | `print("  Largest Win: {result.largest_win:+,.0f} VND")` | Missing 'f' prefix |
| 404 | `print("  Largest Loss: {result.largest_loss:+,.0f} VND")` | Missing 'f' prefix |
| 408 | `print("  Total Commission: {result.total_commission:,.0f} VND")` | Missing 'f' prefix |
| 409 | `print("  Total Slippage: {result.total_slippage:,.0f} VND")` | Missing 'f' prefix |

**Impact**: Method `print_results()` sẽ in các placeholder string thay vì giá trị thực

---

### B. EXCEPTION HANDLING ISSUES (15+ instances)

#### 1. File: `/home/user/vn_analyis_kim/scripts/backtest.py`

**Line 49-53**: Backtester initialization
```python
except Exception:
    print("❌ Lỗi khởi tạo backtester")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

**Issues**:
- Bắt `Exception` quá chung chung (catch-all)
- Không xác định loại lỗi cụ thể
- Khuyến nghị: Bắt `ImportError`, `ValueError`, `AttributeError` riêng

---

**Line 119-123**: Single stock backtest
```python
except Exception:
    print("\n❌ Lỗi backtest")
    import traceback
    traceback.print_exc()
```

**Issues**: Im lặng về lỗi, chỉ in stack trace

---

**Line 141-145**: Multiple stocks backtest
```python
except Exception:
    print("\n❌ Lỗi")
    import traceback
    traceback.print_exc()
```

**Issues**: Thông báo quá chung chung ("Lỗi gì?")

---

**Line 182-184**: Threshold comparison loop
```python
except Exception:
    print("  ⚠️ Lỗi threshold {threshold}%")
```

**Issues**: 
- Missing f-string prefix
- Không log chi tiết lỗi
- Trong loop, một lỗi có thể bị lặp lại

---

#### 2. File: `/home/user/vn_analyis_kim/scripts/run_backtest.py`

**Line 348-349**: Main trading simulation loop
```python
except Exception:
    print(f"⚠️ Lỗi ngày {current_row['time'].date()}")
```

**Issues**:
- Im lặng về lỗi thực tế
- Chỉ in ngày - không biết lỗi gì
- Risk: Bỏ qua lỗi logic quan trọng

---

### C. LOGIC ERRORS - CRITICAL ISSUES

#### 1. Trade P&L Calculation Error (Lines 394-402)

**Location**: `/home/user/vn_analyis_kim/scripts/run_backtest.py`

```python
if len(trades_df) > 0:
    buy_trades = trades_df[trades_df["type"] == "BUY"]
    sell_trades = trades_df[trades_df["type"] == "SELL"]

    for i in range(min(len(buy_trades), len(sell_trades))):
        buy_price = buy_trades.iloc[i]["price"]
        sell_price = sell_trades.iloc[i]["price"]
        shares_traded = sell_trades.iloc[i]["shares"]
        pnl = (
            (sell_price - buy_price) * shares_traded
            - (buy_price * shares_traded * self.commission)
            - (sell_price * shares_trades * self.commission)
        )
```

**VẤN ĐỀ NGHIÊM TRỌNG**:

1. **Giả định sai**: Giả định BUY và SELL được ghép đôi theo thứ tự
   ```
   Đặt hàng gốc:
   BUY[0]  @ 100
   BUY[1]  @ 102
   SELL[0] @ 105 (from TAKE_PROFIT)
   SELL[1] @ 103 (from SIGNAL_SELL)
   ```

2. **Vấn đề thực tế**: Các giao dịch không được ghép đôi đúng
   ```
   Hiện tại logic ghép:
   BUY[0] (100) paired with SELL[0] (105) ✓
   BUY[1] (102) paired with SELL[1] (103) ✓
   
   Nhưng thực tế:
   BUY[0] có thể đóng ở TAKE_PROFIT (SELL[0])
   BUY[1] có thể đóng ở SIGNAL_SELL (SELL[1])
   
   Nếu BUY[0] exit tại dòng 250 (TAKE_PROFIT):
   trades = [..., {type: SELL, exit_reason: TAKE_PROFIT}, ...]
   
   Nhưng BUY[1] không được đóng, nên cũng không có SELL[1]
   ```

3. **Hậu quả**: 
   - P&L calculation sai lệch
   - Win rate không chính xác
   - Báo cáo thống kê không đáng tin

**Ví dụ cụ thể**:
```
Trades DataFrame:
  date  type   price  exit_reason
0  t1   BUY    100    OPEN
1  t2   SELL   105    TAKE_PROFIT
2  t3   BUY    102    OPEN
3  t4   SELL   103    SIGNAL_SELL
4  t5   SELL   101    EOD_EXIT (close remaining BUY)

buy_trades:   [BUY@100, BUY@102]        (2 trades)
sell_trades:  [SELL@105, SELL@103, SELL@101]  (3 trades)

Loop runs for i in range(min(2, 3)) = range(2)
  i=0: pair BUY[0]@100 with SELL[0]@105 ✓
  i=1: pair BUY[1]@102 with SELL[1]@103 ✓
  
Nhưng SELL[2]@101 (EOD_EXIT) không được xử lý!
```

**Fix cần thiết**:
```python
# Theo dõi từng trade bằng cách ghép BUY-SELL theo time
# Hoặc sử dụng trade_id để match chính xác
matched_trades = {}
for buy_trade in buy_trades:
    for sell_trade in sell_trades:
        if (buy_trade.date < sell_trade.date and
            buy_trade.symbol == sell_trade.symbol and
            sell_trade not in matched_trades.values()):
            # Pair them
            matched_trades[buy_trade.name] = sell_trade.name
            break
```

---

#### 2. Profit Factor with float("inf") (Line 449)

```python
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
```

**Issues**:
- Khi `gross_loss = 0` (không có losing trades), trả về `float("inf")`
- Problems:
  - In ra: "Profit Factor: inf"
  - Excel export có thể lỗi
  - So sánh: `inf > 0` = True, nhưng semantically sai
  
**Fix**:
```python
if gross_loss > 0:
    profit_factor = gross_profit / gross_loss
else:
    profit_factor = float('nan') if gross_profit > 0 else 0
```

---

#### 3. ATR NaN Potential (Line 223)

```python
atr_value = current_row.get("atr", current_data["close"].rolling(14).std().iloc[-1])
```

**Issues**:
- Nếu `current_data` có < 14 rows, `rolling(14).std()` trả về toàn NaN
- `.iloc[-1]` sẽ là NaN
- Dòng 269 sử dụng `atr_value` làm fallback - có thể là NaN!

**Impact**:
- Line 274: `stop_loss_price = max(0, execution_price - 2.0 * NaN)` = NaN
- Line 275: `take_profit_price = execution_price + 3.0 * NaN` = NaN

**Fix**:
```python
atr_std = current_data["close"].rolling(14).std()
atr_value = (float(atr_std.iloc[-1]) if pd.notna(atr_std.iloc[-1]) 
             else current_row.get("atr", execution_price * 0.02))
```

---

#### 4. Confidence Validation (Line 261)

```python
if signal == "BUY" and confidence >= confidence_threshold and position == 0:
```

**Issues**:
- `confidence` không được validate
- Nếu `confidence` là None, so sánh >= sẽ raise TypeError

**Fix**:
```python
if (signal == "BUY" and 
    isinstance(confidence, (int, float)) and 
    confidence >= confidence_threshold and 
    position == 0):
```

---

### D. DATA VALIDATION ISSUES

#### 1. Empty DataFrame Handling

**Good Examples**:
- Line 120: `if df.empty: raise ValueError(...)`
- Line 147: `if df.empty: return {...}`

**Issues with edge cases**:
- DataFrame with all NaN values (passes .empty check)
- DataFrame with < 50 rows before reaching line 377

---

#### 2. Division by Zero Risks

**Line 377-378**: Buy & Hold calculation
```python
buy_hold_return = (
    (df.iloc[-1]["close"] - df.iloc[50]["close"]) / df.iloc[50]["close"]
) * 100
```

**Risk**: Nếu `df.iloc[50]["close"] = 0`, sẽ chia cho 0

**Status**: Có check len > 50, nhưng không check giá = 0

---

### E. CONSISTENCY ISSUES

#### 1. Duplicate Backtester Class

**Problem**:
- `/scripts/run_backtest.py`: Class Backtester (873 lines)
- `/backtesting/engine.py`: Class BacktestEngine (412 lines)
- `/scripts/backtest.py`: Import từ `run_backtest`

**Risk**: Không rõ class nào đang được sử dụng

---

### F. NaN/INF PROPAGATION

#### 1. Sharpe Ratio (Line 432)

```python
sharpe_ratio = (
    (returns.mean() / returns.std() * np.sqrt(252))
    if len(returns) > 0 and returns.std() > 0
    else 0
)
```

**Status**: OK - có check `returns.std() > 0`

---

#### 2. Sortino Ratio (Line 438)

```python
sortino_ratio = (
    (returns.mean() / downside_returns.std() * np.sqrt(252))
    if len(downside_returns) > 0 and downside_returns.std() > 0
    else 0
)
```

**Status**: OK - có check downside_returns.std() > 0

---

#### 3. Calmar Ratio (Line 448)

```python
calmar_ratio = (annual_return / (max_drawdown / 100)) if max_drawdown > 0 else 0
```

**Status**: OK - có check max_drawdown > 0

---

## III. TỔNG HỢP BẢN TÓM TẮTPRIORITY MATRIX

```
┌─────────────────┬─────────┬──────────┬──────────┐
│ Vấn đề          │ Loại   │ Mức độ   │ Số lần   │
├─────────────────┼─────────┼──────────┼──────────┤
│ F-string format │ Syntax  │ CRITICAL │ 16       │
│ Trade P&L logic │ Logic   │ CRITICAL │ 1        │
│ Bare exception  │ Handler │ MEDIUM   │ 15+      │
│ ATR NaN         │ Data    │ MEDIUM   │ 1        │
│ profit_factor   │ Logic   │ MEDIUM   │ 1        │
│ Confidence val  │ Logic   │ LOW      │ 1        │
│ Class duplicate │ Import  │ LOW      │ 1        │
└─────────────────┴─────────┴──────────┴──────────┘
```

---

## IV. ACTION ITEMS

### Priority 1 (CRITICAL - Fix Immediately)

- [ ] **Fix 16 F-string missing 'f' prefix**
  - Files: backtest.py (6), engine.py (10)
  - Time: 10 minutes
  - Impact: HIGH (output display)

- [ ] **Fix Trade P&L pairing logic**
  - File: run_backtest.py (lines 394-402)
  - Time: 30-45 minutes
  - Impact: CRITICAL (calculation accuracy)

- [ ] **Add confidence validation**
  - File: run_backtest.py (line 261)
  - Time: 5 minutes
  - Impact: MEDIUM (error prevention)

### Priority 2 (IMPORTANT - Fix Soon)

- [ ] **Replace bare except Exception**
  - Files: backtest.py (5), run_backtest.py (6)
  - Time: 20-30 minutes
  - Impact: MEDIUM (error handling)

- [ ] **Handle float("inf") in profit_factor**
  - File: run_backtest.py (line 449)
  - Time: 5 minutes
  - Impact: MEDIUM (data integrity)

- [ ] **Fix ATR NaN handling**
  - File: run_backtest.py (line 223)
  - Time: 10 minutes
  - Impact: MEDIUM (data validation)

### Priority 3 (NICE-TO-HAVE)

- [ ] Consolidate Backtester classes
- [ ] Add comprehensive unit tests
- [ ] Document edge cases
- [ ] Add logging framework instead of print()

---

## V. QUICK FIX CHECKLIST

```bash
# Test syntax after fixes
python3 -m py_compile scripts/backtest.py scripts/run_backtest.py

# Run pylint for additional issues
pylint scripts/backtest.py scripts/run_backtest.py

# Run tests
python3 -m pytest tests/test_backtesting.py -v
```

---

## VI. REFERENCES

- Python F-strings: https://peps.python.org/pep-0498/
- Exception handling: https://docs.python.org/3/tutorial/errors.html
- Pandas NaN handling: https://pandas.pydata.org/docs/user_guide/missing_data.html

---

**Report Generated**: 2025-11-18  
**Analysis Tool**: Code Review Analysis v1.0
