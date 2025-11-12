"""
Portfolio History Tracker
Lưu lịch sử portfolio để phân tích PnL theo ngày và equity curve
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

PORTFOLIO_HISTORY_FILE = "portfolio_history.json"


class PortfolioHistoryTracker:
    """Theo dõi lịch sử portfolio theo ngày"""
    
    def __init__(self, history_file: str = PORTFOLIO_HISTORY_FILE):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load lịch sử từ file"""
        if not os.path.exists(self.history_file):
            return {
                "daily_snapshots": [],
                "equity_curve": [],
                "last_updated": None
            }
        
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "daily_snapshots": [],
                "equity_curve": [],
                "last_updated": None
            }
    
    def _save_history(self):
        """Lưu lịch sử vào file"""
        self.history["last_updated"] = datetime.now().isoformat()
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def record_daily_snapshot(self, portfolio_data: Dict):
        """
        Ghi lại snapshot portfolio trong ngày
        
        Args:
            portfolio_data: Dict với keys:
                - date: ISO format date string
                - total_value: Tổng giá trị portfolio
                - total_invested: Tổng số tiền đã đầu tư
                - total_pnl: Lợi nhuận/lỗ
                - total_return_pct: % return
                - holdings: Dict {symbol: {shares, current_price, value, pnl}}
                - sector_exposure: Dict {sector: percentage}
        """
        today = datetime.now().date().isoformat()
        
        # Kiểm tra xem đã có snapshot hôm nay chưa
        existing_snapshot = None
        for snapshot in self.history["daily_snapshots"]:
            if snapshot.get("date") == today:
                existing_snapshot = snapshot
                break
        
        snapshot_data = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "total_value": portfolio_data.get("total_value", 0),
            "total_invested": portfolio_data.get("total_invested", 0),
            "total_pnl": portfolio_data.get("total_pnl", 0),
            "total_return_pct": portfolio_data.get("total_return_pct", 0),
            "num_positions": portfolio_data.get("num_positions", 0),
            "holdings": portfolio_data.get("holdings", {}),
            "sector_exposure": portfolio_data.get("sector_exposure", {}),
        }
        
        if existing_snapshot:
            # Update existing snapshot
            idx = self.history["daily_snapshots"].index(existing_snapshot)
            self.history["daily_snapshots"][idx] = snapshot_data
        else:
            # Add new snapshot
            self.history["daily_snapshots"].append(snapshot_data)
        
        # Update equity curve
        self._update_equity_curve(snapshot_data)
        
        self._save_history()
    
    def _update_equity_curve(self, snapshot: Dict):
        """Cập nhật equity curve"""
        equity_point = {
            "date": snapshot["date"],
            "equity": snapshot["total_value"],
            "pnl": snapshot["total_pnl"],
            "return_pct": snapshot["total_return_pct"]
        }
        
        # Kiểm tra xem đã có điểm này chưa
        existing = None
        for point in self.history["equity_curve"]:
            if point.get("date") == snapshot["date"]:
                existing = point
                break
        
        if existing:
            idx = self.history["equity_curve"].index(existing)
            self.history["equity_curve"][idx] = equity_point
        else:
            self.history["equity_curve"].append(equity_point)
        
        # Sắp xếp theo date
        self.history["equity_curve"].sort(key=lambda x: x.get("date", ""))
    
    def get_equity_curve(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Lấy equity curve dưới dạng DataFrame
        
        Args:
            days: Số ngày gần nhất (None = tất cả)
            
        Returns:
            DataFrame với columns: date, equity, pnl, return_pct
        """
        curve = self.history["equity_curve"]
        
        if days:
            cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
            curve = [p for p in curve if p.get("date", "") >= cutoff_date]
        
        if not curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(curve)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        return df
    
    def get_daily_pnl(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Lấy PnL theo ngày
        
        Returns:
            DataFrame với columns: date, pnl, return_pct
        """
        df = self.get_equity_curve(days)
        if df.empty:
            return pd.DataFrame()
        
        return df[["date", "pnl", "return_pct"]].copy()
    
    def get_performance_metrics(self, days: Optional[int] = None) -> Dict:
        """
        Tính các metrics hiệu suất
        
        Returns:
            Dict với các metrics:
            - total_return: Tổng return %
            - daily_avg_return: Return trung bình/ngày
            - volatility: Độ biến động
            - sharpe_ratio: Sharpe ratio (nếu có đủ data)
            - max_drawdown: Maximum drawdown
            - win_rate: Tỷ lệ ngày có lãi
        """
        df = self.get_equity_curve(days)
        
        if df.empty or len(df) < 2:
            return {
                "error": "Không đủ dữ liệu"
            }
        
        # Tính daily returns
        df["daily_return"] = df["return_pct"].pct_change().fillna(0)
        
        # Total return
        initial_return = df["return_pct"].iloc[0]
        final_return = df["return_pct"].iloc[-1]
        total_return = final_return - initial_return
        
        # Average daily return
        daily_avg_return = df["daily_return"].mean()
        
        # Volatility (std of daily returns)
        volatility = df["daily_return"].std()
        
        # Sharpe ratio (giả sử risk-free rate = 0)
        sharpe_ratio = (daily_avg_return / volatility) if volatility > 0 else 0
        
        # Maximum drawdown
        df["cumulative_max"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["cumulative_max"]) / df["cumulative_max"]
        max_drawdown = df["drawdown"].min() * 100  # Convert to percentage
        
        # Win rate (số ngày có lãi / tổng số ngày)
        positive_days = (df["daily_return"] > 0).sum()
        win_rate = (positive_days / len(df)) * 100 if len(df) > 0 else 0
        
        return {
            "total_return_pct": float(total_return),
            "daily_avg_return_pct": float(daily_avg_return * 100),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "win_rate_pct": float(win_rate),
            "total_days": len(df),
            "positive_days": int(positive_days),
        }
    
    def get_holdings_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Lấy lịch sử holdings
        
        Args:
            symbol: Nếu None thì lấy tất cả, nếu có thì chỉ lấy symbol đó
            
        Returns:
            DataFrame với holdings theo thời gian
        """
        snapshots = self.history["daily_snapshots"]
        
        if not snapshots:
            return pd.DataFrame()
        
        records = []
        for snapshot in snapshots:
            date = snapshot.get("date")
            holdings = snapshot.get("holdings", {})
            
            if symbol:
                if symbol in holdings:
                    records.append({
                        "date": date,
                        "symbol": symbol,
                        **holdings[symbol]
                    })
            else:
                for sym, data in holdings.items():
                    records.append({
                        "date": date,
                        "symbol": sym,
                        **data
                    })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "symbol"])
        
        return df
    
    def export_to_csv(self, output_file: str = "portfolio_history.csv"):
        """Export lịch sử ra CSV"""
        df = self.get_equity_curve()
        if not df.empty:
            df.to_csv(output_file, index=False)
            return True
        return False


def format_performance_report(metrics: Dict) -> str:
    """Format performance metrics thành text report"""
    if "error" in metrics:
        return f"❌ {metrics['error']}"
    
    lines = [
        "📊 *Performance Metrics:*\n",
        f"📈 Total Return: {metrics['total_return_pct']:+.2f}%",
        f"📊 Daily Avg Return: {metrics['daily_avg_return_pct']:+.2f}%",
        f"📉 Volatility: {metrics['volatility']:.4f}",
        f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"🔻 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%",
        f"✅ Win Rate: {metrics['win_rate_pct']:.1f}% ({metrics['positive_days']}/{metrics['total_days']} days)",
    ]
    
    return "\n".join(lines)

