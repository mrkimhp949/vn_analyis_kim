"""
API Monitoring System
Giám sát API với ping check, retry policy, và alerts khi fail nhiều lần
"""
import requests
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

API_MONITOR_FILE = 'api_monitor.json'


class APIStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class APIHealth:
    """Health status của một API endpoint"""
    endpoint: str
    status: str
    last_success: Optional[str]
    last_failure: Optional[str]
    consecutive_failures: int
    total_requests: int
    total_failures: int
    avg_response_time: float
    last_response_time: Optional[float]
    last_error: Optional[str]


class APIMonitor:
    """
    Monitor API health với:
    - Ping check
    - Retry policy
    - Failure tracking
    - Alerts khi fail nhiều lần
    """
    
    def __init__(
        self,
        max_consecutive_failures: int = 5,
        failure_threshold_pct: float = 20.0,
        alert_callback: Optional[Callable] = None
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_threshold_pct = failure_threshold_pct
        self.alert_callback = alert_callback
        self.health_data = self._load_health_data()
    
    def _load_health_data(self) -> Dict[str, APIHealth]:
        """Load health data từ file"""
        if os.path.exists(API_MONITOR_FILE):
            try:
                with open(API_MONITOR_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        endpoint: APIHealth(**health_dict)
                        for endpoint, health_dict in data.items()
                    }
            except Exception:
                pass
        return {}
    
    def _save_health_data(self):
        """Lưu health data"""
        data = {
            endpoint: asdict(health)
            for endpoint, health in self.health_data.items()
        }
        with open(API_MONITOR_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _get_or_create_health(self, endpoint: str) -> APIHealth:
        """Lấy hoặc tạo health record"""
        if endpoint not in self.health_data:
            self.health_data[endpoint] = APIHealth(
                endpoint=endpoint,
                status=APIStatus.UNKNOWN.value,
                last_success=None,
                last_failure=None,
                consecutive_failures=0,
                total_requests=0,
                total_failures=0,
                avg_response_time=0.0,
                last_response_time=None,
                last_error=None
            )
        return self.health_data[endpoint]
    
    def ping(
        self,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        timeout: int = 10,
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> tuple[bool, Optional[float], Optional[str]]:
        """
        Ping API endpoint với retry
        
        Returns:
            (success, response_time, error_message)
        """
        health = self._get_or_create_health(endpoint)
        health.total_requests += 1
        
        last_error = None
        response_time = None
        
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = requests.get(
                        endpoint,
                        params=params,
                        timeout=timeout
                    )
                else:
                    response = requests.request(
                        method,
                        endpoint,
                        params=params,
                        timeout=timeout
                    )
                
                response_time = time.time() - start_time
                response.raise_for_status()
                
                # Success
                health.last_success = datetime.now().isoformat()
                health.consecutive_failures = 0
                health.last_response_time = response_time
                
                # Update avg response time (exponential moving average)
                if health.avg_response_time == 0:
                    health.avg_response_time = response_time
                else:
                    health.avg_response_time = (health.avg_response_time * 0.8) + (response_time * 0.2)
                
                # Update status
                health.status = APIStatus.HEALTHY.value
                
                self._save_health_data()
                return True, response_time, None
                
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout}s"
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e}"
            except Exception as e:
                last_error = str(e)
            
            # Retry với delay
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
        
        # All retries failed
        health.last_failure = datetime.now().isoformat()
        health.consecutive_failures += 1
        health.total_failures += 1
        health.last_error = last_error
        health.last_response_time = response_time
        
        # Update status based on failure rate
        failure_rate = (health.total_failures / health.total_requests * 100) if health.total_requests > 0 else 0
        
        if health.consecutive_failures >= self.max_consecutive_failures:
            health.status = APIStatus.DOWN.value
        elif failure_rate >= self.failure_threshold_pct:
            health.status = APIStatus.DEGRADED.value
        else:
            health.status = APIStatus.DEGRADED.value
        
        self._save_health_data()
        
        # Trigger alert nếu cần
        if health.consecutive_failures >= self.max_consecutive_failures:
            self._trigger_alert(endpoint, health)
        
        return False, response_time, last_error
    
    def _trigger_alert(self, endpoint: str, health: APIHealth):
        """Trigger alert khi API fail nhiều lần"""
        if self.alert_callback:
            try:
                message = (
                    f"🚨 API Alert: {endpoint}\n"
                    f"Status: {health.status}\n"
                    f"Consecutive failures: {health.consecutive_failures}\n"
                    f"Last error: {health.last_error}\n"
                    f"Failure rate: {health.total_failures}/{health.total_requests} "
                    f"({health.total_failures/health.total_requests*100:.1f}%)"
                )
                self.alert_callback(message)
            except Exception as e:
                print(f"⚠️ Lỗi trigger alert: {e}")
    
    def get_health(self, endpoint: str) -> Optional[APIHealth]:
        """Lấy health status của endpoint"""
        return self.health_data.get(endpoint)
    
    def get_all_health(self) -> Dict[str, APIHealth]:
        """Lấy health status của tất cả endpoints"""
        return self.health_data.copy()
    
    def is_healthy(self, endpoint: str) -> bool:
        """Kiểm tra endpoint có healthy không"""
        health = self.get_health(endpoint)
        if not health:
            return True  # Unknown = assume healthy
        
        return health.status == APIStatus.HEALTHY.value
    
    def get_status_summary(self) -> Dict:
        """Lấy summary của tất cả APIs"""
        total = len(self.health_data)
        healthy = sum(1 for h in self.health_data.values() if h.status == APIStatus.HEALTHY.value)
        degraded = sum(1 for h in self.health_data.values() if h.status == APIStatus.DEGRADED.value)
        down = sum(1 for h in self.health_data.values() if h.status == APIStatus.DOWN.value)
        
        return {
            'total': total,
            'healthy': healthy,
            'degraded': degraded,
            'down': down,
            'health_rate': (healthy / total * 100) if total > 0 else 0
        }
    
    def reset_endpoint(self, endpoint: str):
        """Reset health data cho endpoint (sau khi fix)"""
        if endpoint in self.health_data:
            health = self.health_data[endpoint]
            health.consecutive_failures = 0
            health.status = APIStatus.HEALTHY.value
            health.last_error = None
            self._save_health_data()


# Global instance
_api_monitor = None

def _default_alert_callback(message: str):
    """Default alert callback - gửi qua Telegram nếu có"""
    try:
        from config import TELEGRAM_TOKEN, CHAT_ID
        if TELEGRAM_TOKEN and CHAT_ID:
            from telegram import Bot
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception:
        # Fallback to print
        print(f"🚨 {message}")

def get_api_monitor(alert_callback: Optional[Callable] = None) -> APIMonitor:
    """Get or create API monitor instance"""
    global _api_monitor
    if _api_monitor is None:
        callback = alert_callback or _default_alert_callback
        _api_monitor = APIMonitor(alert_callback=callback)
    return _api_monitor

