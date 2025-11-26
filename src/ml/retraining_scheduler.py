# -*- coding: utf-8 -*-
"""
Retraining Scheduler
Background service để monitor và trigger automated retraining
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import schedule

from src.ml.retraining_pipeline import (
    AutomatedRetrainingPipeline,
    RetrainingConfig,
    RetrainingResult,
    RetrainingTrigger,
    get_retraining_pipeline,
)

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Background scheduler for automated model retraining

    FEATURES:
    - Periodic checks (hourly, daily, weekly)
    - Drift monitoring
    - Performance monitoring
    - Telegram notifications
    - Status dashboard
    - Emergency stop
    """

    def __init__(
        self,
        pipeline: Optional[AutomatedRetrainingPipeline] = None,
        check_interval_hours: int = 6,  # Check every 6 hours
        enable_notifications: bool = True,
        status_file: str = "models/retraining_status.json",
        telegram_bot=None,
        telegram_chat_id: Optional[str] = None,
    ):
        """
        Args:
            pipeline: Retraining pipeline instance
            check_interval_hours: Hours between checks
            enable_notifications: Send Telegram notifications
            status_file: JSON file for status tracking
            telegram_bot: Telegram bot instance
            telegram_chat_id: Telegram chat ID for notifications
        """
        self.pipeline = pipeline or get_retraining_pipeline()
        self.check_interval_hours = check_interval_hours
        self.enable_notifications = enable_notifications
        self.status_file = Path(status_file)
        self.telegram_bot = telegram_bot
        self.telegram_chat_id = telegram_chat_id

        # State
        self.is_running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()

        # Status tracking
        self.last_check_time = None
        self.last_retraining_time = None
        self.total_retrainings = 0
        self.last_retraining_result = None

        # Load status
        self._load_status()

    def start(self):
        """Start the scheduler in background thread"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        logger.info(f"🚀 Starting retraining scheduler (check every {self.check_interval_hours}h)")

        # Schedule periodic checks
        schedule.every(self.check_interval_hours).hours.do(self._run_check)

        # Start background thread
        self.is_running = True
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("✅ Retraining scheduler started")

        # Send notification
        self._send_notification(
            "🚀 **Retraining Scheduler Started**\n\n"
            f"Check interval: {self.check_interval_hours} hours\n"
            f"Next check: {schedule.next_run()}"
        )

    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler not running")
            return

        logger.info("🛑 Stopping retraining scheduler...")

        self.is_running = False
        self.stop_event.set()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        schedule.clear()

        logger.info("✅ Retraining scheduler stopped")

        # Send notification
        self._send_notification("🛑 **Retraining Scheduler Stopped**")

    def _run_scheduler(self):
        """Background scheduler loop"""
        while not self.stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                time.sleep(60)

    def _run_check(self):
        """Run retraining check"""
        try:
            logger.info("🔍 Running automated retraining check...")
            self.last_check_time = datetime.now()

            # Check triggers
            # NOTE: You need to provide current_performance and current_features
            # This is a placeholder - implement based on your monitoring system
            should_retrain, trigger, reason = self.pipeline.check_triggers()

            if should_retrain:
                logger.info(f"✅ Retraining triggered: {reason}")

                # Send notification
                self._send_notification(
                    f"⚠️ **Retraining Triggered**\n\n"
                    f"Trigger: {trigger.value}\n"
                    f"Reason: {reason}\n\n"
                    f"Starting automated retraining..."
                )

                # Run retraining
                result = self.pipeline.run_retraining(trigger, reason)

                # Update state
                self.last_retraining_time = datetime.now()
                self.last_retraining_result = result
                self.total_retrainings += 1

                # Save status
                self._save_status()

                # Send result notification
                self._send_retraining_result_notification(result)

            else:
                logger.info("✅ No retraining needed")

                # Save status
                self._save_status()

        except Exception as e:
            logger.error(f"Error in retraining check: {e}", exc_info=True)

            # Send error notification
            self._send_notification(
                f"❌ **Retraining Check Error**\n\n"
                f"Error: {str(e)}\n\n"
                f"Please investigate immediately!"
            )

    def manual_trigger(
        self,
        reason: str = "Manual trigger"
    ) -> RetrainingResult:
        """
        Manually trigger retraining

        Args:
            reason: Reason for manual trigger

        Returns:
            RetrainingResult
        """
        logger.info(f"🔧 Manual retraining triggered: {reason}")

        # Send notification
        self._send_notification(
            f"🔧 **Manual Retraining Triggered**\n\n"
            f"Reason: {reason}\n\n"
            f"Starting retraining..."
        )

        # Run retraining
        result = self.pipeline.run_retraining(
            RetrainingTrigger.MANUAL,
            reason
        )

        # Update state
        self.last_retraining_time = datetime.now()
        self.last_retraining_result = result
        self.total_retrainings += 1

        # Save status
        self._save_status()

        # Send result
        self._send_retraining_result_notification(result)

        return result

    def get_status(self) -> dict:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_retraining_time': self.last_retraining_time.isoformat() if self.last_retraining_time else None,
            'total_retrainings': self.total_retrainings,
            'next_check': str(schedule.next_run()) if schedule.jobs else None,
            'last_result': self._result_to_dict(self.last_retraining_result) if self.last_retraining_result else None,
        }

    def _send_notification(self, message: str):
        """Send Telegram notification"""
        if not self.enable_notifications:
            return

        if self.telegram_bot is None or self.telegram_chat_id is None:
            logger.debug("Telegram not configured - skipping notification")
            return

        try:
            import asyncio
            asyncio.create_task(
                self.telegram_bot.send_message(
                    self.telegram_chat_id,
                    message,
                    parse_mode="Markdown"
                )
            )
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")

    def _send_retraining_result_notification(self, result: RetrainingResult):
        """Send notification about retraining result"""
        if result.training_successful:
            message = (
                f"✅ **Retraining Successful**\n\n"
                f"Trigger: {result.trigger.value}\n"
                f"New model: {result.new_model_id} (v{result.new_version})\n\n"
                f"**Performance:**\n"
                f"Train: {result.train_accuracy:.1%} (AUC: {result.train_auc:.3f})\n"
                f"Val: {result.val_accuracy:.1%} (AUC: {result.val_auc:.3f})\n"
                f"Test: {result.test_accuracy:.1%} (AUC: {result.test_auc:.3f})\n\n"
            )

            if result.current_model_id:
                message += f"Improvement: {result.improvement_pct:+.1f}%\n"

            if result.deployed:
                message += "\n🚀 **Model Auto-Deployed**"
            else:
                message += "\nℹ️ Model registered but not deployed (manual activation required)"

        else:
            message = (
                f"❌ **Retraining Failed**\n\n"
                f"Trigger: {result.trigger.value}\n"
                f"Errors:\n"
            )
            for error in result.errors:
                message += f"• {error}\n"

        self._send_notification(message)

    def _result_to_dict(self, result: RetrainingResult) -> dict:
        """Convert RetrainingResult to dict"""
        return {
            'trigger': result.trigger.value,
            'trigger_reason': result.trigger_reason,
            'timestamp': result.timestamp,
            'training_successful': result.training_successful,
            'new_model_id': result.new_model_id,
            'new_version': result.new_version,
            'val_accuracy': result.val_accuracy,
            'test_accuracy': result.test_accuracy,
            'improvement_pct': result.improvement_pct,
            'deployed': result.deployed,
            'errors': result.errors,
        }

    def _load_status(self):
        """Load status from file"""
        if not self.status_file.exists():
            return

        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)

            self.last_check_time = datetime.fromisoformat(status['last_check_time']) if status.get('last_check_time') else None
            self.last_retraining_time = datetime.fromisoformat(status['last_retraining_time']) if status.get('last_retraining_time') else None
            self.total_retrainings = status.get('total_retrainings', 0)

        except Exception as e:
            logger.error(f"Error loading status: {e}")

    def _save_status(self):
        """Save status to file"""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)

            status = {
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'last_retraining_time': self.last_retraining_time.isoformat() if self.last_retraining_time else None,
                'total_retrainings': self.total_retrainings,
                'last_result': self._result_to_dict(self.last_retraining_result) if self.last_retraining_result else None,
            }

            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving status: {e}")


# Singleton instance
_scheduler = None


def get_retraining_scheduler() -> RetrainingScheduler:
    """Get singleton instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = RetrainingScheduler()
    return _scheduler
