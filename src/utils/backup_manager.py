# -*- coding: utf-8 -*-
"""
Database Backup Manager
Automatic backup with cloud storage support
"""
import gzip
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage database backups"""

    def __init__(
        self,
        db_path: str = "trading.db",
        backup_dir: str = "backups",
        retention_days: int = 30,
        compress: bool = True,
    ):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.compress = compress

        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self) -> Optional[str]:
        """
        Create database backup

        Returns:
            Path to backup file or None if failed
        """
        try:
            if not self.db_path.exists():
                logger.error(f"Database file not found: {self.db_path}")
                return None

            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"trading_{timestamp}.db"

            if self.compress:
                backup_name += ".gz"
                backup_path = self.backup_dir / backup_name

                # Compress and copy
                with open(self.db_path, "rb") as f_in:
                    with gzip.open(backup_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                backup_path = self.backup_dir / backup_name
                shutil.copy2(self.db_path, backup_path)

            logger.info(f"✅ Backup created: {backup_path}")

            # Cleanup old backups
            self.cleanup_old_backups()

            return str(backup_path)

        except Exception:
            logger.error("❌ Backup failed", exc_info=True)
            return None

    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            for backup_file in self.backup_dir.glob("trading_*.db*"):
                # Extract timestamp from filename
                try:
                    timestamp_str = backup_file.stem.split("_")[1]
                    if backup_file.suffix == ".gz":
                        timestamp_str = timestamp_str.replace(".db", "")

                    file_date = datetime.strptime(timestamp_str, "%Y%m%d")

                    if file_date < cutoff_date:
                        backup_file.unlink()
                        logger.info(f"🗑️ Deleted old backup: {backup_file.name}")

                except (ValueError, IndexError):
                    # Skip files with invalid format
                    continue

        except Exception:
            logger.error("Error cleaning up backups")

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore database from backup

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """
        try:
            backup_file = Path(backup_path)

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Create backup of current database
            if self.db_path.exists():
                current_backup = self.db_path.with_suffix(".db.before_restore")
                shutil.copy2(self.db_path, current_backup)
                logger.info(f"Current database backed up to: {current_backup}")

            # Restore
            if backup_file.suffix == ".gz":
                # Decompress
                with gzip.open(backup_file, "rb") as f_in:
                    with open(self.db_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_file, self.db_path)

            logger.info(f"✅ Database restored from: {backup_path}")
            return True

        except Exception:
            logger.error("❌ Restore failed", exc_info=True)
            return False

    def list_backups(self) -> list:
        """List all available backups"""
        backups = []

        for backup_file in sorted(self.backup_dir.glob("trading_*.db*"), reverse=True):
            stat = backup_file.stat()
            backups.append(
                {
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return backups

    def upload_to_cloud(self, backup_path: str, provider: str = "s3"):
        """
        Upload backup to cloud storage

        Args:
            backup_path: Path to backup file
            provider: Cloud provider ('s3', 'gcs', 'azure')
        """
        # TODO: Implement cloud upload
        # This is a placeholder for cloud storage integration

        if provider == "s3":
            # AWS S3 upload
            try:
                import boto3

                s3 = boto3.client("s3")
                bucket = os.getenv("S3_BACKUP_BUCKET")

                if bucket:
                    key = f"trading-bot-backups/{Path(backup_path).name}"
                    s3.upload_file(backup_path, bucket, key)
                    logger.info(f"✅ Uploaded to S3: s3://{bucket}/{key}")
                else:
                    logger.warning("S3_BACKUP_BUCKET not configured")

            except ImportError:
                logger.warning("boto3 not installed. Install: pip install boto3")
            except Exception:
                logger.error("S3 upload failed")

        elif provider == "gcs":
            # Google Cloud Storage upload
            try:
                from google.cloud import storage

                client = storage.Client()
                bucket_name = os.getenv("GCS_BACKUP_BUCKET")

                if bucket_name:
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(f"trading-bot-backups/{Path(backup_path).name}")
                    blob.upload_from_filename(backup_path)
                    logger.info(f"✅ Uploaded to GCS: gs://{bucket_name}/{blob.name}")
                else:
                    logger.warning("GCS_BACKUP_BUCKET not configured")

            except ImportError:
                logger.warning("google-cloud-storage not installed")
            except Exception:
                logger.error("GCS upload failed")


# Singleton
_backup_manager = None


def get_backup_manager() -> BackupManager:
    """Get backup manager singleton"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


# Scheduled backup function
def scheduled_backup():
    """Function to be called by scheduler"""
    manager = get_backup_manager()
    backup_path = manager.create_backup()

    if backup_path:
        # Upload to cloud if configured
        cloud_provider = os.getenv("CLOUD_BACKUP_PROVIDER")
        if cloud_provider:
            manager.upload_to_cloud(backup_path, cloud_provider)


if __name__ == "__main__":
    # Test backup manager
    manager = BackupManager()

    print("📦 Creating backup...")
    backup_path = manager.create_backup()

    if backup_path:
        print(f"✅ Backup created: {backup_path}")

        print("\n📋 Available backups:")
        for backup in manager.list_backups():
            print(f"  - {backup['filename']} ({backup['size_mb']:.2f} MB)")

    print("\n✅ Backup manager test complete!")
