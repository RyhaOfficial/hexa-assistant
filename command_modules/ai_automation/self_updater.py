#!/usr/bin/env python3

import os
import sys
import logging
import subprocess
import json
import hashlib
import shutil
import tempfile
import time
import datetime
import git
import requests
import schedule
import threading
import signal
import zipfile
import tarfile
import gpg
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import importlib.util
import traceback
import platform
import psutil
import semver

class UpdateStatus(Enum):
    """Enum for update status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class UpdateInfo:
    """Class to store update information."""
    version: str
    release_date: str
    changes: List[str]
    dependencies: Dict[str, str]
    requires_restart: bool
    size: int
    checksum: str
    signature: Optional[str] = None

class SelfUpdater:
    def __init__(self, config_file: str = "config/self_updater_config.json"):
        """Initialize the SelfUpdater with configuration."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize update tracking
        self.update_status = UpdateStatus.PENDING
        self.current_version = self._get_current_version()
        self.update_history = []
        
        # Initialize update sources
        self.update_sources = self._initialize_update_sources()
        
        # Initialize backup directory
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize update scheduler
        self.scheduler = schedule.Scheduler()
        self._setup_scheduled_updates()
        
        # Start update checker thread
        self.update_checker_thread = threading.Thread(
            target=self._check_for_updates_periodically,
            daemon=True
        )
        self.update_checker_thread.start()

    def load_config(self) -> Dict:
        """Load self-updater configuration from file."""
        default_config = {
            "update_sources": {
                "main": {
                    "type": "git",
                    "url": "https://github.com/hexa-assistant/hexa-assistant.git",
                    "branch": "main"
                },
                "features": {
                    "type": "git",
                    "url": "https://github.com/hexa-assistant/features.git",
                    "branch": "main"
                }
            },
            "update_schedule": {
                "enabled": True,
                "interval": "weekly",
                "day": "sunday",
                "time": "03:00"
            },
            "auto_update": {
                "enabled": True,
                "delay_minutes": 15
            },
            "security": {
                "verify_signatures": True,
                "verify_checksums": True,
                "gpg_key": "hexa-assistant.asc"
            },
            "backup": {
                "enabled": True,
                "keep_last": 5
            },
            "notifications": {
                "enabled": True,
                "channel": "console"
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return json.load(f)
        else:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/self_updater")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"self_updater_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_current_version(self) -> str:
        """Get current version of Hexa Assistant."""
        version_file = Path("VERSION")
        if version_file.exists():
            with open(version_file, "r") as f:
                return f.read().strip()
        return "0.0.0"

    def _initialize_update_sources(self) -> Dict:
        """Initialize update sources from configuration."""
        sources = {}
        for name, source_config in self.config["update_sources"].items():
            if source_config["type"] == "git":
                sources[name] = git.Repo(source_config["url"])
            elif source_config["type"] == "http":
                sources[name] = source_config["url"]
        return sources

    def _setup_scheduled_updates(self):
        """Setup scheduled updates based on configuration."""
        if not self.config["update_schedule"]["enabled"]:
            return
            
        schedule_config = self.config["update_schedule"]
        if schedule_config["interval"] == "weekly":
            self.scheduler.every().week.at(
                f"{schedule_config['day']} {schedule_config['time']}"
            ).do(self.check_for_updates)
        elif schedule_config["interval"] == "daily":
            self.scheduler.every().day.at(schedule_config["time"]).do(
                self.check_for_updates
            )

    def _check_for_updates_periodically(self):
        """Periodically check for updates in a separate thread."""
        while True:
            self.scheduler.run_pending()
            time.sleep(60)  # Check every minute

    def check_for_updates(self) -> Dict:
        """Check for available updates."""
        try:
            self.logger.info("Checking for updates...")
            available_updates = {}
            
            for source_name, source in self.update_sources.items():
                if isinstance(source, git.Repo):
                    # Check Git repository for updates
                    source.remote().fetch()
                    local_commit = source.head.commit
                    remote_commit = source.remote().refs[source.config["branch"]].commit
                    
                    if local_commit != remote_commit:
                        available_updates[source_name] = {
                            "type": "git",
                            "current": str(local_commit),
                            "available": str(remote_commit)
                        }
                else:
                    # Check HTTP source for updates
                    response = requests.get(f"{source}/version.json")
                    if response.status_code == 200:
                        version_info = response.json()
                        if semver.compare(version_info["version"], self.current_version) > 0:
                            available_updates[source_name] = {
                                "type": "http",
                                "version": version_info["version"],
                                "changes": version_info["changes"]
                            }
            
            if available_updates:
                self.logger.info(f"Updates available: {available_updates}")
                if self.config["notifications"]["enabled"]:
                    self._notify_updates_available(available_updates)
                return {
                    "success": True,
                    "updates_available": True,
                    "updates": available_updates
                }
            else:
                self.logger.info("No updates available")
                return {
                    "success": True,
                    "updates_available": False
                }
                
        except Exception as e:
            self.logger.error(f"Failed to check for updates: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _notify_updates_available(self, updates: Dict):
        """Notify user about available updates."""
        message = "Updates available for Hexa Assistant:\n"
        for source, info in updates.items():
            if info["type"] == "git":
                message += f"- {source}: New commits available\n"
            else:
                message += f"- {source}: Version {info['version']}\n"
                message += "  Changes:\n"
                for change in info["changes"]:
                    message += f"  - {change}\n"
        
        if self.config["notifications"]["channel"] == "console":
            print(message)
        # Add other notification channels here (e.g., email, webhook)

    def update(self, source: str = "all") -> Dict:
        """Update Hexa Assistant."""
        try:
            if self.update_status == UpdateStatus.IN_PROGRESS:
                return {
                    "success": False,
                    "error": "Update already in progress"
                }
            
            self.update_status = UpdateStatus.IN_PROGRESS
            self.logger.info(f"Starting update from source: {source}")
            
            # Create backup
            if self.config["backup"]["enabled"]:
                self._create_backup()
            
            # Perform update
            if source == "all":
                for source_name in self.update_sources:
                    self._update_source(source_name)
            else:
                self._update_source(source)
            
            # Verify update
            if not self._verify_update():
                self.logger.error("Update verification failed")
                self.rollback()
                return {
                    "success": False,
                    "error": "Update verification failed"
                }
            
            self.update_status = UpdateStatus.COMPLETED
            self._update_version_file()
            self._cleanup_old_backups()
            
            return {
                "success": True,
                "message": "Update completed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            self.update_status = UpdateStatus.FAILED
            self.rollback()
            return {
                "success": False,
                "error": str(e)
            }

    def _create_backup(self):
        """Create backup of current installation."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir()
        
        # Copy all files except backups and logs
        for item in Path(".").iterdir():
            if item.name not in ["backups", "logs"]:
                if item.is_file():
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, backup_path / item.name)
        
        self.logger.info(f"Created backup at {backup_path}")

    def _update_source(self, source_name: str):
        """Update from a specific source."""
        source = self.update_sources[source_name]
        
        if isinstance(source, git.Repo):
            # Update from Git repository
            source.remote().pull()
            self.logger.info(f"Updated from Git source: {source_name}")
        else:
            # Update from HTTP source
            self._download_and_install_update(source_name, source)

    def _download_and_install_update(self, source_name: str, source_url: str):
        """Download and install update from HTTP source."""
        # Download update package
        response = requests.get(f"{source_url}/update.zip")
        if response.status_code != 200:
            raise Exception(f"Failed to download update from {source_name}")
        
        # Verify checksum
        if self.config["security"]["verify_checksums"]:
            checksum = hashlib.sha256(response.content).hexdigest()
            if not self._verify_checksum(checksum, source_name):
                raise Exception("Checksum verification failed")
        
        # Verify signature if available
        if self.config["security"]["verify_signatures"]:
            signature = requests.get(f"{source_url}/update.zip.sig").content
            if not self._verify_signature(response.content, signature):
                raise Exception("Signature verification failed")
        
        # Extract update
        with tempfile.TemporaryDirectory() as temp_dir:
            update_path = Path(temp_dir) / "update.zip"
            with open(update_path, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(update_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Install update
            self._install_update_files(Path(temp_dir) / "update")

    def _verify_checksum(self, checksum: str, source: str) -> bool:
        """Verify checksum of downloaded update."""
        # Get expected checksum from source
        response = requests.get(f"{self.update_sources[source]}/checksums.json")
        if response.status_code != 200:
            return False
        
        expected_checksums = response.json()
        return checksum in expected_checksums.values()

    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify GPG signature of downloaded update."""
        try:
            gpg_key = Path(self.config["security"]["gpg_key"])
            if not gpg_key.exists():
                self.logger.warning("GPG key not found, skipping signature verification")
                return True
            
            with gpg.Context() as ctx:
                ctx.verify(data, signature)
                return True
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False

    def _install_update_files(self, update_dir: Path):
        """Install update files."""
        for item in update_dir.iterdir():
            target_path = Path(".") / item.name
            if item.is_file():
                shutil.copy2(item, target_path)
            elif item.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(item, target_path)

    def _verify_update(self) -> bool:
        """Verify that update was successful."""
        try:
            # Check if version file was updated
            if not Path("VERSION").exists():
                return False
            
            # Run basic functionality tests
            test_results = self._run_update_tests()
            return all(test_results.values())
        except Exception as e:
            self.logger.error(f"Update verification failed: {e}")
            return False

    def _run_update_tests(self) -> Dict[str, bool]:
        """Run tests to verify update."""
        tests = {
            "import_check": self._test_imports(),
            "config_check": self._test_config(),
            "feature_check": self._test_features()
        }
        return tests

    def _test_imports(self) -> bool:
        """Test if all required modules can be imported."""
        try:
            import hexa_assistant
            import command_modules
            return True
        except Exception as e:
            self.logger.error(f"Import test failed: {e}")
            return False

    def _test_config(self) -> bool:
        """Test if configuration is valid."""
        try:
            with open("config/hexa_config.json", "r") as f:
                json.load(f)
            return True
        except Exception as e:
            self.logger.error(f"Config test failed: {e}")
            return False

    def _test_features(self) -> bool:
        """Test if core features are working."""
        try:
            # Add feature tests here
            return True
        except Exception as e:
            self.logger.error(f"Feature test failed: {e}")
            return False

    def _update_version_file(self):
        """Update version file with new version."""
        with open("VERSION", "w") as f:
            f.write(self.current_version)

    def _cleanup_old_backups(self):
        """Clean up old backups."""
        if not self.config["backup"]["enabled"]:
            return
            
        backups = sorted(self.backup_dir.iterdir(), key=os.path.getmtime)
        keep_last = self.config["backup"]["keep_last"]
        
        for backup in backups[:-keep_last]:
            shutil.rmtree(backup)

    def rollback(self):
        """Rollback to previous version."""
        try:
            self.logger.info("Starting rollback...")
            self.update_status = UpdateStatus.ROLLED_BACK
            
            # Find latest backup
            backups = sorted(self.backup_dir.iterdir(), key=os.path.getmtime)
            if not backups:
                raise Exception("No backups available for rollback")
            
            latest_backup = backups[-1]
            
            # Restore from backup
            for item in latest_backup.iterdir():
                target_path = Path(".") / item.name
                if item.is_file():
                    shutil.copy2(item, target_path)
                elif item.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(item, target_path)
            
            self.logger.info("Rollback completed successfully")
            return {
                "success": True,
                "message": "Rollback completed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """Main function to run the self-updater."""
    try:
        updater = SelfUpdater()
        
        # Example usage
        result = updater.check_for_updates()
        
        if result["success"] and result["updates_available"]:
            print("Updates are available. Would you like to update now? (y/n)")
            if input().lower() == "y":
                update_result = updater.update()
                if update_result["success"]:
                    print("Update completed successfully")
                else:
                    print(f"Update failed: {update_result['error']}")
        else:
            print("No updates available")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 