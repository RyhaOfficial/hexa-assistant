#!/usr/bin/env python3

import os
import sys
import json
import time
import hashlib
import logging
import platform
import subprocess
import datetime
import threading
import queue
import shutil
import tempfile
import gzip
import base64
import socket
import struct
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

# Forensics-specific imports
import psutil
import yara
import magic
import pefile
import pytsk3
import volatility3
from volatility3.framework import interfaces
from volatility3.plugins import windows
from volatility3.plugins import linux
from volatility3.plugins import mac
import pyshark
import evtx
import winreg
import capstone
import keystone
import unicorn
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# AI and analysis imports
from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class ForensicsMode(Enum):
    SILENT = auto()
    INTERACTIVE = auto()
    EMERGENCY = auto()

@dataclass
class ArtifactInfo:
    """Information about a collected forensic artifact"""
    path: str
    type: str
    hash: str
    size: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidencePackage:
    """Container for collected forensic evidence"""
    timestamp: str
    hostname: str
    os_info: Dict[str, str]
    artifacts: List[ArtifactInfo]
    hashes: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ForensicsEngine:
    def __init__(self, config_path: str = "config/forensics_config.json"):
        """Initialize the forensics engine with configuration"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.evidence_dir = self._create_evidence_dir()
        self.mode = ForensicsMode.INTERACTIVE
        self.ai_analyzer = self._init_ai_analyzer()
        self.artifact_queue = queue.Queue()
        self.collection_threads = []
        self.stop_collection = threading.Event()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load forensics configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Create default forensics configuration"""
        return {
            "evidence_dir": "evidence",
            "max_threads": 4,
            "compression": True,
            "encryption": False,
            "encryption_key": None,
            "ioc_rules": "rules/ioc_rules.yar",
            "yara_rules": "rules/yara_rules.yar",
            "memory_dump": {
                "full_dump": True,
                "filter_processes": [],
                "max_size": "4GB"
            },
            "disk_analysis": {
                "recover_deleted": True,
                "scan_sectors": True,
                "hash_files": True
            },
            "network_capture": {
                "duration": 300,
                "interface": "any",
                "filter": ""
            },
            "reporting": {
                "format": "html",
                "include_hashes": True,
                "include_timeline": True
            }
        }
        
    def setup_logging(self):
        """Configure logging for forensics operations"""
        log_dir = "logs/forensics"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/forensics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ForensicsEngine")
        
    def _create_evidence_dir(self) -> str:
        """Create timestamped evidence directory"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_dir = f"evidence/forensics_{timestamp}"
        os.makedirs(evidence_dir, exist_ok=True)
        return evidence_dir
        
    def _init_ai_analyzer(self):
        """Initialize AI models for behavioral analysis"""
        try:
            return {
                "summarizer": pipeline("summarization"),
                "classifier": pipeline("text-classification"),
                "anomaly_detector": IsolationForest(contamination=0.1)
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize AI analyzer: {e}")
            return None
            
    def capture_artifacts(self) -> EvidencePackage:
        """Capture complete system snapshot and artifacts"""
        self.logger.info("Starting artifact collection...")
        
        evidence = EvidencePackage(
            timestamp=datetime.datetime.now().isoformat(),
            hostname=socket.gethostname(),
            os_info=self._get_os_info(),
            artifacts=[],
            hashes={}
        )
        
        # Start collection threads
        collectors = [
            self._collect_processes,
            self._collect_network,
            self._collect_files,
            self._collect_services,
            self._collect_scheduled_tasks
        ]
        
        with ThreadPoolExecutor(max_workers=self.config["max_threads"]) as executor:
            futures = [executor.submit(collector) for collector in collectors]
            for future in as_completed(futures):
                try:
                    artifacts = future.result()
                    evidence.artifacts.extend(artifacts)
                except Exception as e:
                    self.logger.error(f"Collection error: {e}")
                    
        # Hash and store artifacts
        self._process_artifacts(evidence)
        
        return evidence
        
    def _get_os_info(self) -> Dict[str, str]:
        """Get detailed OS information"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
    def _collect_processes(self) -> List[ArtifactInfo]:
        """Collect information about running processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'create_time']):
            try:
                pinfo = proc.info
                processes.append(ArtifactInfo(
                    path=f"processes/{pinfo['pid']}_{pinfo['name']}.json",
                    type="process",
                    hash="",  # Will be computed later
                    size=0,
                    timestamp=datetime.datetime.fromtimestamp(pinfo['create_time']).isoformat(),
                    metadata=pinfo
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
        
    def _collect_network(self) -> List[ArtifactInfo]:
        """Collect network connections and configurations"""
        network_info = []
        
        # Collect active connections
        for conn in psutil.net_connections(kind='inet'):
            try:
                network_info.append(ArtifactInfo(
                    path=f"network/connections/{conn.pid}_{conn.laddr.port}.json",
                    type="network_connection",
                    hash="",
                    size=0,
                    timestamp=datetime.datetime.now().isoformat(),
                    metadata={
                        "pid": conn.pid,
                        "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}",
                        "remote_addr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        "status": conn.status
                    }
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Collect network interfaces
        for iface, addrs in psutil.net_if_addrs().items():
            network_info.append(ArtifactInfo(
                path=f"network/interfaces/{iface}.json",
                type="network_interface",
                hash="",
                size=0,
                timestamp=datetime.datetime.now().isoformat(),
                metadata={"interface": iface, "addresses": [str(addr) for addr in addrs]}
            ))
            
        return network_info
        
    def _collect_files(self) -> List[ArtifactInfo]:
        """Collect information about open files and important system files"""
        files = []
        
        # Collect open files
        for proc in psutil.process_iter(['pid', 'open_files']):
            try:
                for file in proc.info['open_files']:
                    files.append(ArtifactInfo(
                        path=f"files/open/{proc.info['pid']}_{os.path.basename(file.path)}.json",
                        type="open_file",
                        hash="",
                        size=os.path.getsize(file.path) if os.path.exists(file.path) else 0,
                        timestamp=datetime.datetime.fromtimestamp(os.path.getmtime(file.path)).isoformat(),
                        metadata={"pid": proc.info['pid'], "path": file.path}
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return files
        
    def _collect_services(self) -> List[ArtifactInfo]:
        """Collect information about installed services"""
        services = []
        
        if platform.system() == "Windows":
            # Windows services
            import win32serviceutil
            import win32service
            
            for service in win32serviceutil.EnumServices(None, None, win32service.SERVICE_WIN32):
                try:
                    service_info = win32serviceutil.QueryServiceConfig(service[1])
                    services.append(ArtifactInfo(
                        path=f"services/windows/{service[1]}.json",
                        type="windows_service",
                        hash="",
                        size=0,
                        timestamp=datetime.datetime.now().isoformat(),
                        metadata={
                            "name": service[1],
                            "display_name": service[0],
                            "binary_path": service_info[3],
                            "start_type": service_info[1]
                        }
                    ))
                except Exception as e:
                    self.logger.error(f"Error collecting service info: {e}")
        else:
            # Linux/Unix services
            service_dirs = ["/etc/init.d", "/etc/systemd/system"]
            for service_dir in service_dirs:
                if os.path.exists(service_dir):
                    for service_file in os.listdir(service_dir):
                        service_path = os.path.join(service_dir, service_file)
                        if os.path.isfile(service_path):
                            services.append(ArtifactInfo(
                                path=f"services/unix/{service_file}.json",
                                type="unix_service",
                                hash="",
                                size=os.path.getsize(service_path),
                                timestamp=datetime.datetime.fromtimestamp(os.path.getmtime(service_path)).isoformat(),
                                metadata={"path": service_path}
                            ))
                            
        return services
        
    def _collect_scheduled_tasks(self) -> List[ArtifactInfo]:
        """Collect information about scheduled tasks and cron jobs"""
        tasks = []
        
        if platform.system() == "Windows":
            # Windows scheduled tasks
            import win32com.client
            scheduler = win32com.client.Dispatch('Schedule.Service')
            scheduler.Connect()
            root_folder = scheduler.GetFolder("\\")
            tasks_list = root_folder.GetTasks(0)
            
            for task in tasks_list:
                try:
                    task_info = task.Definition
                    tasks.append(ArtifactInfo(
                        path=f"tasks/windows/{task.Name}.json",
                        type="windows_task",
                        hash="",
                        size=0,
                        timestamp=datetime.datetime.now().isoformat(),
                        metadata={
                            "name": task.Name,
                            "path": task_info.Path,
                            "command": task_info.Actions[0].Execute
                        }
                    ))
                except Exception as e:
                    self.logger.error(f"Error collecting task info: {e}")
        else:
            # Linux/Unix cron jobs
            cron_dirs = ["/etc/cron.d", "/etc/cron.daily", "/etc/cron.hourly", "/etc/cron.weekly", "/etc/cron.monthly"]
            for cron_dir in cron_dirs:
                if os.path.exists(cron_dir):
                    for cron_file in os.listdir(cron_dir):
                        cron_path = os.path.join(cron_dir, cron_file)
                        if os.path.isfile(cron_path):
                            tasks.append(ArtifactInfo(
                                path=f"tasks/unix/{cron_file}.json",
                                type="unix_cron",
                                hash="",
                                size=os.path.getsize(cron_path),
                                timestamp=datetime.datetime.fromtimestamp(os.path.getmtime(cron_path)).isoformat(),
                                metadata={"path": cron_path}
                            ))
                            
        return tasks
        
    def _process_artifacts(self, evidence: EvidencePackage):
        """Process collected artifacts (hash, compress, encrypt)"""
        for artifact in evidence.artifacts:
            try:
                # Create directory structure
                artifact_dir = os.path.join(self.evidence_dir, os.path.dirname(artifact.path))
                os.makedirs(artifact_dir, exist_ok=True)
                
                # Save artifact data
                artifact_path = os.path.join(self.evidence_dir, artifact.path)
                with open(artifact_path, 'w') as f:
                    json.dump(artifact.metadata, f, indent=2)
                    
                # Compute hash
                with open(artifact_path, 'rb') as f:
                    artifact.hash = hashlib.sha256(f.read()).hexdigest()
                    evidence.hashes[artifact.path] = artifact.hash
                    
                # Compress if enabled
                if self.config["compression"]:
                    with open(artifact_path, 'rb') as f_in:
                        with gzip.open(f"{artifact_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(artifact_path)
                    artifact.path = f"{artifact.path}.gz"
                    
                # Encrypt if enabled
                if self.config["encryption"] and self.config["encryption_key"]:
                    self._encrypt_file(artifact_path)
                    
            except Exception as e:
                self.logger.error(f"Error processing artifact {artifact.path}: {e}")
                
    def _encrypt_file(self, file_path: str):
        """Encrypt a file using AES-256"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            salt = get_random_bytes(16)
            key = PBKDF2(self.config["encryption_key"], salt, dkLen=32)
            cipher = AES.new(key, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            
            with open(f"{file_path}.enc", 'wb') as f:
                f.write(salt + cipher.nonce + tag + ciphertext)
                
            os.remove(file_path)
            
        except Exception as e:
            self.logger.error(f"Error encrypting file {file_path}: {e}")
            
    def dump_memory(self, full_dump: bool = True) -> str:
        """Capture memory dump using appropriate tool"""
        self.logger.info("Starting memory dump...")
        
        if platform.system() == "Windows":
            return self._dump_windows_memory(full_dump)
        else:
            return self._dump_unix_memory(full_dump)
            
    def _dump_windows_memory(self, full_dump: bool) -> str:
        """Dump Windows memory using appropriate tool"""
        try:
            # Try using volatility's native memory capture
            dump_path = os.path.join(self.evidence_dir, "memory.raw")
            subprocess.run(["volatility", "-f", "memory.raw", "imageinfo"], check=True)
            return dump_path
        except Exception as e:
            self.logger.error(f"Error dumping Windows memory: {e}")
            return ""
            
    def _dump_unix_memory(self, full_dump: bool) -> str:
        """Dump Unix memory using /proc/kcore or /dev/mem"""
        try:
            dump_path = os.path.join(self.evidence_dir, "memory.raw")
            if os.path.exists("/proc/kcore"):
                shutil.copy("/proc/kcore", dump_path)
            elif os.path.exists("/dev/mem"):
                with open("/dev/mem", "rb") as src, open(dump_path, "wb") as dst:
                    dst.write(src.read())
            return dump_path
        except Exception as e:
            self.logger.error(f"Error dumping Unix memory: {e}")
            return ""
            
    def analyze_memory(self, memory_dump: str) -> Dict[str, Any]:
        """Analyze memory dump using volatility"""
        self.logger.info("Analyzing memory dump...")
        
        results = {}
        try:
            # Initialize volatility
            ctx = interfaces.context.Context()
            ctx.config['automagic.LayerStacker.single_location'] = memory_dump
            
            # Run various volatility plugins
            plugins = [
                ("windows.pslist", "Process List"),
                ("windows.netscan", "Network Connections"),
                ("windows.cmdline", "Command Line"),
                ("windows.filescan", "File Objects"),
                ("windows.registry", "Registry")
            ]
            
            for plugin_name, description in plugins:
                try:
                    plugin = volatility3.plugins.get_plugin(plugin_name)
                    results[description] = plugin.run(ctx)
                except Exception as e:
                    self.logger.error(f"Error running plugin {plugin_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing memory: {e}")
            
        return results
        
    def analyze_disk(self, disk_path: str) -> Dict[str, Any]:
        """Analyze disk for forensic artifacts"""
        self.logger.info("Analyzing disk...")
        
        results = {
            "deleted_files": [],
            "timeline": [],
            "suspicious_files": []
        }
        
        try:
            # Initialize TSK
            img_info = pytsk3.Img_Info(disk_path)
            fs_info = pytsk3.FS_Info(img_info)
            
            # Recover deleted files
            if self.config["disk_analysis"]["recover_deleted"]:
                self._recover_deleted_files(fs_info, results)
                
            # Build timeline
            self._build_timeline(fs_info, results)
            
            # Scan for suspicious files
            self._scan_suspicious_files(fs_info, results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing disk: {e}")
            
        return results
        
    def _recover_deleted_files(self, fs_info: pytsk3.FS_Info, results: Dict):
        """Recover deleted files from filesystem"""
        try:
            for file_entry in fs_info.open_dir(path="/"):
                try:
                    if file_entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_REG:
                        if file_entry.info.meta.flags & pytsk3.TSK_FS_META_FLAG_UNALLOC:
                            # File is deleted
                            file_data = file_entry.read_random(0, file_entry.info.meta.size)
                            recovered_path = os.path.join(self.evidence_dir, "recovered", file_entry.info.name.name.decode())
                            os.makedirs(os.path.dirname(recovered_path), exist_ok=True)
                            
                            with open(recovered_path, "wb") as f:
                                f.write(file_data)
                                
                            results["deleted_files"].append({
                                "name": file_entry.info.name.name.decode(),
                                "size": file_entry.info.meta.size,
                                "recovered_path": recovered_path
                            })
                except Exception as e:
                    self.logger.error(f"Error recovering file: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in deleted file recovery: {e}")
            
    def _build_timeline(self, fs_info: pytsk3.FS_Info, results: Dict):
        """Build filesystem timeline"""
        try:
            for file_entry in fs_info.open_dir(path="/"):
                try:
                    if file_entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_REG:
                        results["timeline"].append({
                            "name": file_entry.info.name.name.decode(),
                            "created": file_entry.info.meta.crtime,
                            "modified": file_entry.info.meta.mtime,
                            "accessed": file_entry.info.meta.atime,
                            "size": file_entry.info.meta.size
                        })
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error building timeline: {e}")
            
    def _scan_suspicious_files(self, fs_info: pytsk3.FS_Info, results: Dict):
        """Scan for suspicious files using YARA rules"""
        try:
            rules = yara.compile(self.config["yara_rules"])
            
            for file_entry in fs_info.open_dir(path="/"):
                try:
                    if file_entry.info.meta.type == pytsk3.TSK_FS_META_TYPE_REG:
                        file_data = file_entry.read_random(0, file_entry.info.meta.size)
                        matches = rules.match(data=file_data)
                        
                        if matches:
                            results["suspicious_files"].append({
                                "name": file_entry.info.name.name.decode(),
                                "size": file_entry.info.meta.size,
                                "matches": [match.rule for match in matches]
                            })
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning suspicious files: {e}")
            
    def extract_logs(self) -> Dict[str, List[Dict]]:
        """Extract and parse system logs"""
        self.logger.info("Extracting system logs...")
        
        logs = {
            "system": [],
            "security": [],
            "application": []
        }
        
        try:
            if platform.system() == "Windows":
                # Windows Event Logs
                import win32evtlog
                import win32evtlogutil
                import win32con
                
                server = 'localhost'
                logtype = 'System'
                hand = win32evtlog.OpenEventLog(server, logtype)
                total = win32evtlog.GetNumberOfEventLogRecords(hand)
                
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ|win32evtlog.EVENTLOG_SEQUENTIAL_READ
                events = []
                
                while True:
                    events = win32evtlog.ReadEventLog(hand, flags, 0)
                    if not events:
                        break
                        
                    for event in events:
                        logs["system"].append({
                            "timestamp": event.TimeGenerated.isoformat(),
                            "source": event.SourceName,
                            "event_id": event.EventID,
                            "message": win32evtlogutil.SafeFormatMessage(event, logtype)
                        })
                        
            else:
                # Linux/Unix logs
                log_files = {
                    "system": ["/var/log/syslog", "/var/log/messages"],
                    "security": ["/var/log/auth.log", "/var/log/secure"],
                    "application": ["/var/log/application.log"]
                }
                
                for log_type, files in log_files.items():
                    for log_file in files:
                        if os.path.exists(log_file):
                            with open(log_file, 'r') as f:
                                for line in f:
                                    logs[log_type].append({
                                        "timestamp": line.split()[0:3],
                                        "message": line.strip()
                                    })
                                    
        except Exception as e:
            self.logger.error(f"Error extracting logs: {e}")
            
        return logs
        
    def scan_iocs(self) -> Dict[str, List[Dict]]:
        """Scan system for Indicators of Compromise"""
        self.logger.info("Scanning for IOCs...")
        
        results = {
            "files": [],
            "processes": [],
            "network": [],
            "registry": []
        }
        
        try:
            # Load IOC rules
            rules = yara.compile(self.config["ioc_rules"])
            
            # Scan files
            for root, _, files in os.walk("/"):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            matches = rules.match(data=f.read())
                            if matches:
                                results["files"].append({
                                    "path": file_path,
                                    "matches": [match.rule for match in matches]
                                })
                    except Exception:
                        continue
                        
            # Scan processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    cmdline = " ".join(proc_info['cmdline']) if proc_info['cmdline'] else ""
                    matches = rules.match(data=cmdline.encode())
                    if matches:
                        results["processes"].append({
                            "pid": proc_info['pid'],
                            "name": proc_info['name'],
                            "matches": [match.rule for match in matches]
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # Scan network connections
            for conn in psutil.net_connections(kind='inet'):
                try:
                    if conn.raddr:
                        remote_addr = f"{conn.raddr.ip}:{conn.raddr.port}"
                        matches = rules.match(data=remote_addr.encode())
                        if matches:
                            results["network"].append({
                                "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}",
                                "remote_addr": remote_addr,
                                "matches": [match.rule for match in matches]
                            })
                except Exception:
                    continue
                    
            # Scan Windows registry
            if platform.system() == "Windows":
                self._scan_windows_registry(results)
                
        except Exception as e:
            self.logger.error(f"Error scanning IOCs: {e}")
            
        return results
        
    def _scan_windows_registry(self, results: Dict):
        """Scan Windows registry for IOCs"""
        try:
            rules = yara.compile(self.config["ioc_rules"])
            registry_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce",
                r"SYSTEM\CurrentControlSet\Services"
            ]
            
            for path in registry_paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                    i = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(key, i)
                            matches = rules.match(data=str(value).encode())
                            if matches:
                                results["registry"].append({
                                    "path": path,
                                    "name": name,
                                    "value": value,
                                    "matches": [match.rule for match in matches]
                                })
                            i += 1
                        except WindowsError:
                            break
                except Exception as e:
                    self.logger.error(f"Error scanning registry path {path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error scanning Windows registry: {e}")
            
    def generate_report(self, evidence: EvidencePackage) -> str:
        """Generate comprehensive forensic report"""
        self.logger.info("Generating forensic report...")
        
        try:
            report_path = os.path.join(self.evidence_dir, "forensic_report.html")
            
            # Collect all findings
            findings = {
                "system_info": self._get_os_info(),
                "artifacts": evidence.artifacts,
                "memory_analysis": self.analyze_memory(os.path.join(self.evidence_dir, "memory.raw")),
                "disk_analysis": self.analyze_disk("/"),
                "logs": self.extract_logs(),
                "iocs": self.scan_iocs()
            }
            
            # Generate HTML report
            with open(report_path, 'w') as f:
                f.write(self._generate_html_report(findings))
                
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return ""
            
    def _generate_html_report(self, findings: Dict) -> str:
        """Generate HTML forensic report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .section { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
                .finding { margin: 10px 0; padding: 5px; background: #f5f5f5; }
                .timestamp { color: #666; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <h1>Forensic Analysis Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        """.format(timestamp=datetime.datetime.now().isoformat())
        
        # Add each section
        for section, data in findings.items():
            html += f"""
            <div class="section">
                <h2>{section.replace('_', ' ').title()}</h2>
                <pre>{json.dumps(data, indent=2)}</pre>
            </div>
            """
            
        html += """
        </body>
        </html>
        """
        
        return html
        
    def run(self, mode: ForensicsMode = ForensicsMode.INTERACTIVE):
        """Run the forensics engine in specified mode"""
        self.mode = mode
        self.logger.info(f"Starting forensics engine in {mode.name} mode")
        
        try:
            # Capture artifacts
            evidence = self.capture_artifacts()
            
            # Dump memory
            memory_dump = self.dump_memory(full_dump=True)
            
            # Analyze memory
            memory_analysis = self.analyze_memory(memory_dump)
            
            # Analyze disk
            disk_analysis = self.analyze_disk("/")
            
            # Extract logs
            logs = self.extract_logs()
            
            # Scan for IOCs
            iocs = self.scan_iocs()
            
            # Generate report
            report_path = self.generate_report(evidence)
            
            self.logger.info(f"Forensics analysis complete. Report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error running forensics engine: {e}")
            return None
            
def main():
    """Main function to run the forensics engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hexa Assistant Forensics Engine")
    parser.add_argument("--mode", choices=["silent", "interactive", "emergency"],
                      default="interactive", help="Operation mode")
    parser.add_argument("--config", default="config/forensics_config.json",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    engine = ForensicsEngine(args.config)
    mode = ForensicsMode[args.mode.upper()]
    
    report_path = engine.run(mode)
    if report_path:
        print(f"Forensics analysis complete. Report saved to: {report_path}")
    else:
        print("Forensics analysis failed.")
        
if __name__ == "__main__":
    main() 