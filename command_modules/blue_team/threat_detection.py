#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import platform
import subprocess
import datetime
import threading
import queue
import hashlib
import socket
import re
import glob
import signal
import psutil
import yara
import magic
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# AI and ML imports
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Network monitoring
import scapy.all as scapy
from scapy.layers import http

# Process monitoring
import psutil
import win32process
import win32api
import win32security
import win32event
import win32service
import win32serviceutil

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Log parsing
import evtx
import win32evtlog
import win32evtlogutil
import win32con

# Sandbox and VM detection
import ctypes
import wmi
import virtualbox

# Voice and CLI
import pyttsx3
import speech_recognition as sr
import argparse

class ThreatLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class DetectionType(Enum):
    PROCESS = auto()
    NETWORK = auto()
    FILE = auto()
    LOG = auto()
    BEHAVIOR = auto()
    IOC = auto()
    SANDBOX = auto()

@dataclass
class Threat:
    """Information about a detected threat"""
    id: str
    timestamp: str
    level: ThreatLevel
    type: DetectionType
    description: str
    source: str
    details: Dict[str, Any]
    iocs: List[str] = field(default_factory=list)
    score: float = 0.0
    status: str = "NEW"
    actions_taken: List[str] = field(default_factory=list)

class ThreatDetectionEngine:
    def __init__(self, config_path: str = "config/threat_detection_config.json"):
        """Initialize the threat detection engine"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_ai_models()
        self.setup_monitoring()
        self.threat_queue = queue.Queue()
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load threat detection configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Create default threat detection configuration"""
        return {
            "monitoring": {
                "process_scan_interval": 5,
                "network_scan_interval": 1,
                "file_scan_interval": 60,
                "log_scan_interval": 30
            },
            "ai_models": {
                "anomaly_detection": {
                    "contamination": 0.1,
                    "random_state": 42
                },
                "behavior_analysis": {
                    "threshold": 0.8
                }
            },
            "threat_intelligence": {
                "feeds": [
                    "https://api.circl.lu/v1/",
                    "https://api.abuseipdb.com/api/v2/",
                    "https://www.virustotal.com/vtapi/v2/"
                ],
                "update_interval": 3600
            },
            "alerting": {
                "desktop_notifications": True,
                "voice_alerts": True,
                "sound_alerts": True,
                "log_alerts": True
            },
            "response": {
                "autonomous_mode": False,
                "allowed_actions": [
                    "kill_process",
                    "block_ip",
                    "isolate_host"
                ]
            },
            "reporting": {
                "format": "html",
                "include_iocs": True,
                "include_timeline": True
            }
        }
        
    def setup_logging(self):
        """Configure logging for threat detection"""
        log_dir = "logs/threat_detection"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/threat_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ThreatDetectionEngine")
        
    def setup_ai_models(self):
        """Initialize AI models for threat detection"""
        try:
            self.models = {
                "anomaly_detector": IsolationForest(
                    contamination=self.config["ai_models"]["anomaly_detection"]["contamination"],
                    random_state=self.config["ai_models"]["anomaly_detection"]["random_state"]
                ),
                "behavior_analyzer": pipeline("text-classification"),
                "summarizer": pipeline("summarization")
            }
            self.scaler = StandardScaler()
        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
            
    def setup_monitoring(self):
        """Initialize system monitoring components"""
        # Process monitoring
        self.process_monitor = threading.Thread(target=self._monitor_processes, daemon=True)
        self.process_monitor.start()
        
        # Network monitoring
        self.network_monitor = threading.Thread(target=self._monitor_network, daemon=True)
        self.network_monitor.start()
        
        # File monitoring
        self.file_observer = Observer()
        self.file_observer.schedule(FileChangeHandler(self), path="/", recursive=True)
        self.file_observer.start()
        
        # Log monitoring
        self.log_monitor = threading.Thread(target=self._monitor_logs, daemon=True)
        self.log_monitor.start()
        
    def _monitor_processes(self):
        """Monitor running processes for suspicious behavior"""
        while True:
            try:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'create_time']):
                    try:
                        pinfo = proc.info
                        processes.append({
                            'pid': pinfo['pid'],
                            'name': pinfo['name'],
                            'cmdline': pinfo['cmdline'],
                            'username': pinfo['username'],
                            'create_time': pinfo['create_time']
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
                # Analyze process behavior
                self._analyze_process_behavior(processes)
                
                time.sleep(self.config["monitoring"]["process_scan_interval"])
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                
    def _monitor_network(self):
        """Monitor network traffic for suspicious activity"""
        while True:
            try:
                # Capture network packets
                packets = scapy.sniff(timeout=1)
                
                # Analyze network traffic
                self._analyze_network_traffic(packets)
                
                time.sleep(self.config["monitoring"]["network_scan_interval"])
            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")
                
    def _monitor_logs(self):
        """Monitor system logs for suspicious events"""
        while True:
            try:
                if platform.system() == "Windows":
                    self._monitor_windows_logs()
                else:
                    self._monitor_unix_logs()
                    
                time.sleep(self.config["monitoring"]["log_scan_interval"])
            except Exception as e:
                self.logger.error(f"Error in log monitoring: {e}")
                
    def _monitor_windows_logs(self):
        """Monitor Windows Event Logs"""
        try:
            server = 'localhost'
            logtype = 'Security'
            hand = win32evtlog.OpenEventLog(server, logtype)
            total = win32evtlog.GetNumberOfEventLogRecords(hand)
            
            flags = win32evtlog.EVENTLOG_BACKWARDS_READ|win32evtlog.EVENTLOG_SEQUENTIAL_READ
            events = []
            
            while True:
                events = win32evtlog.ReadEventLog(hand, flags, 0)
                if not events:
                    break
                    
                for event in events:
                    self._analyze_windows_event(event)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring Windows logs: {e}")
            
    def _monitor_unix_logs(self):
        """Monitor Unix/Linux system logs"""
        log_files = {
            "auth": "/var/log/auth.log",
            "syslog": "/var/log/syslog",
            "messages": "/var/log/messages"
        }
        
        for log_type, log_file in log_files.items():
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            self._analyze_unix_log(line, log_type)
                except Exception as e:
                    self.logger.error(f"Error reading log file {log_file}: {e}")
                    
    def _analyze_process_behavior(self, processes: List[Dict]):
        """Analyze process behavior for anomalies"""
        try:
            # Extract features
            features = []
            for proc in processes:
                feature_vector = [
                    len(proc['cmdline']) if proc['cmdline'] else 0,
                    time.time() - proc['create_time'],
                    len(proc['name']),
                    hash(proc['username']) % 1000  # Normalize username hash
                ]
                features.append(feature_vector)
                
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            predictions = self.models["anomaly_detector"].predict(features_scaled)
            scores = self.models["anomaly_detector"].score_samples(features_scaled)
            
            # Check for suspicious processes
            for i, (proc, pred, score) in enumerate(zip(processes, predictions, scores)):
                if pred == -1 or score < self.config["ai_models"]["anomaly_detection"]["threshold"]:
                    self._create_threat(
                        ThreatLevel.HIGH,
                        DetectionType.PROCESS,
                        f"Suspicious process detected: {proc['name']}",
                        "process_monitor",
                        {
                            "pid": proc['pid'],
                            "name": proc['name'],
                            "cmdline": proc['cmdline'],
                            "username": proc['username'],
                            "anomaly_score": float(score)
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Error analyzing process behavior: {e}")
            
    def _analyze_network_traffic(self, packets: List[scapy.Packet]):
        """Analyze network traffic for suspicious activity"""
        try:
            connections = defaultdict(lambda: {"count": 0, "bytes": 0})
            
            for packet in packets:
                if packet.haslayer(scapy.IP):
                    src = packet[scapy.IP].src
                    dst = packet[scapy.IP].dst
                    key = f"{src}->{dst}"
                    connections[key]["count"] += 1
                    connections[key]["bytes"] += len(packet)
                    
            # Check for suspicious patterns
            for conn, stats in connections.items():
                src, dst = conn.split("->")
                
                # Check for data exfiltration
                if stats["bytes"] > 1000000:  # 1MB threshold
                    self._create_threat(
                        ThreatLevel.HIGH,
                        DetectionType.NETWORK,
                        f"Possible data exfiltration detected: {conn}",
                        "network_monitor",
                        {
                            "source": src,
                            "destination": dst,
                            "bytes_transferred": stats["bytes"],
                            "packet_count": stats["count"]
                        }
                    )
                    
                # Check for known malicious IPs
                if self._check_ip_reputation(dst):
                    self._create_threat(
                        ThreatLevel.CRITICAL,
                        DetectionType.NETWORK,
                        f"Connection to known malicious IP: {dst}",
                        "network_monitor",
                        {
                            "source": src,
                            "destination": dst,
                            "bytes_transferred": stats["bytes"],
                            "packet_count": stats["count"]
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Error analyzing network traffic: {e}")
            
    def _analyze_windows_event(self, event):
        """Analyze Windows Event Log entries"""
        try:
            event_id = event.EventID
            source = event.SourceName
            
            # Check for suspicious events
            suspicious_events = {
                4624: "Successful logon",
                4625: "Failed logon",
                4688: "Process creation",
                4698: "Scheduled task creation",
                4699: "Scheduled task deletion",
                4700: "Scheduled task modification"
            }
            
            if event_id in suspicious_events:
                message = win32evtlogutil.SafeFormatMessage(event, "Security")
                
                # Analyze event message
                if self._is_suspicious_event(event_id, message):
                    self._create_threat(
                        ThreatLevel.MEDIUM,
                        DetectionType.LOG,
                        f"Suspicious Windows Event: {suspicious_events[event_id]}",
                        "log_monitor",
                        {
                            "event_id": event_id,
                            "source": source,
                            "message": message,
                            "time": event.TimeGenerated.isoformat()
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Error analyzing Windows event: {e}")
            
    def _analyze_unix_log(self, line: str, log_type: str):
        """Analyze Unix/Linux log entries"""
        try:
            # Check for suspicious patterns
            suspicious_patterns = {
                "auth": [
                    r"Failed password for .* from .*",
                    r"Accepted password for .* from .*",
                    r"sudo: .* : command not found",
                    r"sudo: .* : .* incorrect password attempts"
                ],
                "syslog": [
                    r"error|Error|ERROR",
                    r"warning|Warning|WARNING",
                    r"critical|Critical|CRITICAL"
                ]
            }
            
            if log_type in suspicious_patterns:
                for pattern in suspicious_patterns[log_type]:
                    if re.search(pattern, line):
                        self._create_threat(
                            ThreatLevel.MEDIUM,
                            DetectionType.LOG,
                            f"Suspicious {log_type} log entry detected",
                            "log_monitor",
                            {
                                "log_type": log_type,
                                "pattern": pattern,
                                "message": line.strip(),
                                "time": datetime.datetime.now().isoformat()
                            }
                        )
                        
        except Exception as e:
            self.logger.error(f"Error analyzing Unix log: {e}")
            
    def _check_ip_reputation(self, ip: str) -> bool:
        """Check IP reputation against threat intelligence feeds"""
        try:
            # Check cached results first
            if ip in self._ip_cache:
                return self._ip_cache[ip]
                
            # Query threat intelligence feeds
            for feed in self.config["threat_intelligence"]["feeds"]:
                try:
                    response = requests.get(f"{feed}/ip/{ip}")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("malicious", False):
                            self._ip_cache[ip] = True
                            return True
                except Exception:
                    continue
                    
            self._ip_cache[ip] = False
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking IP reputation: {e}")
            return False
            
    def _is_suspicious_event(self, event_id: int, message: str) -> bool:
        """Determine if a Windows event is suspicious"""
        try:
            # Check for known suspicious patterns
            suspicious_patterns = {
                4624: [  # Successful logon
                    r"logon type:\s*3",  # Network logon
                    r"workstation name:\s*[^\\]+$"  # Unknown workstation
                ],
                4625: [  # Failed logon
                    r"logon type:\s*3",  # Network logon
                    r"failure reason:\s*Unknown user name or bad password"
                ],
                4688: [  # Process creation
                    r"new process name:\s*.*\.exe",
                    r"parent process name:\s*.*\\cmd\.exe"
                ]
            }
            
            if event_id in suspicious_patterns:
                for pattern in suspicious_patterns[event_id]:
                    if re.search(pattern, message, re.IGNORECASE):
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking suspicious event: {e}")
            return False
            
    def _create_threat(self, level: ThreatLevel, type: DetectionType, description: str,
                      source: str, details: Dict[str, Any]):
        """Create a new threat detection"""
        try:
            threat = Threat(
                id=f"THREAT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(description)}",
                timestamp=datetime.datetime.now().isoformat(),
                level=level,
                type=type,
                description=description,
                source=source,
                details=details
            )
            
            # Score the threat
            threat.score = self._score_threat(threat)
            
            # Add to queue for processing
            self.threat_queue.put(threat)
            
        except Exception as e:
            self.logger.error(f"Error creating threat: {e}")
            
    def _score_threat(self, threat: Threat) -> float:
        """Score a threat based on various factors"""
        try:
            score = 0.0
            
            # Base score from threat level
            level_scores = {
                ThreatLevel.LOW: 0.25,
                ThreatLevel.MEDIUM: 0.5,
                ThreatLevel.HIGH: 0.75,
                ThreatLevel.CRITICAL: 1.0
            }
            score += level_scores[threat.level]
            
            # Additional factors
            if threat.type == DetectionType.NETWORK:
                if "bytes_transferred" in threat.details:
                    if threat.details["bytes_transferred"] > 1000000:
                        score += 0.2
                        
            elif threat.type == DetectionType.PROCESS:
                if "anomaly_score" in threat.details:
                    score += (1 - threat.details["anomaly_score"]) * 0.3
                    
            elif threat.type == DetectionType.LOG:
                if threat.details.get("event_id") in [4624, 4625]:  # Logon events
                    score += 0.2
                    
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error scoring threat: {e}")
            return 0.0
            
    def _process_alerts(self):
        """Process detected threats and generate alerts"""
        while True:
            try:
                threat = self.threat_queue.get()
                
                # Generate alert
                self._generate_alert(threat)
                
                # Take action if in autonomous mode
                if self.config["response"]["autonomous_mode"]:
                    self._respond_to_threat(threat)
                    
                # Generate report
                self._generate_report(threat)
                
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
                
    def _generate_alert(self, threat: Threat):
        """Generate alerts for detected threats"""
        try:
            # Desktop notification
            if self.config["alerting"]["desktop_notifications"]:
                self._send_desktop_notification(threat)
                
            # Voice alert
            if self.config["alerting"]["voice_alerts"]:
                self._send_voice_alert(threat)
                
            # Sound alert
            if self.config["alerting"]["sound_alerts"]:
                self._play_alert_sound()
                
            # Log alert
            if self.config["alerting"]["log_alerts"]:
                self._log_alert(threat)
                
        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")
            
    def _send_desktop_notification(self, threat: Threat):
        """Send desktop notification for threat"""
        try:
            import win10toast
            toaster = win10toast.ToastNotifier()
            toaster.show_toast(
                "Threat Detected",
                f"{threat.level.name}: {threat.description}",
                duration=10,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Error sending desktop notification: {e}")
            
    def _send_voice_alert(self, threat: Threat):
        """Send voice alert for threat"""
        try:
            engine = pyttsx3.init()
            engine.say(f"Threat detected: {threat.description}")
            engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error sending voice alert: {e}")
            
    def _play_alert_sound(self):
        """Play alert sound"""
        try:
            import winsound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        except Exception as e:
            self.logger.error(f"Error playing alert sound: {e}")
            
    def _log_alert(self, threat: Threat):
        """Log threat alert"""
        try:
            alert_dir = "alerts"
            os.makedirs(alert_dir, exist_ok=True)
            
            alert_file = f"{alert_dir}/threat_{threat.id}.json"
            with open(alert_file, 'w') as f:
                json.dump(threat.__dict__, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
            
    def _respond_to_threat(self, threat: Threat):
        """Take autonomous action against threat"""
        try:
            if threat.level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                if "kill_process" in self.config["response"]["allowed_actions"]:
                    if threat.type == DetectionType.PROCESS:
                        self._kill_process(threat.details["pid"])
                        
                if "block_ip" in self.config["response"]["allowed_actions"]:
                    if threat.type == DetectionType.NETWORK:
                        self._block_ip(threat.details["destination"])
                        
                if "isolate_host" in self.config["response"]["allowed_actions"]:
                    self._isolate_host()
                    
        except Exception as e:
            self.logger.error(f"Error responding to threat: {e}")
            
    def _kill_process(self, pid: int):
        """Kill a suspicious process"""
        try:
            os.kill(pid, signal.SIGTERM)
            self.logger.info(f"Killed process {pid}")
        except Exception as e:
            self.logger.error(f"Error killing process {pid}: {e}")
            
    def _block_ip(self, ip: str):
        """Block a malicious IP"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule",
                              "name=BlockIP", f"remoteip={ip}", "dir=out", "action=block"])
            else:
                subprocess.run(["iptables", "-A", "OUTPUT", "-d", ip, "-j", "DROP"])
            self.logger.info(f"Blocked IP {ip}")
        except Exception as e:
            self.logger.error(f"Error blocking IP {ip}: {e}")
            
    def _isolate_host(self):
        """Isolate the host from the network"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["netsh", "advfirewall", "set", "allprofiles", "state", "on"])
            else:
                subprocess.run(["iptables", "-P", "INPUT", "DROP"])
                subprocess.run(["iptables", "-P", "OUTPUT", "DROP"])
            self.logger.info("Host isolated from network")
        except Exception as e:
            self.logger.error(f"Error isolating host: {e}")
            
    def _generate_report(self, threat: Threat):
        """Generate detailed threat report"""
        try:
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate HTML report
            if self.config["reporting"]["format"] == "html":
                report_file = f"{report_dir}/threat_report_{threat.id}.html"
                with open(report_file, 'w') as f:
                    f.write(self._generate_html_report(threat))
                    
            # Generate Markdown report
            else:
                report_file = f"{report_dir}/threat_report_{threat.id}.md"
                with open(report_file, 'w') as f:
                    f.write(self._generate_markdown_report(threat))
                    
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            
    def _generate_html_report(self, threat: Threat) -> str:
        """Generate HTML threat report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Threat Report - {threat.id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
                .threat-level {{ font-weight: bold; }}
                .threat-level.LOW {{ color: #666; }}
                .threat-level.MEDIUM {{ color: #f90; }}
                .threat-level.HIGH {{ color: #f00; }}
                .threat-level.CRITICAL {{ color: #900; }}
            </style>
        </head>
        <body>
            <h1>Threat Report</h1>
            <div class="section">
                <h2>Threat Information</h2>
                <p><strong>ID:</strong> {threat.id}</p>
                <p><strong>Timestamp:</strong> {threat.timestamp}</p>
                <p><strong>Level:</strong> <span class="threat-level {threat.level.name}">{threat.level.name}</span></p>
                <p><strong>Type:</strong> {threat.type.name}</p>
                <p><strong>Description:</strong> {threat.description}</p>
                <p><strong>Source:</strong> {threat.source}</p>
                <p><strong>Score:</strong> {threat.score:.2f}</p>
            </div>
            <div class="section">
                <h2>Details</h2>
                <pre>{json.dumps(threat.details, indent=2)}</pre>
            </div>
            <div class="section">
                <h2>Indicators of Compromise</h2>
                <ul>
                    {''.join(f'<li>{ioc}</li>' for ioc in threat.iocs)}
                </ul>
            </div>
            <div class="section">
                <h2>Actions Taken</h2>
                <ul>
                    {''.join(f'<li>{action}</li>' for action in threat.actions_taken)}
                </ul>
            </div>
        </body>
        </html>
        """
        return html
        
    def _generate_markdown_report(self, threat: Threat) -> str:
        """Generate Markdown threat report"""
        md = f"""
        # Threat Report

        ## Threat Information
        - **ID:** {threat.id}
        - **Timestamp:** {threat.timestamp}
        - **Level:** {threat.level.name}
        - **Type:** {threat.type.name}
        - **Description:** {threat.description}
        - **Source:** {threat.source}
        - **Score:** {threat.score:.2f}

        ## Details
        ```json
        {json.dumps(threat.details, indent=2)}
        ```

        ## Indicators of Compromise
        {''.join(f'- {ioc}\n' for ioc in threat.iocs)}

        ## Actions Taken
        {''.join(f'- {action}\n' for action in threat.actions_taken)}
        """
        return md
        
    def run(self, mode: str = "interactive"):
        """Run the threat detection engine"""
        self.logger.info(f"Starting threat detection engine in {mode} mode")
        
        try:
            # Start monitoring threads
            self.process_monitor.start()
            self.network_monitor.start()
            self.log_monitor.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down threat detection engine")
            self.file_observer.stop()
            self.file_observer.join()
            
        except Exception as e:
            self.logger.error(f"Error running threat detection engine: {e}")
            
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine
        
    def on_modified(self, event):
        if not event.is_directory:
            self.engine._create_threat(
                ThreatLevel.MEDIUM,
                DetectionType.FILE,
                f"File modified: {event.src_path}",
                "file_monitor",
                {
                    "path": event.src_path,
                    "event": "modified",
                    "time": datetime.datetime.now().isoformat()
                }
            )
            
    def on_created(self, event):
        if not event.is_directory:
            self.engine._create_threat(
                ThreatLevel.MEDIUM,
                DetectionType.FILE,
                f"File created: {event.src_path}",
                "file_monitor",
                {
                    "path": event.src_path,
                    "event": "created",
                    "time": datetime.datetime.now().isoformat()
                }
            )
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.engine._create_threat(
                ThreatLevel.MEDIUM,
                DetectionType.FILE,
                f"File deleted: {event.src_path}",
                "file_monitor",
                {
                    "path": event.src_path,
                    "event": "deleted",
                    "time": datetime.datetime.now().isoformat()
                }
            )
            
def main():
    """Main function to run the threat detection engine"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Threat Detection Engine")
    parser.add_argument("--mode", choices=["silent", "interactive", "emergency"],
                      default="interactive", help="Operation mode")
    parser.add_argument("--config", default="config/threat_detection_config.json",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    engine = ThreatDetectionEngine(args.config)
    engine.run(args.mode)
    
if __name__ == "__main__":
    main()
