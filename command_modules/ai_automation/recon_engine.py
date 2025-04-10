#!/usr/bin/env python3

import os
import sys
import json
import time
import socket
import whois
import dns.resolver
import requests
import nmap
import shodan
import censys
import threading
import asyncio
import aiohttp
import subprocess
import logging
import datetime
import argparse
import ipaddress
import concurrent.futures
from typing import Dict, List, Set, Union, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class ReconStatus(Enum):
    """Enum for reconnaissance status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ReconTarget:
    """Class to store target information."""
    domain: str
    ip_addresses: List[str]
    subdomains: Set[str]
    ports: Dict[int, Dict[str, str]]  # port -> {service: str, version: str}
    vulnerabilities: List[Dict[str, Any]]
    whois_data: Dict[str, Any]
    dns_records: Dict[str, List[str]]
    technologies: List[str]
    geolocation: Dict[str, Any]
    osint_data: Dict[str, Any]

class ReconEngine:
    def __init__(self, config_file: str = "config/recon_config.json"):
        """Initialize the ReconEngine with configuration."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize API clients
        self.initialize_api_clients()
        
        # Initialize tools and dependencies
        self.check_dependencies()
        
        # Initialize storage
        self.results_dir = Path("results/recon")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict:
        """Load reconnaissance configuration from file."""
        default_config = {
            "api_keys": {
                "shodan": "",
                "censys": {
                    "api_id": "",
                    "api_secret": ""
                },
                "virustotal": "",
                "hunter": "",
                "hibp": ""
            },
            "scan_settings": {
                "ports": {
                    "common": True,
                    "all": False,
                    "custom": []
                },
                "subdomain_wordlist": "wordlists/subdomains.txt",
                "threads": 10,
                "timeout": 30
            },
            "stealth": {
                "use_proxy": False,
                "proxy_list": [],
                "delay_between_requests": 1
            },
            "reporting": {
                "formats": ["json", "html"],
                "include_screenshots": True
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
        log_dir = Path("logs/recon")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"recon_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_api_clients(self):
        """Initialize API clients for various services."""
        try:
            # Initialize Shodan client
            if self.config["api_keys"]["shodan"]:
                self.shodan_client = shodan.Shodan(self.config["api_keys"]["shodan"])
            
            # Initialize Censys client
            if self.config["api_keys"]["censys"]["api_id"]:
                self.censys_client = censys.search.CensysHosts(
                    api_id=self.config["api_keys"]["censys"]["api_id"],
                    api_secret=self.config["api_keys"]["censys"]["api_secret"]
                )
            
            # Initialize other API clients as needed
            self.logger.info("API clients initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {e}")

    def check_dependencies(self):
        """Check if required tools are installed."""
        required_tools = ["nmap", "amass", "subfinder", "nikto", "whatweb"]
        missing_tools = []
        
        for tool in required_tools:
            if not self._check_tool_installed(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            self.logger.warning(f"Missing required tools: {', '.join(missing_tools)}")
            self._install_missing_tools(missing_tools)

    def _check_tool_installed(self, tool: str) -> bool:
        """Check if a tool is installed."""
        try:
            subprocess.run([tool, "--version"], capture_output=True)
            return True
        except FileNotFoundError:
            return False

    def _install_missing_tools(self, tools: List[str]):
        """Install missing tools."""
        try:
            for tool in tools:
                self.logger.info(f"Installing {tool}...")
                if sys.platform.startswith('linux'):
                    subprocess.run(['apt-get', 'install', '-y', tool])
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['brew', 'install', tool])
                else:
                    self.logger.error(f"Automatic installation not supported for {sys.platform}")
        except Exception as e:
            self.logger.error(f"Failed to install tools: {e}")

    async def perform_recon(self, target: str) -> ReconTarget:
        """Perform comprehensive reconnaissance on a target."""
        try:
            self.logger.info(f"Starting reconnaissance for target: {target}")
            
            # Parse target
            parsed_target = urlparse(target)
            domain = parsed_target.netloc or parsed_target.path
            
            # Initialize target object
            recon_target = ReconTarget(
                domain=domain,
                ip_addresses=[],
                subdomains=set(),
                ports={},
                vulnerabilities=[],
                whois_data={},
                dns_records={},
                technologies=[],
                geolocation={},
                osint_data={}
            )
            
            # Create tasks for parallel execution
            tasks = [
                self.gather_dns_info(recon_target),
                self.enumerate_subdomains(recon_target),
                self.scan_ports(recon_target),
                self.gather_whois_info(recon_target),
                self.detect_technologies(recon_target),
                self.gather_osint_data(recon_target),
                self.scan_vulnerabilities(recon_target)
            ]
            
            # Execute tasks concurrently
            await asyncio.gather(*tasks)
            
            # Generate report
            self.generate_report(recon_target)
            
            return recon_target
            
        except Exception as e:
            self.logger.error(f"Reconnaissance failed: {e}")
            raise

    async def gather_dns_info(self, target: ReconTarget):
        """Gather DNS information for the target."""
        try:
            resolver = dns.resolver.Resolver()
            record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA']
            
            for record_type in record_types:
                try:
                    answers = resolver.resolve(target.domain, record_type)
                    target.dns_records[record_type] = [str(rdata) for rdata in answers]
                    
                    # Store IP addresses
                    if record_type in ['A', 'AAAA']:
                        target.ip_addresses.extend([str(rdata) for rdata in answers])
                        
                except dns.resolver.NoAnswer:
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to get {record_type} records: {e}")
            
            self.logger.info(f"DNS information gathered for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"Failed to gather DNS information: {e}")

    async def enumerate_subdomains(self, target: ReconTarget):
        """Enumerate subdomains using various techniques."""
        try:
            # Use multiple sources for subdomain enumeration
            tasks = [
                self._amass_enumeration(target),
                self._subfinder_enumeration(target),
                self._certificate_search(target),
                self._brute_force_subdomains(target)
            ]
            
            await asyncio.gather(*tasks)
            
            self.logger.info(f"Found {len(target.subdomains)} subdomains")
            
        except Exception as e:
            self.logger.error(f"Subdomain enumeration failed: {e}")

    async def _amass_enumeration(self, target: ReconTarget):
        """Enumerate subdomains using Amass."""
        try:
            cmd = ["amass", "enum", "-d", target.domain]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                subdomains = stdout.decode().splitlines()
                target.subdomains.update(subdomains)
                
        except Exception as e:
            self.logger.error(f"Amass enumeration failed: {e}")

    async def scan_ports(self, target: ReconTarget):
        """Scan ports using nmap."""
        try:
            nm = nmap.PortScanner()
            
            for ip in target.ip_addresses:
                # Determine ports to scan
                if self.config["scan_settings"]["ports"]["all"]:
                    ports = "1-65535"
                elif self.config["scan_settings"]["ports"]["custom"]:
                    ports = ",".join(map(str, self.config["scan_settings"]["ports"]["custom"]))
                else:
                    ports = "21-23,25,53,80,110,139,443,445,3306,3389,8080,8443"
                
                # Perform scan
                nm.scan(ip, ports, arguments="-sV -sS -T4")
                
                # Store results
                for host in nm.all_hosts():
                    for proto in nm[host].all_protocols():
                        ports = nm[host][proto].keys()
                        for port in ports:
                            target.ports[port] = {
                                "service": nm[host][proto][port]["name"],
                                "version": nm[host][proto][port]["version"]
                            }
            
            self.logger.info(f"Port scan completed for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"Port scanning failed: {e}")

    async def gather_whois_info(self, target: ReconTarget):
        """Gather WHOIS information."""
        try:
            w = whois.whois(target.domain)
            target.whois_data = w
            self.logger.info(f"WHOIS information gathered for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"WHOIS lookup failed: {e}")

    async def detect_technologies(self, target: ReconTarget):
        """Detect technologies used by the target."""
        try:
            # Use WhatWeb for technology detection
            cmd = ["whatweb", "-a", "3", "--log-json", "/dev/stdout", f"http://{target.domain}"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                results = json.loads(stdout.decode())
                for result in results:
                    if "plugins" in result:
                        target.technologies.extend(result["plugins"].keys())
            
            self.logger.info(f"Technology detection completed for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"Technology detection failed: {e}")

    async def gather_osint_data(self, target: ReconTarget):
        """Gather OSINT data from various sources."""
        try:
            tasks = [
                self._shodan_search(target),
                self._censys_search(target),
                self._search_breached_data(target),
                self._search_social_media(target)
            ]
            
            await asyncio.gather(*tasks)
            
            self.logger.info(f"OSINT data gathered for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"OSINT data gathering failed: {e}")

    async def _shodan_search(self, target: ReconTarget):
        """Search Shodan for target information."""
        if hasattr(self, 'shodan_client'):
            try:
                for ip in target.ip_addresses:
                    results = self.shodan_client.host(ip)
                    target.osint_data.setdefault("shodan", []).append(results)
            except Exception as e:
                self.logger.error(f"Shodan search failed: {e}")

    async def scan_vulnerabilities(self, target: ReconTarget):
        """Scan for vulnerabilities."""
        try:
            tasks = [
                self._nikto_scan(target),
                self._custom_vulnerability_checks(target)
            ]
            
            await asyncio.gather(*tasks)
            
            self.logger.info(f"Vulnerability scanning completed for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"Vulnerability scanning failed: {e}")

    async def _nikto_scan(self, target: ReconTarget):
        """Perform Nikto web vulnerability scan."""
        try:
            cmd = ["nikto", "-h", target.domain, "-Format", "json"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if stdout:
                results = json.loads(stdout.decode())
                target.vulnerabilities.extend(results.get("vulnerabilities", []))
                
        except Exception as e:
            self.logger.error(f"Nikto scan failed: {e}")

    def generate_report(self, target: ReconTarget):
        """Generate reconnaissance report."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_base = self.results_dir / f"recon_report_{target.domain}_{timestamp}"
            
            # Generate reports in specified formats
            if "json" in self.config["reporting"]["formats"]:
                self._generate_json_report(target, report_base)
            
            if "html" in self.config["reporting"]["formats"]:
                self._generate_html_report(target, report_base)
            
            self.logger.info(f"Report generated for {target.domain}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

    def _generate_json_report(self, target: ReconTarget, report_base: Path):
        """Generate JSON report."""
        try:
            report_data = {
                "target": target.domain,
                "timestamp": datetime.datetime.now().isoformat(),
                "ip_addresses": list(target.ip_addresses),
                "subdomains": list(target.subdomains),
                "ports": target.ports,
                "vulnerabilities": target.vulnerabilities,
                "whois_data": target.whois_data,
                "dns_records": target.dns_records,
                "technologies": target.technologies,
                "geolocation": target.geolocation,
                "osint_data": target.osint_data
            }
            
            with open(f"{report_base}.json", "w") as f:
                json.dump(report_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"JSON report generation failed: {e}")

    def _generate_html_report(self, target: ReconTarget, report_base: Path):
        """Generate HTML report."""
        try:
            # Load HTML template
            template_path = Path("templates/recon_report_template.html")
            if not template_path.exists():
                self._create_html_template()
            
            with open(template_path, "r") as f:
                template = f.read()
            
            # Replace placeholders with data
            report_html = template.replace("{{TARGET}}", target.domain)
            # Add more replacements for other data
            
            with open(f"{report_base}.html", "w") as f:
                f.write(report_html)
                
        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")

    def _create_html_template(self):
        """Create HTML report template."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reconnaissance Report - {{TARGET}}</title>
            <style>
                /* Add CSS styles here */
            </style>
        </head>
        <body>
            <h1>Reconnaissance Report - {{TARGET}}</h1>
            <!-- Add more template sections -->
        </body>
        </html>
        """
        
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "recon_report_template.html", "w") as f:
            f.write(template)

def main():
    """Main function to run the reconnaissance engine."""
    try:
        parser = argparse.ArgumentParser(description="Hexa Assistant Reconnaissance Engine")
        parser.add_argument("target", help="Target domain or IP address")
        parser.add_argument("--config", help="Path to config file", default="config/recon_config.json")
        args = parser.parse_args()
        
        recon_engine = ReconEngine(args.config)
        asyncio.run(recon_engine.perform_recon(args.target))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 