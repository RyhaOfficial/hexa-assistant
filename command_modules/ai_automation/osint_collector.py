#!/usr/bin/env python3

import os
import sys
import json
import time
import socket
import whois
import dns.resolver
import requests
import shodan
import hunter
import clearbit
import pipl
import asyncio
import aiohttp
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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class OSINTStatus(Enum):
    """Enum for OSINT collection status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class OSINTTarget:
    """Class to store OSINT target information."""
    domain: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    whois_data: Dict[str, Any] = None
    dns_records: Dict[str, List[str]] = None
    social_profiles: Dict[str, Dict[str, Any]] = None
    leaks: List[Dict[str, Any]] = None
    documents: List[Dict[str, Any]] = None
    github_data: Dict[str, Any] = None
    employee_info: List[Dict[str, Any]] = None
    api_keys: List[Dict[str, Any]] = None
    geolocation: Dict[str, Any] = None

class OSINTCollector:
    def __init__(self, config_file: str = "config/osint_config.json"):
        """Initialize the OSINTCollector with configuration."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize API clients
        self.initialize_api_clients()
        
        # Initialize storage
        self.results_dir = Path("results/osint")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize browser for web scraping
        self.setup_browser()

    def load_config(self) -> Dict:
        """Load OSINT configuration from file."""
        default_config = {
            "api_keys": {
                "shodan": "",
                "hunter": "",
                "clearbit": "",
                "pipl": "",
                "hibp": "",
                "github": ""
            },
            "collection_settings": {
                "social_media": {
                    "linkedin": True,
                    "twitter": True,
                    "facebook": True,
                    "instagram": True,
                    "github": True
                },
                "document_types": ["pdf", "doc", "docx", "txt"],
                "scan_interval": 3600,  # seconds
                "max_retries": 3
            },
            "stealth": {
                "use_proxy": False,
                "proxy_list": [],
                "delay_between_requests": 1,
                "rotate_user_agent": True
            },
            "reporting": {
                "formats": ["json", "html", "pdf"],
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
        log_dir = Path("logs/osint")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"osint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
            
            # Initialize Hunter client
            if self.config["api_keys"]["hunter"]:
                self.hunter_client = hunter.Hunter(self.config["api_keys"]["hunter"])
            
            # Initialize Clearbit client
            if self.config["api_keys"]["clearbit"]:
                self.clearbit_client = clearbit.Client(self.config["api_keys"]["clearbit"])
            
            # Initialize Pipl client
            if self.config["api_keys"]["pipl"]:
                self.pipl_client = pipl.Client(self.config["api_keys"]["pipl"])
            
            self.logger.info("API clients initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {e}")

    def setup_browser(self):
        """Setup headless browser for web scraping."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            if self.config["stealth"]["rotate_user_agent"]:
                chrome_options.add_argument(f"user-agent={self._get_random_user_agent()}")
            
            self.browser = webdriver.Chrome(options=chrome_options)
            self.logger.info("Browser setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup browser: {e}")

    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        return random.choice(user_agents)

    async def collect_osint(self, target: str, target_type: str) -> OSINTTarget:
        """Collect OSINT data for a target."""
        try:
            self.logger.info(f"Starting OSINT collection for {target_type}: {target}")
            
            # Initialize target object
            osint_target = OSINTTarget()
            
            # Set target type
            if target_type == "domain":
                osint_target.domain = target
            elif target_type == "email":
                osint_target.email = target
            elif target_type == "username":
                osint_target.username = target
            elif target_type == "ip":
                osint_target.ip_address = target
            
            # Create tasks for parallel execution
            tasks = []
            
            if osint_target.domain:
                tasks.extend([
                    self.gather_whois_info(osint_target),
                    self.gather_dns_info(osint_target),
                    self.scan_github(osint_target),
                    self.search_documents(osint_target)
                ])
            
            if osint_target.email:
                tasks.extend([
                    self.search_email_leaks(osint_target),
                    self.find_social_profiles(osint_target),
                    self.search_employee_info(osint_target)
                ])
            
            if osint_target.username:
                tasks.extend([
                    self.find_social_profiles(osint_target),
                    self.search_username_leaks(osint_target)
                ])
            
            if osint_target.ip_address:
                tasks.extend([
                    self.gather_ip_info(osint_target),
                    self.scan_ip_services(osint_target)
                ])
            
            # Execute tasks concurrently
            await asyncio.gather(*tasks)
            
            # Generate report
            self.generate_report(osint_target)
            
            return osint_target
            
        except Exception as e:
            self.logger.error(f"OSINT collection failed: {e}")
            raise

    async def gather_whois_info(self, target: OSINTTarget):
        """Gather WHOIS information."""
        try:
            if target.domain:
                w = whois.whois(target.domain)
                target.whois_data = w
                self.logger.info(f"WHOIS information gathered for {target.domain}")
        except Exception as e:
            self.logger.error(f"WHOIS lookup failed: {e}")

    async def gather_dns_info(self, target: OSINTTarget):
        """Gather DNS information."""
        try:
            if target.domain:
                resolver = dns.resolver.Resolver()
                record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'SOA']
                
                for record_type in record_types:
                    try:
                        answers = resolver.resolve(target.domain, record_type)
                        target.dns_records[record_type] = [str(rdata) for rdata in answers]
                    except dns.resolver.NoAnswer:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to get {record_type} records: {e}")
                
                self.logger.info(f"DNS information gathered for {target.domain}")
        except Exception as e:
            self.logger.error(f"DNS lookup failed: {e}")

    async def search_email_leaks(self, target: OSINTTarget):
        """Search for email leaks."""
        try:
            if target.email and self.config["api_keys"]["hibp"]:
                headers = {
                    "hibp-api-key": self.config["api_keys"]["hibp"],
                    "user-agent": "OSINT-Collector"
                }
                
                url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{target.email}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            leaks = await response.json()
                            target.leaks.extend(leaks)
                            self.logger.info(f"Found {len(leaks)} leaks for {target.email}")
        except Exception as e:
            self.logger.error(f"Email leak search failed: {e}")

    async def find_social_profiles(self, target: OSINTTarget):
        """Find social media profiles."""
        try:
            search_terms = []
            if target.email:
                search_terms.append(target.email)
            if target.username:
                search_terms.append(target.username)
            
            for term in search_terms:
                # Search LinkedIn
                if self.config["collection_settings"]["social_media"]["linkedin"]:
                    await self._search_linkedin(term, target)
                
                # Search Twitter
                if self.config["collection_settings"]["social_media"]["twitter"]:
                    await self._search_twitter(term, target)
                
                # Search other platforms
                for platform in ["facebook", "instagram", "github"]:
                    if self.config["collection_settings"]["social_media"][platform]:
                        await self._search_platform(platform, term, target)
            
            self.logger.info(f"Social profile search completed for {term}")
        except Exception as e:
            self.logger.error(f"Social profile search failed: {e}")

    async def _search_linkedin(self, term: str, target: OSINTTarget):
        """Search LinkedIn for profiles."""
        try:
            # Use Clearbit API for LinkedIn data
            if hasattr(self, 'clearbit_client'):
                person = self.clearbit_client.person.find(email=term)
                if person:
                    target.social_profiles.setdefault("linkedin", {}).update({
                        "name": person.get("name", {}).get("fullName"),
                        "title": person.get("headline"),
                        "company": person.get("employment", {}).get("name"),
                        "location": person.get("location", {}).get("name")
                    })
        except Exception as e:
            self.logger.error(f"LinkedIn search failed: {e}")

    async def scan_github(self, target: OSINTTarget):
        """Scan GitHub for sensitive information."""
        try:
            if target.domain:
                # Search for repositories
                search_query = f"org:{target.domain}"
                if hasattr(self, 'github_client'):
                    repos = self.github_client.search.repositories(search_query)
                    target.github_data = {
                        "repositories": [repo.raw_data for repo in repos],
                        "api_keys": await self._scan_for_api_keys(repos)
                    }
        except Exception as e:
            self.logger.error(f"GitHub scan failed: {e}")

    async def _scan_for_api_keys(self, repos) -> List[Dict[str, Any]]:
        """Scan repositories for API keys."""
        api_keys = []
        key_patterns = {
            "aws": r"AKIA[0-9A-Z]{16}",
            "github": r"ghp_[0-9a-zA-Z]{36}",
            "google": r"AIza[0-9A-Za-z-_]{35}"
        }
        
        for repo in repos:
            try:
                contents = repo.get_contents("")
                for content in contents:
                    if content.type == "file":
                        file_content = content.decoded_content.decode()
                        for key_type, pattern in key_patterns.items():
                            matches = re.finditer(pattern, file_content)
                            for match in matches:
                                api_keys.append({
                                    "type": key_type,
                                    "key": match.group(),
                                    "repository": repo.full_name,
                                    "file": content.path
                                })
            except Exception as e:
                self.logger.warning(f"Failed to scan repository {repo.full_name}: {e}")
        
        return api_keys

    async def search_documents(self, target: OSINTTarget):
        """Search for publicly available documents."""
        try:
            if target.domain:
                search_queries = [
                    f"site:{target.domain} filetype:pdf",
                    f"site:{target.domain} filetype:doc",
                    f"site:{target.domain} filetype:docx"
                ]
                
                for query in search_queries:
                    results = await self._search_google(query)
                    for result in results:
                        target.documents.append({
                            "url": result["link"],
                            "title": result["title"],
                            "snippet": result["snippet"],
                            "type": result["fileFormat"]
                        })
        except Exception as e:
            self.logger.error(f"Document search failed: {e}")

    async def _search_google(self, query: str) -> List[Dict[str, Any]]:
        """Perform Google search."""
        try:
            url = f"https://www.google.com/search?q={query}"
            self.browser.get(url)
            
            results = []
            for result in self.browser.find_elements(By.CSS_SELECTOR, "div.g"):
                try:
                    title = result.find_element(By.CSS_SELECTOR, "h3").text
                    link = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    snippet = result.find_element(By.CSS_SELECTOR, "div.VwiC3b").text
                    
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "fileFormat": self._get_file_format(link)
                    })
                except Exception:
                    continue
            
            return results
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []

    def _get_file_format(self, url: str) -> str:
        """Get file format from URL."""
        extension = url.split(".")[-1].lower()
        if extension in ["pdf", "doc", "docx", "txt"]:
            return extension
        return "unknown"

    def generate_report(self, target: OSINTTarget):
        """Generate OSINT report."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_base = self.results_dir / f"osint_report_{timestamp}"
            
            # Generate reports in specified formats
            if "json" in self.config["reporting"]["formats"]:
                self._generate_json_report(target, report_base)
            
            if "html" in self.config["reporting"]["formats"]:
                self._generate_html_report(target, report_base)
            
            if "pdf" in self.config["reporting"]["formats"]:
                self._generate_pdf_report(target, report_base)
            
            self.logger.info(f"Report generated for target")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

    def _generate_json_report(self, target: OSINTTarget, report_base: Path):
        """Generate JSON report."""
        try:
            report_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "target": {
                    "domain": target.domain,
                    "email": target.email,
                    "username": target.username,
                    "ip_address": target.ip_address
                },
                "whois_data": target.whois_data,
                "dns_records": target.dns_records,
                "social_profiles": target.social_profiles,
                "leaks": target.leaks,
                "documents": target.documents,
                "github_data": target.github_data,
                "employee_info": target.employee_info,
                "api_keys": target.api_keys,
                "geolocation": target.geolocation
            }
            
            with open(f"{report_base}.json", "w") as f:
                json.dump(report_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"JSON report generation failed: {e}")

    def _generate_html_report(self, target: OSINTTarget, report_base: Path):
        """Generate HTML report."""
        try:
            # Load HTML template
            template_path = Path("templates/osint_report_template.html")
            if not template_path.exists():
                self._create_html_template()
            
            with open(template_path, "r") as f:
                template = f.read()
            
            # Replace placeholders with data
            report_html = template.replace("{{TARGET}}", str(target.domain or target.email or target.username or target.ip_address))
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
            <title>OSINT Report - {{TARGET}}</title>
            <style>
                /* Add CSS styles here */
            </style>
        </head>
        <body>
            <h1>OSINT Report - {{TARGET}}</h1>
            <!-- Add more template sections -->
        </body>
        </html>
        """
        
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        with open(template_dir / "osint_report_template.html", "w") as f:
            f.write(template)

def main():
    """Main function to run the OSINT collector."""
    try:
        parser = argparse.ArgumentParser(description="Hexa Assistant OSINT Collector")
        parser.add_argument("target", help="Target (domain, email, username, or IP)")
        parser.add_argument("--type", choices=["domain", "email", "username", "ip"], required=True,
                          help="Type of target")
        parser.add_argument("--config", help="Path to config file", default="config/osint_config.json")
        args = parser.parse_args()
        
        collector = OSINTCollector(args.config)
        asyncio.run(collector.collect_osint(args.target, args.type))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
