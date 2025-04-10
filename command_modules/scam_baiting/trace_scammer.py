#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import datetime
import socket
import re
import asyncio
import aiohttp
import phonenumbers
import requests
import whois
import dns.resolver
import shodan
import ipapi
import emailrep
import abuseipdb
import hunter
import numverify
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

# Voice and Audio
import speech_recognition as sr
import whisper
import pydub
from pydub import AudioSegment
import librosa
import numpy as np

# AI and ML
from transformers import pipeline
import openai
from langchain import OpenAI, LLMChain, PromptTemplate
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Network and Security
import socks
import requests
from requests.exceptions import RequestException
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import ssl
import OpenSSL

# Database
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# CLI
import argparse
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.tree import Tree
from rich.markdown import Markdown

class InputType(Enum):
    PHONE = auto()
    EMAIL = auto()
    IP = auto()
    DOMAIN = auto()
    USERNAME = auto()
    VOICE = auto()
    TEXT = auto()

class ScamType(Enum):
    IRS = auto()
    TECH_SUPPORT = auto()
    BANK = auto()
    LOTTERY = auto()
    SEXTORTION = auto()
    CRYPTO = auto()
    OTHER = auto()

@dataclass
class ScammerProfile:
    """Comprehensive scammer profile"""
    id: str
    input_type: InputType
    input_value: str
    scam_type: ScamType
    confidence: float
    first_seen: str
    last_seen: str
    phone_info: Dict[str, Any] = field(default_factory=dict)
    email_info: Dict[str, Any] = field(default_factory=dict)
    ip_info: Dict[str, Any] = field(default_factory=dict)
    domain_info: Dict[str, Any] = field(default_factory=dict)
    username_info: Dict[str, Any] = field(default_factory=dict)
    voice_info: Dict[str, Any] = field(default_factory=dict)
    social_graph: Dict[str, Any] = field(default_factory=dict)
    behavior_profile: Dict[str, Any] = field(default_factory=dict)
    location_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TraceScammer:
    def __init__(self, config_path: str = "config/trace_scammer_config.json"):
        """Initialize the scammer tracer"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_ai_models()
        self.setup_database()
        self.setup_api_clients()
        self.setup_proxy_chain()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load trace scammer configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Create default trace scammer configuration"""
        return {
            "api_keys": {
                "shodan": "",
                "ipapi": "",
                "emailrep": "",
                "abuseipdb": "",
                "hunter": "",
                "numverify": ""
            },
            "ai": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 150
            },
            "proxy": {
                "enabled": True,
                "chain": ["socks5://127.0.0.1:9050"],
                "rotate_interval": 300
            },
            "logging": {
                "enabled": True,
                "path": "traced_scammers",
                "format": "json"
            }
        }
        
    def setup_logging(self):
        """Configure logging for scammer tracer"""
        log_dir = self.config["logging"]["path"]
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/trace_scammer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TraceScammer")
        
    def setup_ai_models(self):
        """Initialize AI models for scammer tracing"""
        try:
            self.models = {
                "transcriber": whisper.load_model("base"),
                "classifier": pipeline("text-classification"),
                "chat": self._setup_chat_model(),
                "voice_fingerprint": self._setup_voice_model()
            }
        except Exception as e:
            self.logger.error(f"Error initializing AI models: {e}")
            
    def _setup_chat_model(self):
        """Setup chat model for AI decisions"""
        try:
            from langchain import OpenAI, LLMChain, PromptTemplate
            
            return OpenAI(
                model_name=self.config["ai"]["model"],
                temperature=self.config["ai"]["temperature"],
                max_tokens=self.config["ai"]["max_tokens"]
            )
        except Exception as e:
            self.logger.error(f"Error setting up chat model: {e}")
            return None
            
    def _setup_voice_model(self):
        """Setup voice fingerprinting model"""
        try:
            # Load pre-trained voice embedding model
            model = torch.hub.load('pytorch/fairseq', 'wav2vec2_base')
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Error setting up voice model: {e}")
            return None
            
    def setup_database(self):
        """Initialize database for scammer profiles"""
        try:
            db_path = "traced_scammers/database.sqlite"
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            
    def setup_api_clients(self):
        """Initialize API clients for OSINT"""
        try:
            self.clients = {
                "shodan": shodan.Shodan(self.config["api_keys"]["shodan"]),
                "ipapi": ipapi.Client(self.config["api_keys"]["ipapi"]),
                "emailrep": emailrep.Client(self.config["api_keys"]["emailrep"]),
                "abuseipdb": abuseipdb.Client(self.config["api_keys"]["abuseipdb"]),
                "hunter": hunter.Client(self.config["api_keys"]["hunter"]),
                "numverify": numverify.Client(self.config["api_keys"]["numverify"])
            }
        except Exception as e:
            self.logger.error(f"Error initializing API clients: {e}")
            
    def setup_proxy_chain(self):
        """Setup proxy chain for anonymous tracing"""
        try:
            if self.config["proxy"]["enabled"]:
                # Configure proxy chain
                for proxy in self.config["proxy"]["chain"]:
                    protocol, host, port = proxy.split("://")
                    socks.set_default_proxy(
                        getattr(socks, protocol.upper()),
                        host,
                        int(port)
                    )
                socket.socket = socks.socksocket
                
        except Exception as e:
            self.logger.error(f"Error setting up proxy chain: {e}")
            
    def trace_phone_number(self, number: str) -> Dict[str, Any]:
        """Trace phone number information"""
        try:
            # Format number
            parsed = phonenumbers.parse(number)
            formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            
            # Get carrier info
            carrier = phonenumbers.carrier.name_for_number(parsed, "en")
            region = phonenumbers.region_code_for_number(parsed)
            
            # Get location info
            location = self.clients["numverify"].get_location(formatted)
            
            # Check for leaks
            leaks = self._check_number_leaks(formatted)
            
            return {
                "number": formatted,
                "carrier": carrier,
                "region": region,
                "location": location,
                "leaks": leaks
            }
            
        except Exception as e:
            self.logger.error(f"Error tracing phone number: {e}")
            return {}
            
    def _check_number_leaks(self, number: str) -> List[Dict[str, Any]]:
        """Check phone number in leaked databases"""
        try:
            # Implement leak checking logic
            return []
        except Exception as e:
            self.logger.error(f"Error checking number leaks: {e}")
            return []
            
    def trace_email_address(self, email: str) -> Dict[str, Any]:
        """Trace email address information"""
        try:
            # Check email reputation
            reputation = self.clients["emailrep"].get_reputation(email)
            
            # Check breaches
            breaches = self._check_email_breaches(email)
            
            # Get domain info
            domain = email.split("@")[1]
            domain_info = self.trace_domain(domain)
            
            # Check social media
            social = self._check_social_media(email)
            
            return {
                "email": email,
                "reputation": reputation,
                "breaches": breaches,
                "domain": domain_info,
                "social": social
            }
            
        except Exception as e:
            self.logger.error(f"Error tracing email address: {e}")
            return {}
            
    def _check_email_breaches(self, email: str) -> List[Dict[str, Any]]:
        """Check email in breach databases"""
        try:
            # Implement breach checking logic
            return []
        except Exception as e:
            self.logger.error(f"Error checking email breaches: {e}")
            return []
            
    def _check_social_media(self, email: str) -> List[Dict[str, Any]]:
        """Check email across social media platforms"""
        try:
            # Implement social media checking logic
            return []
        except Exception as e:
            self.logger.error(f"Error checking social media: {e}")
            return []
            
    def trace_ip_address(self, ip: str) -> Dict[str, Any]:
        """Trace IP address information"""
        try:
            # Get IP info
            ip_info = self.clients["ipapi"].get_ip_info(ip)
            
            # Check abuse status
            abuse = self.clients["abuseipdb"].get_ip_info(ip)
            
            # Get Shodan data
            shodan_data = self.clients["shodan"].search(f"hostname:{ip}")
            
            # Detect VPN/Proxy
            vpn = self._detect_vpn(ip)
            
            return {
                "ip": ip,
                "info": ip_info,
                "abuse": abuse,
                "shodan": shodan_data,
                "vpn": vpn
            }
            
        except Exception as e:
            self.logger.error(f"Error tracing IP address: {e}")
            return {}
            
    def _detect_vpn(self, ip: str) -> bool:
        """Detect if IP is a VPN/Proxy"""
        try:
            # Implement VPN detection logic
            return False
        except Exception as e:
            self.logger.error(f"Error detecting VPN: {e}")
            return False
            
    def trace_domain(self, domain: str) -> Dict[str, Any]:
        """Trace domain information"""
        try:
            # Get WHOIS info
            whois_info = whois.whois(domain)
            
            # Get DNS records
            dns_records = self._get_dns_records(domain)
            
            # Get SSL info
            ssl_info = self._get_ssl_info(domain)
            
            # Scan with urlscan
            urlscan = self._scan_url(domain)
            
            return {
                "domain": domain,
                "whois": whois_info,
                "dns": dns_records,
                "ssl": ssl_info,
                "urlscan": urlscan
            }
            
        except Exception as e:
            self.logger.error(f"Error tracing domain: {e}")
            return {}
            
    def _get_dns_records(self, domain: str) -> Dict[str, List[str]]:
        """Get DNS records for domain"""
        try:
            records = {}
            for record_type in ['A', 'MX', 'NS', 'TXT']:
                try:
                    answers = dns.resolver.resolve(domain, record_type)
                    records[record_type] = [str(rdata) for rdata in answers]
                except:
                    records[record_type] = []
            return records
        except Exception as e:
            self.logger.error(f"Error getting DNS records: {e}")
            return {}
            
    def _get_ssl_info(self, domain: str) -> Dict[str, Any]:
        """Get SSL certificate information"""
        try:
            context = ssl.create_default_context()
            with context.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.connect((domain, 443))
                cert = s.getpeercert()
            return cert
        except Exception as e:
            self.logger.error(f"Error getting SSL info: {e}")
            return {}
            
    def _scan_url(self, domain: str) -> Dict[str, Any]:
        """Scan domain with urlscan.io"""
        try:
            # Implement urlscan.io scanning logic
            return {}
        except Exception as e:
            self.logger.error(f"Error scanning URL: {e}")
            return {}
            
    def trace_username(self, username: str) -> Dict[str, Any]:
        """Trace username across platforms"""
        try:
            # Check common platforms
            platforms = self._check_username_platforms(username)
            
            # Get profile matches
            matches = self._get_profile_matches(username)
            
            return {
                "username": username,
                "platforms": platforms,
                "matches": matches
            }
            
        except Exception as e:
            self.logger.error(f"Error tracing username: {e}")
            return {}
            
    def _check_username_platforms(self, username: str) -> Dict[str, bool]:
        """Check username across platforms"""
        try:
            # Implement platform checking logic
            return {}
        except Exception as e:
            self.logger.error(f"Error checking username platforms: {e}")
            return {}
            
    def _get_profile_matches(self, username: str) -> List[Dict[str, Any]]:
        """Get matching profiles for username"""
        try:
            # Implement profile matching logic
            return []
        except Exception as e:
            self.logger.error(f"Error getting profile matches: {e}")
            return []
            
    def analyze_voice_sample(self, audio_path: str) -> Dict[str, Any]:
        """Analyze voice sample for fingerprinting"""
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Extract features
            features = self._extract_voice_features(audio)
            
            # Generate fingerprint
            fingerprint = self._generate_voice_fingerprint(features)
            
            # Match with database
            matches = self._match_voice_fingerprint(fingerprint)
            
            return {
                "audio": audio_path,
                "features": features,
                "fingerprint": fingerprint,
                "matches": matches
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing voice sample: {e}")
            return {}
            
    def _extract_voice_features(self, audio: AudioSegment) -> np.ndarray:
        """Extract voice features from audio"""
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Extract features using librosa
            mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=audio.frame_rate)
            
            return mfcc
        except Exception as e:
            self.logger.error(f"Error extracting voice features: {e}")
            return np.array([])
            
    def _generate_voice_fingerprint(self, features: np.ndarray) -> np.ndarray:
        """Generate voice fingerprint from features"""
        try:
            # Use voice model to generate embedding
            with torch.no_grad():
                embedding = self.models["voice_fingerprint"](features)
            return embedding.numpy()
        except Exception as e:
            self.logger.error(f"Error generating voice fingerprint: {e}")
            return np.array([])
            
    def _match_voice_fingerprint(self, fingerprint: np.ndarray) -> List[Dict[str, Any]]:
        """Match voice fingerprint with database"""
        try:
            # Implement fingerprint matching logic
            return []
        except Exception as e:
            self.logger.error(f"Error matching voice fingerprint: {e}")
            return []
            
    def profile_scam_behavior(self, text: str) -> Dict[str, Any]:
        """Profile scam behavior from text"""
        try:
            # Classify scam type
            result = self.models["classifier"](text)
            scam_type = ScamType[result[0]["label"].upper()]
            
            # Analyze psychological triggers
            triggers = self._analyze_triggers(text)
            
            # Detect language patterns
            patterns = self._detect_patterns(text)
            
            # Generate signature
            signature = self._generate_signature(text)
            
            return {
                "text": text,
                "type": scam_type,
                "triggers": triggers,
                "patterns": patterns,
                "signature": signature
            }
            
        except Exception as e:
            self.logger.error(f"Error profiling scam behavior: {e}")
            return {}
            
    def _analyze_triggers(self, text: str) -> List[str]:
        """Analyze psychological triggers in text"""
        try:
            # Implement trigger analysis logic
            return []
        except Exception as e:
            self.logger.error(f"Error analyzing triggers: {e}")
            return []
            
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect language patterns in text"""
        try:
            # Implement pattern detection logic
            return []
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
            
    def _generate_signature(self, text: str) -> str:
        """Generate unique signature for scam"""
        try:
            # Implement signature generation logic
            return ""
        except Exception as e:
            self.logger.error(f"Error generating signature: {e}")
            return ""
            
    def build_social_graph(self, profile: ScammerProfile) -> Dict[str, Any]:
        """Build social graph for scammer"""
        try:
            # Get linked accounts
            accounts = self._get_linked_accounts(profile)
            
            # Get call patterns
            calls = self._get_call_patterns(profile)
            
            # Get target patterns
            targets = self._get_target_patterns(profile)
            
            return {
                "accounts": accounts,
                "calls": calls,
                "targets": targets
            }
            
        except Exception as e:
            self.logger.error(f"Error building social graph: {e}")
            return {}
            
    def _get_linked_accounts(self, profile: ScammerProfile) -> List[Dict[str, Any]]:
        """Get accounts linked to scammer"""
        try:
            # Implement account linking logic
            return []
        except Exception as e:
            self.logger.error(f"Error getting linked accounts: {e}")
            return []
            
    def _get_call_patterns(self, profile: ScammerProfile) -> List[Dict[str, Any]]:
        """Get call patterns for scammer"""
        try:
            # Implement call pattern analysis logic
            return []
        except Exception as e:
            self.logger.error(f"Error getting call patterns: {e}")
            return []
            
    def _get_target_patterns(self, profile: ScammerProfile) -> List[Dict[str, Any]]:
        """Get target patterns for scammer"""
        try:
            # Implement target pattern analysis logic
            return []
        except Exception as e:
            self.logger.error(f"Error getting target patterns: {e}")
            return []
            
    def generate_report(self, profile: ScammerProfile) -> Dict[str, Any]:
        """Generate comprehensive report for scammer"""
        try:
            # Build report
            report = {
                "id": profile.id,
                "input": {
                    "type": profile.input_type.name,
                    "value": profile.input_value
                },
                "scam": {
                    "type": profile.scam_type.name,
                    "confidence": profile.confidence
                },
                "timeline": {
                    "first_seen": profile.first_seen,
                    "last_seen": profile.last_seen
                },
                "phone": profile.phone_info,
                "email": profile.email_info,
                "ip": profile.ip_info,
                "domain": profile.domain_info,
                "username": profile.username_info,
                "voice": profile.voice_info,
                "social": profile.social_graph,
                "behavior": profile.behavior_profile,
                "location": profile.location_info,
                "metadata": profile.metadata
            }
            
            # Save to file
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {}
            
    def _save_report(self, report: Dict[str, Any]):
        """Save report to file"""
        try:
            # Save to JSON
            report_file = f"traced_scammers/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
                
            # Save to database
            profile = ScammerProfile(
                id=report["id"],
                input_type=InputType[report["input"]["type"]],
                input_value=report["input"]["value"],
                scam_type=ScamType[report["scam"]["type"]],
                confidence=report["scam"]["confidence"],
                first_seen=report["timeline"]["first_seen"],
                last_seen=report["timeline"]["last_seen"],
                phone_info=report["phone"],
                email_info=report["email"],
                ip_info=report["ip"],
                domain_info=report["domain"],
                username_info=report["username"],
                voice_info=report["voice"],
                social_graph=report["social"],
                behavior_profile=report["behavior"],
                location_info=report["location"],
                metadata=report["metadata"]
            )
            
            self.session.add(profile)
            self.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            
    def process_voice_command(self, command: str):
        """Process voice command for tracing"""
        try:
            # Transcribe command
            transcript = self.models["transcriber"].transcribe(command)
            
            # Parse command
            if "trace this number" in transcript.lower():
                number = self._extract_number(transcript)
                self.trace_phone_number(number)
            elif "track the guy who just called" in transcript.lower():
                # Implement recent call tracking logic
                pass
            elif "find where this IP is from" in transcript.lower():
                ip = self._extract_ip(transcript)
                self.trace_ip_address(ip)
            elif "who owns the site" in transcript.lower():
                domain = self._extract_domain(transcript)
                self.trace_domain(domain)
                
        except Exception as e:
            self.logger.error(f"Error processing voice command: {e}")
            
    def _extract_number(self, text: str) -> str:
        """Extract phone number from text"""
        try:
            match = re.search(r'\+?\d{10,}', text)
            return match.group() if match else None
        except:
            return None
            
    def _extract_ip(self, text: str) -> str:
        """Extract IP address from text"""
        try:
            match = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text)
            return match.group() if match else None
        except:
            return None
            
    def _extract_domain(self, text: str) -> str:
        """Extract domain from text"""
        try:
            match = re.search(r'[\w-]+\.\w+', text)
            return match.group() if match else None
        except:
            return None
            
    def run(self, mode: str = "interactive"):
        """Run the scammer tracer"""
        self.logger.info(f"Starting scammer tracer in {mode} mode")
        
        try:
            # Start voice recognition
            r = sr.Recognizer()
            with sr.Microphone() as source:
                while True:
                    audio = r.listen(source)
                    
                    try:
                        # Process voice command
                        command = r.recognize_google(audio)
                        self.process_voice_command(command)
                        
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        self.logger.error(f"Error with speech recognition: {e}")
                        
        except KeyboardInterrupt:
            self.logger.info("Shutting down scammer tracer")
            
        except Exception as e:
            self.logger.error(f"Error running scammer tracer: {e}")
            
# Database Models
Base = declarative_base()

class ScammerProfile(Base):
    __tablename__ = 'scammer_profiles'
    
    id = Column(Integer, primary_key=True)
    input_type = Column(String)
    input_value = Column(String)
    scam_type = Column(String)
    confidence = Column(Float)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    phone_info = Column(JSON)
    email_info = Column(JSON)
    ip_info = Column(JSON)
    domain_info = Column(JSON)
    username_info = Column(JSON)
    voice_info = Column(JSON)
    social_graph = Column(JSON)
    behavior_profile = Column(JSON)
    location_info = Column(JSON)
    metadata = Column(JSON)
    
def main():
    """Main function to run the scammer tracer"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Scammer Tracer")
    parser.add_argument("--phone", help="Phone number to trace")
    parser.add_argument("--email", help="Email address to trace")
    parser.add_argument("--ip", help="IP address to trace")
    parser.add_argument("--url", help="Domain/URL to trace")
    parser.add_argument("--username", help="Username to trace")
    parser.add_argument("--voice", help="Voice sample to analyze")
    parser.add_argument("--text", help="Scam text to analyze")
    parser.add_argument("--config", default="config/trace_scammer_config.json",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    tracer = TraceScammer(args.config)
    
    if args.phone:
        tracer.trace_phone_number(args.phone)
    elif args.email:
        tracer.trace_email_address(args.email)
    elif args.ip:
        tracer.trace_ip_address(args.ip)
    elif args.url:
        tracer.trace_domain(args.url)
    elif args.username:
        tracer.trace_username(args.username)
    elif args.voice:
        tracer.analyze_voice_sample(args.voice)
    elif args.text:
        tracer.profile_scam_behavior(args.text)
        
if __name__ == "__main__":
    main()
