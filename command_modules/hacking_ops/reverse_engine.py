#!/usr/bin/env python3

import os
import sys
import logging
import subprocess
import json
import hashlib
import re
import time
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

try:
    import pefile
    import r2pipe
    import yara
    import pycryptodome
    from Crypto.Cipher import AES, XOR
    from scapy.all import *
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", 
                   "pefile", "r2pipe", "yara-python", "pycryptodome", "scapy"])
    import pefile
    import r2pipe
    import yara
    from Crypto.Cipher import AES, XOR
    from scapy.all import *

class ReverseEngine:
    def __init__(self, target_file: str, output_dir: str = "reports/reverse_engineering"):
        """Initialize the ReverseEngine with target file and output directory."""
        self.target_file = Path(target_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Analysis results
        self.analysis_results = {
            "file_info": {},
            "strings": [],
            "imports": [],
            "sections": [],
            "resources": [],
            "network_activity": [],
            "encryption_findings": [],
            "malware_indicators": [],
            "vulnerabilities": []
        }
        
        # Initialize tools and configurations
        self.initialize_tools()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"reverse_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_tools(self):
        """Initialize necessary reverse engineering tools and configurations."""
        self.logger.info("Initializing reverse engineering tools...")
        
        # Initialize YARA rules
        self.load_yara_rules()
        
        # Check for required tools
        self.check_required_tools()

    def load_yara_rules(self):
        """Load YARA rules for malware detection."""
        rules_dir = Path(__file__).parent / "rules"
        rules_dir.mkdir(exist_ok=True)
        
        # Create basic YARA rules if they don't exist
        if not (rules_dir / "malware_rules.yar").exists():
            self.create_default_yara_rules(rules_dir)
            
        try:
            self.yara_rules = yara.compile(str(rules_dir / "malware_rules.yar"))
        except Exception as e:
            self.logger.error(f"Failed to load YARA rules: {e}")
            self.yara_rules = None

    def create_default_yara_rules(self, rules_dir: Path):
        """Create default YARA rules for basic malware detection."""
        default_rules = """
rule PotentialMalware {
    strings:
        $shell_code = {55 8B EC} // Common shellcode pattern
        $suspicious_api = "WinExec" ascii wide
        $network = "socket" ascii wide
        $encryption = "CryptoAPI" ascii wide
        
    condition:
        any of them
}
"""
        with open(rules_dir / "malware_rules.yar", "w") as f:
            f.write(default_rules)

    def check_required_tools(self):
        """Check if required external tools are available."""
        required_tools = {
            "radare2": "r2",
            "strings": "strings",
            "objdump": "objdump"
        }
        
        for tool, command in required_tools.items():
            try:
                subprocess.run([command, "--version"], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
                self.logger.info(f"{tool} is available")
            except FileNotFoundError:
                self.logger.warning(f"{tool} is not installed")

    def analyze_file(self) -> Dict:
        """Perform comprehensive analysis of the target file."""
        self.logger.info(f"Starting analysis of {self.target_file}")
        
        try:
            # Basic file analysis
            self.analyze_file_format()
            
            # Static analysis
            self.perform_static_analysis()
            
            # Dynamic analysis (in sandbox)
            self.perform_dynamic_analysis()
            
            # Encryption analysis
            self.analyze_encryption()
            
            # Network analysis
            self.analyze_network_behavior()
            
            # Generate report
            self.generate_report()
            
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def analyze_file_format(self):
        """Analyze the file format and basic properties."""
        self.logger.info("Analyzing file format...")
        
        file_info = {
            "name": self.target_file.name,
            "size": self.target_file.stat().st_size,
            "md5": self.calculate_file_hash("md5"),
            "sha256": self.calculate_file_hash("sha256"),
            "type": self.detect_file_type()
        }
        
        self.analysis_results["file_info"] = file_info

    def calculate_file_hash(self, algorithm: str) -> str:
        """Calculate file hash using specified algorithm."""
        hash_func = getattr(hashlib, algorithm)()
        with open(self.target_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def detect_file_type(self) -> str:
        """Detect the type of the target file."""
        try:
            with open(self.target_file, "rb") as f:
                magic_bytes = f.read(4)
                
            if magic_bytes.startswith(b"MZ"):
                return "PE Executable"
            elif magic_bytes.startswith(b"\x7fELF"):
                return "ELF Binary"
            elif magic_bytes.startswith(b"PK"):
                return "ZIP/APK/JAR"
            else:
                return "Unknown"
        except Exception as e:
            self.logger.error(f"File type detection failed: {e}")
            return "Unknown"

    def perform_static_analysis(self):
        """Perform static analysis of the binary."""
        self.logger.info("Performing static analysis...")
        
        if self.analysis_results["file_info"]["type"] == "PE Executable":
            self.analyze_pe_file()
        else:
            self.analyze_generic_binary()

    def analyze_pe_file(self):
        """Analyze PE file structure and characteristics."""
        try:
            pe = pefile.PE(self.target_file)
            
            # Analyze imports
            self.analysis_results["imports"] = [
                {"dll": entry.dll.decode(),
                 "functions": [imp.name.decode() for imp in entry.imports if imp.name]}
                for entry in pe.DIRECTORY_ENTRY_IMPORT
            ]
            
            # Analyze sections
            self.analysis_results["sections"] = [
                {"name": section.Name.decode().strip("\x00"),
                 "virtual_address": hex(section.VirtualAddress),
                 "size": section.SizeOfRawData,
                 "entropy": section.get_entropy()}
                for section in pe.sections
            ]
            
            # Check for potential packed/encrypted sections
            self.detect_packed_sections(pe)
            
        except Exception as e:
            self.logger.error(f"PE analysis failed: {e}")

    def detect_packed_sections(self, pe):
        """Detect potentially packed or encrypted sections."""
        for section in pe.sections:
            if section.get_entropy() > 7.0:  # High entropy indicates encryption/packing
                self.analysis_results["encryption_findings"].append({
                    "type": "high_entropy_section",
                    "section_name": section.Name.decode().strip("\x00"),
                    "entropy": section.get_entropy()
                })

    def analyze_generic_binary(self):
        """Analyze non-PE format binaries."""
        try:
            # Use radare2 for binary analysis
            r2 = r2pipe.open(str(self.target_file))
            r2.cmd("aaa")  # Analyze all
            
            # Get functions
            functions = r2.cmdj("aflj")
            if functions:
                self.analysis_results["functions"] = [
                    {"name": func["name"],
                     "address": hex(func["offset"]),
                     "size": func["size"]}
                    for func in functions
                ]
            
            # Extract strings
            strings = r2.cmdj("izj")
            if strings:
                self.analysis_results["strings"] = [
                    {"string": s["string"],
                     "type": s["type"],
                     "offset": hex(s["offset"])}
                    for s in strings
                ]
            
            r2.quit()
            
        except Exception as e:
            self.logger.error(f"Generic binary analysis failed: {e}")

    def perform_dynamic_analysis(self):
        """Perform dynamic analysis in a sandbox environment."""
        self.logger.info("Performing dynamic analysis...")
        
        # TODO: Implement sandbox environment using Docker or VM
        # For now, just analyze basic behavior
        self.analyze_behavior_patterns()

    def analyze_behavior_patterns(self):
        """Analyze behavioral patterns in the binary."""
        # Apply YARA rules
        if self.yara_rules:
            try:
                matches = self.yara_rules.match(str(self.target_file))
                for match in matches:
                    self.analysis_results["malware_indicators"].append({
                        "rule_name": match.rule,
                        "strings": match.strings,
                        "tags": match.tags
                    })
            except Exception as e:
                self.logger.error(f"YARA analysis failed: {e}")

    def analyze_encryption(self):
        """Analyze encryption and attempt to identify encryption methods."""
        self.logger.info("Analyzing encryption...")
        
        # Look for common encryption patterns
        encryption_patterns = {
            "xor": rb"\x00{16}",  # Potential XOR key pattern
            "aes": rb"AES",  # AES reference
            "base64": rb"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
        }
        
        with open(self.target_file, "rb") as f:
            content = f.read()
            
            for enc_type, pattern in encryption_patterns.items():
                if re.search(pattern, content):
                    self.analysis_results["encryption_findings"].append({
                        "type": enc_type,
                        "confidence": "medium",
                        "location": "binary"
                    })

    def analyze_network_behavior(self):
        """Analyze potential network behavior."""
        self.logger.info("Analyzing network behavior...")
        
        # Look for network-related strings and patterns
        network_patterns = {
            "url": rb"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+",
            "ip": rb"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
            "email": rb"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        }
        
        with open(self.target_file, "rb") as f:
            content = f.read()
            
            for pattern_type, pattern in network_patterns.items():
                matches = re.finditer(pattern, content)
                for match in matches:
                    self.analysis_results["network_activity"].append({
                        "type": pattern_type,
                        "value": match.group().decode(errors="ignore"),
                        "offset": hex(match.start())
                    })

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        self.logger.info("Generating analysis report...")
        
        report_file = self.output_dir / f"report_{self.target_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "target_file": str(self.target_file),
            "analysis_results": self.analysis_results
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)
            
        self.logger.info(f"Report generated: {report_file}")

    def attempt_decryption(self, data: bytes, method: str = "all") -> Optional[bytes]:
        """Attempt to decrypt data using various methods."""
        decryption_methods = {
            "xor": self._try_xor_decrypt,
            "caesar": self._try_caesar_decrypt,
            "base64": self._try_base64_decrypt
        }
        
        if method == "all":
            for decrypt_func in decryption_methods.values():
                try:
                    result = decrypt_func(data)
                    if result:
                        return result
                except Exception as e:
                    self.logger.debug(f"Decryption attempt failed: {e}")
        elif method in decryption_methods:
            return decryption_methods[method](data)
            
        return None

    def _try_xor_decrypt(self, data: bytes, max_key_length: int = 4) -> Optional[bytes]:
        """Attempt XOR decryption with different key lengths."""
        for key_length in range(1, max_key_length + 1):
            for key in range(256 ** key_length):
                key_bytes = key.to_bytes(key_length, "little")
                result = bytes(a ^ b for a, b in zip(data, key_bytes * (len(data) // len(key_bytes) + 1)))
                
                # Check if result looks like text
                if all(32 <= b <= 126 or b in (9, 10, 13) for b in result[:20]):
                    return result
        return None

    def _try_caesar_decrypt(self, data: bytes, max_shift: int = 26) -> Optional[bytes]:
        """Attempt Caesar cipher decryption."""
        for shift in range(max_shift):
            result = bytes((b + shift) % 256 for b in data)
            if all(32 <= b <= 126 or b in (9, 10, 13) for b in result[:20]):
                return result
        return None

    def _try_base64_decrypt(self, data: bytes) -> Optional[bytes]:
        """Attempt base64 decryption."""
        try:
            import base64
            return base64.b64decode(data)
        except:
            return None

def main():
    """Main function to run the reverse engineering tool."""
    if len(sys.argv) < 2:
        print("Usage: python reverse_engine.py <target_file> [output_dir]")
        sys.exit(1)
        
    target_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reports/reverse_engineering"
    
    try:
        reverse_engine = ReverseEngine(target_file, output_dir)
        results = reverse_engine.analyze_file()
        print(f"Analysis complete. Results saved in {output_dir}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
