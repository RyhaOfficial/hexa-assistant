#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import argparse
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import difflib
import re
import hashlib
from datetime import datetime
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import yaml
import jinja2
import gpt_interface  # Custom GPT interface module

class PatchType(Enum):
    """Enum for different types of patches."""
    CVE = "cve"
    CODE = "code"
    SYSTEM = "system"
    EXPLOIT = "exploit"
    CUSTOM = "custom"

class PatchStatus(Enum):
    """Enum for patch generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    TESTING = "testing"

@dataclass
class PatchInfo:
    """Class to store patch information."""
    patch_id: str
    patch_type: PatchType
    target: str
    description: str
    severity: str
    cve_id: Optional[str] = None
    source_code: Optional[str] = None
    generated_patch: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    timestamp: str = datetime.now().isoformat()
    status: PatchStatus = PatchStatus.PENDING

class AutoPatchGenerator:
    def __init__(self, config_file: str = "config/patch_config.json"):
        """Initialize the AutoPatchGenerator with configuration."""
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize directories
        self.setup_directories()
        
        # Initialize GPT interface
        self.gpt = gpt_interface.GPTInterface(self.config["gpt_settings"])
        
        # Initialize template engine
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates/patches")
        )

    def load_config(self) -> Dict:
        """Load patch generator configuration from file."""
        default_config = {
            "api_keys": {
                "cve_circl": "",
                "github": "",
                "openai": ""
            },
            "patch_settings": {
                "auto_apply": False,
                "test_patches": True,
                "backup_before_apply": True,
                "supported_languages": ["python", "c", "cpp", "javascript", "php"],
                "patch_formats": ["sh", "patch", "ps1", "txt"]
            },
            "gpt_settings": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s"
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
        log_dir = Path("logs/patches")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"patch_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories for patch generation."""
        directories = [
            "patch_logs",
            "patch_temp",
            "templates/patches",
            "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def generate_patch(self, patch_info: PatchInfo) -> PatchInfo:
        """Generate a patch based on the provided information."""
        try:
            self.logger.info(f"Starting patch generation for {patch_info.patch_id}")
            patch_info.status = PatchStatus.GENERATING
            
            if patch_info.patch_type == PatchType.CVE:
                await self._handle_cve_patch(patch_info)
            elif patch_info.patch_type == PatchType.CODE:
                await self._handle_code_patch(patch_info)
            elif patch_info.patch_type == PatchType.SYSTEM:
                await self._handle_system_patch(patch_info)
            elif patch_info.patch_type == PatchType.EXPLOIT:
                await self._handle_exploit_patch(patch_info)
            elif patch_info.patch_type == PatchType.CUSTOM:
                await self._handle_custom_patch(patch_info)
            
            # Test the generated patch
            if self.config["patch_settings"]["test_patches"]:
                patch_info.status = PatchStatus.TESTING
                patch_info.test_results = await self._test_patch(patch_info)
            
            patch_info.status = PatchStatus.COMPLETED
            self._save_patch_info(patch_info)
            
            return patch_info
            
        except Exception as e:
            self.logger.error(f"Patch generation failed: {e}")
            patch_info.status = PatchStatus.FAILED
            raise

    async def _handle_cve_patch(self, patch_info: PatchInfo):
        """Handle CVE-based patch generation."""
        try:
            # Fetch CVE details from CIRCL
            cve_data = await self._fetch_cve_details(patch_info.cve_id)
            
            # Analyze CVE data using GPT
            analysis = await self.gpt.analyze_cve(cve_data)
            
            # Generate appropriate patch based on analysis
            if "code_fix" in analysis:
                patch_info.generated_patch = await self._generate_code_fix(analysis["code_fix"])
            elif "config_fix" in analysis:
                patch_info.generated_patch = await this._generate_config_fix(analysis["config_fix"])
            elif "system_fix" in analysis:
                patch_info.generated_patch = await this._generate_system_fix(analysis["system_fix"])
            
            self.logger.info(f"CVE patch generated for {patch_info.cve_id}")
            
        except Exception as e:
            self.logger.error(f"CVE patch generation failed: {e}")
            raise

    async def _fetch_cve_details(self, cve_id: str) -> Dict:
        """Fetch CVE details from CIRCL API."""
        try:
            url = f"https://cve.circl.lu/api/cve/{cve_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Failed to fetch CVE details: {response.status}")
        except Exception as e:
            self.logger.error(f"CVE fetch failed: {e}")
            raise

    async def _handle_code_patch(self, patch_info: PatchInfo):
        """Handle code-based patch generation."""
        try:
            # Analyze source code using GPT
            analysis = await this.gpt.analyze_code(patch_info.source_code)
            
            # Generate secure code replacement
            secure_code = await this.gpt.generate_secure_code(analysis)
            
            # Create patch file
            patch_info.generated_patch = await this._create_code_patch(
                patch_info.source_code,
                secure_code
            )
            
            self.logger.info(f"Code patch generated for {patch_info.target}")
            
        except Exception as e:
            self.logger.error(f"Code patch generation failed: {e}")
            raise

    async def _handle_system_patch(self, patch_info: PatchInfo):
        """Handle system-based patch generation."""
        try:
            # Generate system hardening script
            hardening_script = await this._generate_hardening_script(patch_info.target)
            
            # Create appropriate script based on OS
            if sys.platform.startswith("linux"):
                patch_info.generated_patch = await this._create_bash_script(hardening_script)
            elif sys.platform.startswith("win"):
                patch_info.generated_patch = await this._create_powershell_script(hardening_script)
            
            self.logger.info(f"System patch generated for {patch_info.target}")
            
        except Exception as e:
            self.logger.error(f"System patch generation failed: {e}")
            raise

    async def _handle_exploit_patch(self, patch_info: PatchInfo):
        """Handle exploit-based patch generation."""
        try:
            # Analyze exploit pattern
            analysis = await this.gpt.analyze_exploit(patch_info.target)
            
            # Generate mitigation rules
            if "ids_rules" in analysis:
                patch_info.generated_patch = await this._generate_ids_rules(analysis["ids_rules"])
            elif "waf_rules" in analysis:
                patch_info.generated_patch = await this._generate_waf_rules(analysis["waf_rules"])
            elif "firewall_rules" in analysis:
                patch_info.generated_patch = await this._generate_firewall_rules(analysis["firewall_rules"])
            
            self.logger.info(f"Exploit patch generated for {patch_info.target}")
            
        except Exception as e:
            self.logger.error(f"Exploit patch generation failed: {e}")
            raise

    async def _test_patch(self, patch_info: PatchInfo) -> Dict[str, Any]:
        """Test the generated patch."""
        try:
            test_results = {
                "syntax_check": False,
                "compilation_check": False,
                "vulnerability_check": False,
                "details": {}
            }
            
            # Perform syntax check
            if patch_info.patch_type == PatchType.CODE:
                test_results["syntax_check"] = await this._check_syntax(patch_info.generated_patch)
            
            # Perform compilation check for C/C++
            if patch_info.patch_type == PatchType.CODE and patch_info.target.endswith((".c", ".cpp")):
                test_results["compilation_check"] = await this._check_compilation(patch_info.generated_patch)
            
            # Check if vulnerability is still present
            test_results["vulnerability_check"] = await this._check_vulnerability(patch_info)
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Patch testing failed: {e}")
            return {"error": str(e)}

    def _save_patch_info(self, patch_info: PatchInfo):
        """Save patch information to log file."""
        try:
            log_file = Path("patch_logs") / f"{patch_info.patch_id}.json"
            with open(log_file, "w") as f:
                json.dump(patch_info.__dict__, f, indent=4)
            
            # Also save patch content
            patch_file = Path("patch_logs") / f"patch_{patch_info.patch_id}.{self._get_patch_extension(patch_info)}"
            with open(patch_file, "w") as f:
                f.write(patch_info.generated_patch)
            
            self.logger.info(f"Patch information saved to {log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save patch information: {e}")

    def _get_patch_extension(self, patch_info: PatchInfo) -> str:
        """Get appropriate file extension for patch type."""
        if patch_info.patch_type == PatchType.CODE:
            return "patch"
        elif patch_info.patch_type == PatchType.SYSTEM:
            return "sh" if sys.platform.startswith("linux") else "ps1"
        else:
            return "txt"

    async def apply_patch(self, patch_info: PatchInfo, auto_apply: bool = False) -> bool:
        """Apply the generated patch."""
        try:
            if not auto_apply and not self.config["patch_settings"]["auto_apply"]:
                response = input(f"Apply patch for {patch_info.patch_id}? (Y/N): ")
                if response.lower() != "y":
                    return False
            
            # Create backup if enabled
            if self.config["patch_settings"]["backup_before_apply"]:
                await this._create_backup(patch_info)
            
            # Apply patch based on type
            if patch_info.patch_type == PatchType.CODE:
                success = await this._apply_code_patch(patch_info)
            elif patch_info.patch_type == PatchType.SYSTEM:
                success = await this._apply_system_patch(patch_info)
            else:
                success = await this._apply_generic_patch(patch_info)
            
            if success:
                self.logger.info(f"Patch {patch_info.patch_id} applied successfully")
            else:
                self.logger.error(f"Failed to apply patch {patch_info.patch_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Patch application failed: {e}")
            return False

    async def _create_backup(self, patch_info: PatchInfo):
        """Create backup before applying patch."""
        try:
            backup_dir = Path("backups") / patch_info.patch_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            if patch_info.patch_type == PatchType.CODE:
                # Backup source file
                shutil.copy2(patch_info.target, backup_dir / Path(patch_info.target).name)
            elif patch_info.patch_type == PatchType.SYSTEM:
                # Backup system configuration
                await this._backup_system_config(backup_dir)
            
            self.logger.info(f"Backup created in {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            raise

def main():
    """Main function to run the patch generator."""
    try:
        parser = argparse.ArgumentParser(description="Hexa Assistant Auto Patch Generator")
        parser.add_argument("--cve", help="CVE ID to patch")
        parser.add_argument("--file", help="File to patch")
        parser.add_argument("--type", choices=["cve", "code", "system", "exploit", "custom"],
                          help="Type of patch to generate")
        parser.add_argument("--auto-fix", action="store_true", help="Automatically apply fixes")
        parser.add_argument("--dry-run", action="store_true", help="Generate patch without applying")
        parser.add_argument("--config", help="Path to config file", default="config/patch_config.json")
        args = parser.parse_args()
        
        generator = AutoPatchGenerator(args.config)
        
        # Create patch info
        patch_info = PatchInfo(
            patch_id=f"patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            patch_type=PatchType(args.type) if args.type else PatchType.CUSTOM,
            target=args.cve or args.file or "",
            description="",
            severity="unknown"
        )
        
        # Generate and apply patch
        asyncio.run(generator.generate_patch(patch_info))
        
        if not args.dry_run:
            asyncio.run(generator.apply_patch(patch_info, args.auto_fix))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 