import subprocess
import os
import platform
import logging
import psutil
import random
import time
import json
import shutil
import socket
import struct
import fcntl
import requests
from typing import Dict, List, Optional, Union
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from stem import Signal
from stem.control import Controller
import stem.process
import pyvpn
import pyproxy

# Set up logging
logging.basicConfig(
    filename='stealth_mode.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StealthMode:
    """Manages stealth and anonymity features for Hexa Assistant."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.stealth_config_path = Path("config/stealth_config.json")
        self.stealth_config = self._load_stealth_config()
        self.is_active = False
        self.current_vpn = None
        self.current_proxy = None
        self.tor_process = None
        self.original_mac = self._get_current_mac()
        self.original_ip = self._get_current_ip()
        logging.info("Initialized StealthMode")

    def _load_stealth_config(self) -> Dict:
        """Load stealth configuration from JSON file."""
        try:
            if self.stealth_config_path.exists():
                with open(self.stealth_config_path, 'r') as f:
                    return json.load(f)
            return {
                'vpn_servers': [],
                'proxy_servers': [],
                'tor_enabled': False,
                'mac_spoofing': False,
                'browser_fingerprinting': False,
                'process_hiding': False,
                'file_hiding': False,
                'log_cleaning': False
            }
        except Exception as e:
            logging.error(f"Error loading stealth config: {str(e)}")
            return {}

    def _save_stealth_config(self) -> bool:
        """Save stealth configuration to JSON file."""
        try:
            os.makedirs(self.stealth_config_path.parent, exist_ok=True)
            with open(self.stealth_config_path, 'w') as f:
                json.dump(self.stealth_config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Error saving stealth config: {str(e)}")
            return False

    def activate(self) -> bool:
        """Activate stealth mode with all configured features."""
        try:
            if self.is_active:
                logging.warning("Stealth mode is already active")
                return True

            # Enable all configured stealth features
            if self.stealth_config.get('vpn_enabled', False):
                self._connect_vpn()
            
            if self.stealth_config.get('tor_enabled', False):
                self._start_tor()
            
            if self.stealth_config.get('mac_spoofing', False):
                self._spoof_mac_address()
            
            if self.stealth_config.get('browser_fingerprinting', False):
                self._randomize_browser_fingerprint()
            
            if self.stealth_config.get('process_hiding', False):
                self._hide_processes()
            
            if self.stealth_config.get('file_hiding', False):
                self._hide_files()
            
            if self.stealth_config.get('log_cleaning', False):
                self._clean_logs()

            self.is_active = True
            logging.info("Stealth mode activated successfully")
            return True
        except Exception as e:
            logging.error(f"Error activating stealth mode: {str(e)}")
            return False

    def deactivate(self) -> bool:
        """Deactivate stealth mode and restore original settings."""
        try:
            if not self.is_active:
                logging.warning("Stealth mode is not active")
                return True

            # Restore original settings
            if self.stealth_config.get('vpn_enabled', False):
                self._disconnect_vpn()
            
            if self.stealth_config.get('tor_enabled', False):
                self._stop_tor()
            
            if self.stealth_config.get('mac_spoofing', False):
                self._restore_mac_address()
            
            if self.stealth_config.get('process_hiding', False):
                self._unhide_processes()
            
            if self.stealth_config.get('file_hiding', False):
                self._unhide_files()

            self.is_active = False
            logging.info("Stealth mode deactivated successfully")
            return True
        except Exception as e:
            logging.error(f"Error deactivating stealth mode: {str(e)}")
            return False

    def _connect_vpn(self) -> bool:
        """Connect to a VPN server."""
        try:
            if not self.stealth_config.get('vpn_servers'):
                logging.error("No VPN servers configured")
                return False

            # Select a random VPN server
            vpn_server = random.choice(self.stealth_config['vpn_servers'])
            self.current_vpn = vpn_server

            # Connect using pyvpn
            vpn = pyvpn.VPN(vpn_server['host'], vpn_server['username'], vpn_server['password'])
            vpn.connect()
            
            logging.info(f"Connected to VPN server: {vpn_server['host']}")
            return True
        except Exception as e:
            logging.error(f"Error connecting to VPN: {str(e)}")
            return False

    def _disconnect_vpn(self) -> bool:
        """Disconnect from the current VPN."""
        try:
            if not self.current_vpn:
                return True

            vpn = pyvpn.VPN(self.current_vpn['host'], 
                          self.current_vpn['username'], 
                          self.current_vpn['password'])
            vpn.disconnect()
            
            self.current_vpn = None
            logging.info("Disconnected from VPN")
            return True
        except Exception as e:
            logging.error(f"Error disconnecting from VPN: {str(e)}")
            return False

    def _start_tor(self) -> bool:
        """Start Tor service."""
        try:
            # Start Tor process
            self.tor_process = stem.process.launch_tor_with_config(
                config={
                    'SocksPort': '9050',
                    'ControlPort': '9051',
                }
            )
            
            # Set up controller
            with Controller.from_port(port=9051) as controller:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
            
            logging.info("Tor service started successfully")
            return True
        except Exception as e:
            logging.error(f"Error starting Tor: {str(e)}")
            return False

    def _stop_tor(self) -> bool:
        """Stop Tor service."""
        try:
            if self.tor_process:
                self.tor_process.terminate()
                self.tor_process = None
                logging.info("Tor service stopped successfully")
            return True
        except Exception as e:
            logging.error(f"Error stopping Tor: {str(e)}")
            return False

    def _get_current_mac(self) -> str:
        """Get current MAC address."""
        try:
            if self.system == 'linux':
                with open('/sys/class/net/eth0/address') as f:
                    return f.read().strip()
            elif self.system == 'windows':
                result = subprocess.run(['getmac'], capture_output=True, text=True)
                return result.stdout.split('\n')[3].split()[0]
            return ''
        except Exception as e:
            logging.error(f"Error getting MAC address: {str(e)}")
            return ''

    def _spoof_mac_address(self) -> bool:
        """Spoof MAC address."""
        try:
            if self.system == 'linux':
                # Generate random MAC
                new_mac = ':'.join(['%02x' % random.randint(0, 255) for _ in range(6)])
                subprocess.run(['ifconfig', 'eth0', 'down'], check=True)
                subprocess.run(['ifconfig', 'eth0', 'hw', 'ether', new_mac], check=True)
                subprocess.run(['ifconfig', 'eth0', 'up'], check=True)
            elif self.system == 'windows':
                # Windows MAC spoofing requires registry changes
                logging.warning("MAC spoofing on Windows requires additional setup")
                return False
            
            logging.info("MAC address spoofed successfully")
            return True
        except Exception as e:
            logging.error(f"Error spoofing MAC address: {str(e)}")
            return False

    def _restore_mac_address(self) -> bool:
        """Restore original MAC address."""
        try:
            if self.system == 'linux':
                subprocess.run(['ifconfig', 'eth0', 'down'], check=True)
                subprocess.run(['ifconfig', 'eth0', 'hw', 'ether', self.original_mac], check=True)
                subprocess.run(['ifconfig', 'eth0', 'up'], check=True)
            elif self.system == 'windows':
                logging.warning("MAC address restoration on Windows requires additional setup")
                return False
            
            logging.info("Original MAC address restored")
            return True
        except Exception as e:
            logging.error(f"Error restoring MAC address: {str(e)}")
            return False

    def _randomize_browser_fingerprint(self) -> bool:
        """Randomize browser fingerprint."""
        try:
            # This is a simplified version - actual implementation would need more complex fingerprinting
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ]
            
            # Set random user agent
            os.environ['HTTP_USER_AGENT'] = random.choice(user_agents)
            
            logging.info("Browser fingerprint randomized")
            return True
        except Exception as e:
            logging.error(f"Error randomizing browser fingerprint: {str(e)}")
            return False

    def _hide_processes(self) -> bool:
        """Hide Hexa Assistant processes."""
        try:
            # This is a simplified version - actual implementation would require root/administrator privileges
            # and more sophisticated process hiding techniques
            if self.system == 'linux':
                # Use LD_PRELOAD to hide processes
                os.environ['LD_PRELOAD'] = '/usr/lib/libprocesshider.so'
            elif self.system == 'windows':
                # Windows process hiding would require driver-level modifications
                logging.warning("Process hiding on Windows requires additional setup")
                return False
            
            logging.info("Processes hidden successfully")
            return True
        except Exception as e:
            logging.error(f"Error hiding processes: {str(e)}")
            return False

    def _unhide_processes(self) -> bool:
        """Unhide Hexa Assistant processes."""
        try:
            if self.system == 'linux':
                if 'LD_PRELOAD' in os.environ:
                    del os.environ['LD_PRELOAD']
            elif self.system == 'windows':
                logging.warning("Process unhiding on Windows requires additional setup")
                return False
            
            logging.info("Processes unhidden successfully")
            return True
        except Exception as e:
            logging.error(f"Error unhiding processes: {str(e)}")
            return False

    def _hide_files(self) -> bool:
        """Hide Hexa Assistant files and directories."""
        try:
            # This is a simplified version - actual implementation would require more sophisticated file hiding
            if self.system == 'linux':
                # Use dot prefix for hidden files
                for file in Path('.').glob('hexa_*'):
                    file.rename(f'.{file.name}')
            elif self.system == 'windows':
                # Use hidden attribute
                for file in Path('.').glob('hexa_*'):
                    subprocess.run(['attrib', '+h', str(file)], check=True)
            
            logging.info("Files hidden successfully")
            return True
        except Exception as e:
            logging.error(f"Error hiding files: {str(e)}")
            return False

    def _unhide_files(self) -> bool:
        """Unhide Hexa Assistant files and directories."""
        try:
            if self.system == 'linux':
                # Remove dot prefix
                for file in Path('.').glob('.hexa_*'):
                    file.rename(file.name[1:])
            elif self.system == 'windows':
                # Remove hidden attribute
                for file in Path('.').glob('hexa_*'):
                    subprocess.run(['attrib', '-h', str(file)], check=True)
            
            logging.info("Files unhidden successfully")
            return True
        except Exception as e:
            logging.error(f"Error unhiding files: {str(e)}")
            return False

    def _clean_logs(self) -> bool:
        """Clean system and application logs."""
        try:
            if self.system == 'linux':
                # Clean system logs
                subprocess.run(['shred', '-u', '/var/log/*'], check=True)
                subprocess.run(['shred', '-u', '~/.bash_history'], check=True)
            elif self.system == 'windows':
                # Clean Windows logs
                subprocess.run(['wevtutil', 'cl', 'System'], check=True)
                subprocess.run(['wevtutil', 'cl', 'Application'], check=True)
                subprocess.run(['wevtutil', 'cl', 'Security'], check=True)
            
            logging.info("Logs cleaned successfully")
            return True
        except Exception as e:
            logging.error(f"Error cleaning logs: {str(e)}")
            return False

    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES."""
        try:
            cipher = AES.new(key, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            return cipher.nonce + tag + ciphertext
        except Exception as e:
            logging.error(f"Error encrypting data: {str(e)}")
            return b''

    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES."""
        try:
            nonce = encrypted_data[:16]
            tag = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            return cipher.decrypt_and_verify(ciphertext, tag)
        except Exception as e:
            logging.error(f"Error decrypting data: {str(e)}")
            return b''

    def _get_current_ip(self) -> str:
        """Get current IP address."""
        try:
            response = requests.get('https://api.ipify.org?format=json')
            return response.json()['ip']
        except Exception as e:
            logging.error(f"Error getting IP address: {str(e)}")
            return ''

    def rotate_ip(self) -> bool:
        """Rotate IP address using available methods."""
        try:
            if self.current_vpn:
                self._disconnect_vpn()
                return self._connect_vpn()
            elif self.tor_process:
                with Controller.from_port(port=9051) as controller:
                    controller.authenticate()
                    controller.signal(Signal.NEWNYM)
                return True
            return False
        except Exception as e:
            logging.error(f"Error rotating IP: {str(e)}")
            return False

    def get_status(self) -> Dict:
        """Get current stealth mode status."""
        return {
            'active': self.is_active,
            'vpn_connected': bool(self.current_vpn),
            'tor_running': bool(self.tor_process),
            'current_ip': self._get_current_ip(),
            'mac_spoofed': self.original_mac != self._get_current_mac(),
            'features': self.stealth_config
        }
