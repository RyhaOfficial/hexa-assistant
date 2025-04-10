import subprocess
import os
import platform
import logging
import shutil
import sys
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from urllib.request import urlretrieve
import tempfile
import zipfile
import tarfile
import git

# Configure logging
logging.basicConfig(
    filename='tool_installer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ToolInstaller:
    """Main class for managing tool installations across different platforms"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.logger = logging.getLogger(__name__)
        self.tools_config = self._load_tools_config()
        self.install_path = self._get_install_path()
        
    def _load_tools_config(self) -> Dict:
        """Load tool configuration from JSON file"""
        config_path = Path(__file__).parent / 'tools_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _get_install_path(self) -> Path:
        """Get the appropriate installation path based on platform"""
        if self.system == 'windows':
            return Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files'))
        elif self.system == 'darwin':  # macOS
            return Path('/usr/local/bin')
        else:  # Linux
            return Path('/usr/local/bin')
            
    def _get_package_manager(self) -> Tuple[str, List[str]]:
        """Get the appropriate package manager for the current platform"""
        if self.system == 'windows':
            return 'choco', ['choco', 'install', '-y']
        elif self.system == 'darwin':
            return 'brew', ['brew', 'install']
        else:  # Linux
            if os.path.exists('/usr/bin/apt'):
                return 'apt', ['sudo', 'apt-get', 'install', '-y']
            elif os.path.exists('/usr/bin/yum'):
                return 'yum', ['sudo', 'yum', 'install', '-y']
            elif os.path.exists('/usr/bin/pacman'):
                return 'pacman', ['sudo', 'pacman', '-S', '--noconfirm']
            else:
                raise Exception("Unsupported Linux distribution")
                
    def install_tool(self, tool_name: str, version: str = 'latest', 
                    silent: bool = False) -> bool:
        """
        Install a tool using the appropriate method for the platform
        
        Args:
            tool_name (str): Name of the tool to install
            version (str): Version to install (default: 'latest')
            silent (bool): Whether to run installation silently
            
        Returns:
            bool: True if installation successful, False otherwise
        """
        try:
            # Check if tool is already installed
            if self.check_tool_installed(tool_name):
                self.logger.info(f"Tool {tool_name} is already installed")
                return True
                
            # Get installation method from config
            tool_config = self.tools_config.get(tool_name, {})
            install_method = tool_config.get('install_method', 'package_manager')
            
            if install_method == 'package_manager':
                return self._install_via_package_manager(tool_name, silent)
            elif install_method == 'pip':
                return self._install_via_pip(tool_name, version, silent)
            elif install_method == 'github':
                return self._install_from_github(tool_name, version, silent)
            elif install_method == 'custom':
                return self._install_custom_tool(tool_name, tool_config, silent)
            else:
                self.logger.error(f"Unknown installation method for {tool_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing {tool_name}: {str(e)}")
            return False
            
    def _install_via_package_manager(self, tool_name: str, silent: bool) -> bool:
        """Install tool using system package manager"""
        try:
            pkg_manager, cmd = self._get_package_manager()
            cmd.append(tool_name)
            
            if silent:
                cmd.append('-y')
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {tool_name} via {pkg_manager}")
                return True
            else:
                self.logger.error(f"Failed to install {tool_name}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing via package manager: {str(e)}")
            return False
            
    def _install_via_pip(self, tool_name: str, version: str, silent: bool) -> bool:
        """Install Python package via pip"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install']
            
            if version != 'latest':
                cmd.append(f"{tool_name}=={version}")
            else:
                cmd.append(tool_name)
                
            if silent:
                cmd.append('--quiet')
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {tool_name} via pip")
                return True
            else:
                self.logger.error(f"Failed to install {tool_name}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing via pip: {str(e)}")
            return False
            
    def _install_from_github(self, tool_name: str, version: str, silent: bool) -> bool:
        """Install tool from GitHub repository"""
        try:
            tool_config = self.tools_config[tool_name]
            repo_url = tool_config['github_url']
            
            # Create temporary directory for cloning
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone repository
                git.Repo.clone_from(repo_url, temp_dir)
                
                # Checkout specific version if specified
                if version != 'latest':
                    repo = git.Repo(temp_dir)
                    repo.git.checkout(version)
                    
                # Run installation commands
                install_commands = tool_config.get('install_commands', [])
                for cmd in install_commands:
                    subprocess.run(cmd, cwd=temp_dir, shell=True, check=True)
                    
            self.logger.info(f"Successfully installed {tool_name} from GitHub")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing from GitHub: {str(e)}")
            return False
            
    def _install_custom_tool(self, tool_name: str, tool_config: Dict, 
                           silent: bool) -> bool:
        """Install tool using custom installation method"""
        try:
            # Download the tool
            download_url = tool_config['download_url']
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                urlretrieve(download_url, temp_file.name)
                
            # Extract if needed
            if download_url.endswith('.zip'):
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    zip_ref.extractall(self.install_path)
            elif download_url.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(temp_file.name, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.install_path)
                    
            # Run post-installation commands
            post_install = tool_config.get('post_install', [])
            for cmd in post_install:
                subprocess.run(cmd, shell=True, check=True)
                
            self.logger.info(f"Successfully installed {tool_name} via custom method")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing custom tool: {str(e)}")
            return False
            
    def uninstall_tool(self, tool_name: str) -> bool:
        """
        Uninstall a tool
        
        Args:
            tool_name (str): Name of the tool to uninstall
            
        Returns:
            bool: True if uninstallation successful, False otherwise
        """
        try:
            tool_config = self.tools_config.get(tool_name, {})
            uninstall_method = tool_config.get('uninstall_method', 'package_manager')
            
            if uninstall_method == 'package_manager':
                pkg_manager, cmd = self._get_package_manager()
                cmd[2] = 'remove'  # Change install to remove
                cmd.append(tool_name)
                
            elif uninstall_method == 'pip':
                cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', tool_name]
                
            else:
                self.logger.error(f"Unknown uninstallation method for {tool_name}")
                return False
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully uninstalled {tool_name}")
                return True
            else:
                self.logger.error(f"Failed to uninstall {tool_name}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uninstalling {tool_name}: {str(e)}")
            return False
            
    def check_tool_installed(self, tool_name: str) -> bool:
        """
        Check if a tool is installed
        
        Args:
            tool_name (str): Name of the tool to check
            
        Returns:
            bool: True if tool is installed, False otherwise
        """
        try:
            tool_config = self.tools_config.get(tool_name, {})
            check_command = tool_config.get('check_command', f"which {tool_name}")
            
            result = subprocess.run(check_command, shell=True, capture_output=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Error checking tool installation: {str(e)}")
            return False
            
    def get_tool_version(self, tool_name: str) -> Optional[str]:
        """
        Get the installed version of a tool
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[str]: Version string if found, None otherwise
        """
        try:
            tool_config = self.tools_config.get(tool_name, {})
            version_command = tool_config.get('version_command', f"{tool_name} --version")
            
            result = subprocess.run(version_command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting tool version: {str(e)}")
            return None
            
    def update_tool(self, tool_name: str) -> bool:
        """
        Update a tool to the latest version
        
        Args:
            tool_name (str): Name of the tool to update
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            tool_config = self.tools_config.get(tool_name, {})
            update_method = tool_config.get('update_method', 'package_manager')
            
            if update_method == 'package_manager':
                pkg_manager, cmd = self._get_package_manager()
                cmd[2] = 'upgrade'  # Change install to upgrade
                cmd.append(tool_name)
                
            elif update_method == 'pip':
                cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', tool_name]
                
            else:
                self.logger.error(f"Unknown update method for {tool_name}")
                return False
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully updated {tool_name}")
                return True
            else:
                self.logger.error(f"Failed to update {tool_name}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating {tool_name}: {str(e)}")
            return False
            
    def manage_dependencies(self, tool_name: str) -> bool:
        """
        Install dependencies for a tool
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            bool: True if dependencies installed successfully, False otherwise
        """
        try:
            tool_config = self.tools_config.get(tool_name, {})
            dependencies = tool_config.get('dependencies', [])
            
            for dep in dependencies:
                if not self.install_tool(dep, silent=True):
                    self.logger.error(f"Failed to install dependency: {dep}")
                    return False
                    
            self.logger.info(f"Successfully installed dependencies for {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error managing dependencies: {str(e)}")
            return False
