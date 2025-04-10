import subprocess
import os
import platform
import logging
import psutil
import json
import shutil
import time
from typing import Dict, List, Optional, Union
from pathlib import Path

# Set up logging
logging.basicConfig(
    filename='vm_manager.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VMManager:
    """Manages virtual machines for Hexa Assistant across multiple platforms."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.vm_config_path = Path("config/vm_config.json")
        self.vm_config = self._load_vm_config()
        self.virtualization_platform = self._detect_virtualization_platform()
        logging.info(f"Initialized VMManager on {self.system} using {self.virtualization_platform}")

    def _load_vm_config(self) -> Dict:
        """Load VM configuration from JSON file."""
        try:
            if self.vm_config_path.exists():
                with open(self.vm_config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading VM config: {str(e)}")
            return {}

    def _save_vm_config(self) -> bool:
        """Save VM configuration to JSON file."""
        try:
            os.makedirs(self.vm_config_path.parent, exist_ok=True)
            with open(self.vm_config_path, 'w') as f:
                json.dump(self.vm_config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Error saving VM config: {str(e)}")
            return False

    def _detect_virtualization_platform(self) -> str:
        """Detect available virtualization platform on the system."""
        if self.system == 'windows':
            if self._check_hyperv_available():
                return 'hyperv'
            return 'virtualbox'
        elif self.system == 'linux':
            if self._check_qemu_available():
                return 'qemu'
            return 'virtualbox'
        elif self.system == 'darwin':  # macOS
            return 'virtualbox'
        return 'virtualbox'  # Default fallback

    def _check_hyperv_available(self) -> bool:
        """Check if Hyper-V is available on Windows."""
        try:
            result = subprocess.run(['powershell', 'Get-WindowsOptionalFeature', '-FeatureName', 'Microsoft-Hyper-V-All', '-Online'],
                                  capture_output=True, text=True)
            return 'Enabled' in result.stdout
        except Exception:
            return False

    def _check_qemu_available(self) -> bool:
        """Check if QEMU is available on Linux."""
        try:
            subprocess.run(['qemu-system-x86_64', '--version'], capture_output=True)
            return True
        except Exception:
            return False

    def create_vm(self, os_name: str, resources: Dict, vm_name: str, platform: Optional[str] = None) -> bool:
        """Create a new VM with specified configuration."""
        try:
            if platform:
                self.virtualization_platform = platform

            # Validate resources
            if not self._validate_resources(resources):
                logging.error(f"Invalid resource configuration for VM {vm_name}")
                return False

            # Create VM based on platform
            if self.virtualization_platform == 'virtualbox':
                return self._create_virtualbox_vm(os_name, resources, vm_name)
            elif self.virtualization_platform == 'hyperv':
                return self._create_hyperv_vm(os_name, resources, vm_name)
            elif self.virtualization_platform == 'qemu':
                return self._create_qemu_vm(os_name, resources, vm_name)
            
            logging.error(f"Unsupported virtualization platform: {self.virtualization_platform}")
            return False
        except Exception as e:
            logging.error(f"Error creating VM {vm_name}: {str(e)}")
            return False

    def _validate_resources(self, resources: Dict) -> bool:
        """Validate VM resource configuration."""
        required = ['ram', 'cpu', 'disk']
        return all(key in resources for key in required)

    def _create_virtualbox_vm(self, os_name: str, resources: Dict, vm_name: str) -> bool:
        """Create a VM using VirtualBox."""
        try:
            # Create VM
            subprocess.run(['VBoxManage', 'createvm', '--name', vm_name, '--register'], check=True)
            
            # Configure resources
            subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--memory', str(resources['ram'])], check=True)
            subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--cpus', str(resources['cpu'])], check=True)
            
            # Create and attach storage
            subprocess.run(['VBoxManage', 'createhd', '--filename', f'{vm_name}.vdi', 
                          '--size', str(resources['disk']), '--format', 'VDI'], check=True)
            subprocess.run(['VBoxManage', 'storagectl', vm_name, '--name', 'SATA', 
                          '--add', 'sata', '--controller', 'IntelAhci'], check=True)
            subprocess.run(['VBoxManage', 'storageattach', vm_name, '--storagectl', 'SATA',
                          '--port', '0', '--device', '0', '--type', 'hdd',
                          '--medium', f'{vm_name}.vdi'], check=True)
            
            # Configure OS type
            subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--ostype', os_name], check=True)
            
            # Save VM configuration
            self.vm_config[vm_name] = {
                'platform': 'virtualbox',
                'os': os_name,
                'resources': resources,
                'status': 'created'
            }
            self._save_vm_config()
            
            logging.info(f"Successfully created VirtualBox VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating VirtualBox VM: {str(e)}")
            return False

    def _create_hyperv_vm(self, os_name: str, resources: Dict, vm_name: str) -> bool:
        """Create a VM using Hyper-V."""
        try:
            # Create VM using PowerShell commands
            subprocess.run(['powershell', 'New-VM', '-Name', vm_name, '-MemoryStartupBytes', 
                          f"{resources['ram']}MB", '-Generation', '2'], check=True)
            
            # Configure CPU
            subprocess.run(['powershell', 'Set-VMProcessor', '-VMName', vm_name, 
                          '-Count', str(resources['cpu'])], check=True)
            
            # Create and attach virtual disk
            subprocess.run(['powershell', 'New-VHD', '-Path', f"{vm_name}.vhdx", 
                          '-SizeBytes', f"{resources['disk']}GB", '-Dynamic'], check=True)
            subprocess.run(['powershell', 'Add-VMHardDiskDrive', '-VMName', vm_name, 
                          '-Path', f"{vm_name}.vhdx"], check=True)
            
            # Save VM configuration
            self.vm_config[vm_name] = {
                'platform': 'hyperv',
                'os': os_name,
                'resources': resources,
                'status': 'created'
            }
            self._save_vm_config()
            
            logging.info(f"Successfully created Hyper-V VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating Hyper-V VM: {str(e)}")
            return False

    def _create_qemu_vm(self, os_name: str, resources: Dict, vm_name: str) -> bool:
        """Create a VM using QEMU."""
        try:
            # Create disk image
            subprocess.run(['qemu-img', 'create', '-f', 'qcow2', f'{vm_name}.qcow2', 
                          f"{resources['disk']}G"], check=True)
            
            # Save VM configuration
            self.vm_config[vm_name] = {
                'platform': 'qemu',
                'os': os_name,
                'resources': resources,
                'status': 'created',
                'disk_image': f'{vm_name}.qcow2'
            }
            self._save_vm_config()
            
            logging.info(f"Successfully created QEMU VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating QEMU VM: {str(e)}")
            return False

    def start_vm(self, vm_name: str) -> bool:
        """Start a VM."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'startvm', vm_name], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Start-VM', '-Name', vm_name], check=True)
            elif platform == 'qemu':
                # QEMU requires more complex startup with OS image
                logging.error("QEMU VM startup requires OS image configuration")
                return False

            self.vm_config[vm_name]['status'] = 'running'
            self._save_vm_config()
            logging.info(f"Successfully started VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error starting VM {vm_name}: {str(e)}")
            return False

    def stop_vm(self, vm_name: str) -> bool:
        """Stop a VM."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'controlvm', vm_name, 'poweroff'], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Stop-VM', '-Name', vm_name, '-Force'], check=True)
            elif platform == 'qemu':
                # QEMU VMs need to be killed as they run as processes
                subprocess.run(['pkill', '-f', vm_name], check=True)

            self.vm_config[vm_name]['status'] = 'stopped'
            self._save_vm_config()
            logging.info(f"Successfully stopped VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error stopping VM {vm_name}: {str(e)}")
            return False

    def pause_vm(self, vm_name: str) -> bool:
        """Pause a VM."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'controlvm', vm_name, 'pause'], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Suspend-VM', '-Name', vm_name], check=True)
            elif platform == 'qemu':
                logging.error("QEMU pause not implemented")
                return False

            self.vm_config[vm_name]['status'] = 'paused'
            self._save_vm_config()
            logging.info(f"Successfully paused VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error pausing VM {vm_name}: {str(e)}")
            return False

    def take_snapshot(self, vm_name: str, snapshot_name: str) -> bool:
        """Take a snapshot of a VM."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'snapshot', vm_name, 'take', snapshot_name], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Checkpoint-VM', '-Name', vm_name, 
                              '-SnapshotName', snapshot_name], check=True)
            elif platform == 'qemu':
                logging.error("QEMU snapshots not implemented")
                return False

            if 'snapshots' not in self.vm_config[vm_name]:
                self.vm_config[vm_name]['snapshots'] = []
            self.vm_config[vm_name]['snapshots'].append(snapshot_name)
            self._save_vm_config()
            logging.info(f"Successfully took snapshot {snapshot_name} of VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error taking snapshot of VM {vm_name}: {str(e)}")
            return False

    def restore_snapshot(self, vm_name: str, snapshot_name: str) -> bool:
        """Restore a VM from a snapshot."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'snapshot', vm_name, 'restore', snapshot_name], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Restore-VMSnapshot', '-VMName', vm_name,
                              '-Name', snapshot_name, '-Confirm:$false'], check=True)
            elif platform == 'qemu':
                logging.error("QEMU snapshot restore not implemented")
                return False

            logging.info(f"Successfully restored snapshot {snapshot_name} of VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error restoring snapshot of VM {vm_name}: {str(e)}")
            return False

    def resize_vm(self, vm_name: str, resources: Dict) -> bool:
        """Resize VM resources."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            if not self._validate_resources(resources):
                logging.error(f"Invalid resource configuration for VM {vm_name}")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--memory', str(resources['ram'])], check=True)
                subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--cpus', str(resources['cpu'])], check=True)
                # Note: Resizing disk requires additional steps with VBoxManage
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Set-VMMemory', '-VMName', vm_name,
                              '-StartupBytes', f"{resources['ram']}MB"], check=True)
                subprocess.run(['powershell', 'Set-VMProcessor', '-VMName', vm_name,
                              '-Count', str(resources['cpu'])], check=True)
            elif platform == 'qemu':
                logging.error("QEMU resource resizing not implemented")
                return False

            self.vm_config[vm_name]['resources'] = resources
            self._save_vm_config()
            logging.info(f"Successfully resized VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error resizing VM {vm_name}: {str(e)}")
            return False

    def check_vm_status(self, vm_name: str) -> str:
        """Check the current status of a VM."""
        try:
            if vm_name not in self.vm_config:
                return "not_found"

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                result = subprocess.run(['VBoxManage', 'showvminfo', vm_name, '--machinereadable'],
                                     capture_output=True, text=True)
                if 'VMState="running"' in result.stdout:
                    return "running"
                elif 'VMState="paused"' in result.stdout:
                    return "paused"
                return "stopped"
            elif platform == 'hyperv':
                result = subprocess.run(['powershell', 'Get-VM', '-Name', vm_name],
                                     capture_output=True, text=True)
                if 'Running' in result.stdout:
                    return "running"
                elif 'Paused' in result.stdout:
                    return "paused"
                return "stopped"
            elif platform == 'qemu':
                # Check if QEMU process is running
                result = subprocess.run(['pgrep', '-f', vm_name], capture_output=True)
                return "running" if result.returncode == 0 else "stopped"

            return "unknown"
        except Exception as e:
            logging.error(f"Error checking status of VM {vm_name}: {str(e)}")
            return "error"

    def manage_vm_networking(self, vm_name: str, network_config: Dict) -> bool:
        """Configure VM networking."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                # Configure network adapter
                subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--nic1', network_config.get('type', 'nat')], check=True)
                if 'mac' in network_config:
                    subprocess.run(['VBoxManage', 'modifyvm', vm_name, '--macaddress1', network_config['mac']], check=True)
            elif platform == 'hyperv':
                # Configure network adapter
                subprocess.run(['powershell', 'Set-VMNetworkAdapter', '-VMName', vm_name,
                              '-StaticMacAddress', network_config.get('mac', '')], check=True)
            elif platform == 'qemu':
                logging.error("QEMU networking configuration not implemented")
                return False

            if 'networking' not in self.vm_config[vm_name]:
                self.vm_config[vm_name]['networking'] = {}
            self.vm_config[vm_name]['networking'].update(network_config)
            self._save_vm_config()
            logging.info(f"Successfully configured networking for VM: {vm_name}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error configuring networking for VM {vm_name}: {str(e)}")
            return False

    def monitor_vm(self, vm_name: str) -> Dict:
        """Monitor VM resource usage."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return {}

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                result = subprocess.run(['VBoxManage', 'metrics', 'collect', '--period', '1',
                                      '--samples', '1', vm_name], capture_output=True, text=True)
                # Parse metrics from result.stdout
                metrics = self._parse_virtualbox_metrics(result.stdout)
            elif platform == 'hyperv':
                result = subprocess.run(['powershell', 'Get-VM', '-Name', vm_name, '|',
                                      'Select-Object', 'CPUUsage,MemoryAssigned,MemoryDemand'],
                                     capture_output=True, text=True)
                metrics = self._parse_hyperv_metrics(result.stdout)
            elif platform == 'qemu':
                logging.error("QEMU monitoring not implemented")
                return {}

            return metrics
        except Exception as e:
            logging.error(f"Error monitoring VM {vm_name}: {str(e)}")
            return {}

    def _parse_virtualbox_metrics(self, metrics_output: str) -> Dict:
        """Parse VirtualBox metrics output."""
        metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_in': 0,
            'network_out': 0
        }
        # Parse metrics from the output string
        # This is a simplified version - actual implementation would need more robust parsing
        for line in metrics_output.split('\n'):
            if 'CPU/Load/User' in line:
                metrics['cpu_usage'] = float(line.split('=')[1].strip())
            elif 'Guest/RAM/Usage' in line:
                metrics['memory_usage'] = float(line.split('=')[1].strip())
        return metrics

    def _parse_hyperv_metrics(self, metrics_output: str) -> Dict:
        """Parse Hyper-V metrics output."""
        metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_in': 0,
            'network_out': 0
        }
        # Parse metrics from the output string
        # This is a simplified version - actual implementation would need more robust parsing
        for line in metrics_output.split('\n'):
            if 'CPUUsage' in line:
                metrics['cpu_usage'] = float(line.split(':')[1].strip())
            elif 'MemoryAssigned' in line:
                metrics['memory_usage'] = float(line.split(':')[1].strip())
        return metrics

    def list_vms(self) -> List[str]:
        """List all configured VMs."""
        return list(self.vm_config.keys())

    def delete_vm(self, vm_name: str) -> bool:
        """Delete a VM and its associated files."""
        try:
            if vm_name not in self.vm_config:
                logging.error(f"VM {vm_name} not found in configuration")
                return False

            platform = self.vm_config[vm_name]['platform']
            if platform == 'virtualbox':
                subprocess.run(['VBoxManage', 'unregistervm', vm_name, '--delete'], check=True)
            elif platform == 'hyperv':
                subprocess.run(['powershell', 'Remove-VM', '-Name', vm_name, '-Force'], check=True)
            elif platform == 'qemu':
                # Delete QEMU disk image
                if 'disk_image' in self.vm_config[vm_name]:
                    os.remove(self.vm_config[vm_name]['disk_image'])

            # Remove from configuration
            del self.vm_config[vm_name]
            self._save_vm_config()
            logging.info(f"Successfully deleted VM: {vm_name}")
            return True
        except Exception as e:
            logging.error(f"Error deleting VM {vm_name}: {str(e)}")
            return False
