import psutil
import subprocess
import platform
import logging
import time
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='process_controller.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ProcessInfo:
    """Data class to store process information"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: float
    username: str

class ProcessController:
    """Main class for process management and control"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.logger = logging.getLogger(__name__)
        self.monitoring_processes = set()
        
    def start_process(self, command: str, args: List[str] = None) -> bool:
        """
        Start a new process with the given command and arguments
        
        Args:
            command (str): The command to execute
            args (List[str], optional): List of arguments for the command
            
        Returns:
            bool: True if process started successfully, False otherwise
        """
        try:
            if args is None:
                args = []
            
            # Start the process
            process = subprocess.Popen([command] + args)
            
            # Verify process is running
            if psutil.pid_exists(process.pid):
                self.logger.info(f"Started process: {command} with PID: {process.pid}")
                return True
            else:
                self.logger.error(f"Failed to start process: {command}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting process {command}: {str(e)}")
            return False
            
    def stop_process(self, pid: int, force: bool = False) -> bool:
        """
        Stop a process by its PID
        
        Args:
            pid (int): Process ID to stop
            force (bool): Whether to force kill the process
            
        Returns:
            bool: True if process stopped successfully, False otherwise
        """
        try:
            if not psutil.pid_exists(pid):
                self.logger.warning(f"Process with PID {pid} not found")
                return False
                
            process = psutil.Process(pid)
            
            if force:
                if self.system == 'windows':
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)])
                else:
                    process.kill()
            else:
                process.terminate()
                
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                if force:
                    process.kill()
                else:
                    return False
                    
            self.logger.info(f"Stopped process with PID: {pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping process {pid}: {str(e)}")
            return False
            
    def list_processes(self, name_filter: Optional[str] = None) -> List[ProcessInfo]:
        """
        List all running processes, optionally filtered by name
        
        Args:
            name_filter (str, optional): Filter processes by name
            
        Returns:
            List[ProcessInfo]: List of process information
        """
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 
                                          'memory_percent', 'create_time', 'username']):
                try:
                    if name_filter and name_filter.lower() not in proc.info['name'].lower():
                        continue
                        
                    process_info = ProcessInfo(
                        pid=proc.info['pid'],
                        name=proc.info['name'],
                        status=proc.info['status'],
                        cpu_percent=proc.info['cpu_percent'],
                        memory_percent=proc.info['memory_percent'],
                        create_time=proc.info['create_time'],
                        username=proc.info['username']
                    )
                    processes.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error listing processes: {str(e)}")
            
        return processes
        
    def monitor_process(self, pid: int, interval: float = 1.0, 
                       cpu_threshold: float = 80.0, 
                       memory_threshold: float = 80.0) -> Dict[str, Union[float, str]]:
        """
        Monitor a specific process and its resource usage
        
        Args:
            pid (int): Process ID to monitor
            interval (float): Monitoring interval in seconds
            cpu_threshold (float): CPU usage threshold for alerts
            memory_threshold (float): Memory usage threshold for alerts
            
        Returns:
            Dict[str, Union[float, str]]: Process monitoring data
        """
        try:
            if not psutil.pid_exists(pid):
                return {"error": "Process not found"}
                
            process = psutil.Process(pid)
            self.monitoring_processes.add(pid)
            
            while pid in self.monitoring_processes:
                try:
                    cpu_percent = process.cpu_percent(interval=interval)
                    memory_percent = process.memory_percent()
                    
                    if cpu_percent > cpu_threshold:
                        self.alert_user(f"High CPU usage ({cpu_percent}%) for process {process.name()}")
                        
                    if memory_percent > memory_threshold:
                        self.alert_user(f"High memory usage ({memory_percent}%) for process {process.name()}")
                        
                    return {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "status": process.status(),
                        "threads": process.num_threads(),
                        "io_counters": process.io_counters()._asdict() if hasattr(process, 'io_counters') else None
                    }
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.monitoring_processes.remove(pid)
                    return {"error": "Process no longer exists"}
                    
        except Exception as e:
            self.logger.error(f"Error monitoring process {pid}: {str(e)}")
            return {"error": str(e)}
            
    def check_process_status(self, pid: int) -> str:
        """
        Check the status of a process
        
        Args:
            pid (int): Process ID to check
            
        Returns:
            str: Process status or error message
        """
        try:
            if not psutil.pid_exists(pid):
                return "Process not found"
                
            process = psutil.Process(pid)
            return process.status()
            
        except Exception as e:
            self.logger.error(f"Error checking process status {pid}: {str(e)}")
            return f"Error: {str(e)}"
            
    def pause_process(self, pid: int) -> bool:
        """
        Pause a running process
        
        Args:
            pid (int): Process ID to pause
            
        Returns:
            bool: True if process paused successfully, False otherwise
        """
        try:
            if self.system == 'windows':
                self.logger.warning("Process pausing not supported on Windows")
                return False
                
            if not psutil.pid_exists(pid):
                return False
                
            process = psutil.Process(pid)
            process.suspend()
            return True
            
        except Exception as e:
            self.logger.error(f"Error pausing process {pid}: {str(e)}")
            return False
            
    def resume_process(self, pid: int) -> bool:
        """
        Resume a paused process
        
        Args:
            pid (int): Process ID to resume
            
        Returns:
            bool: True if process resumed successfully, False otherwise
        """
        try:
            if self.system == 'windows':
                self.logger.warning("Process resuming not supported on Windows")
                return False
                
            if not psutil.pid_exists(pid):
                return False
                
            process = psutil.Process(pid)
            process.resume()
            return True
            
        except Exception as e:
            self.logger.error(f"Error resuming process {pid}: {str(e)}")
            return False
            
    def send_signal(self, pid: int, signal: str) -> bool:
        """
        Send a signal to a process
        
        Args:
            pid (int): Process ID to send signal to
            signal (str): Signal to send (e.g., 'SIGTERM', 'SIGKILL')
            
        Returns:
            bool: True if signal sent successfully, False otherwise
        """
        try:
            if not psutil.pid_exists(pid):
                return False
                
            process = psutil.Process(pid)
            if hasattr(process, signal):
                getattr(process, signal)()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending signal {signal} to process {pid}: {str(e)}")
            return False
            
    def log_process_activity(self, activity: str) -> None:
        """
        Log process-related activity
        
        Args:
            activity (str): Activity description to log
        """
        self.logger.info(activity)
        
    def alert_user(self, message: str) -> None:
        """
        Send an alert to the user
        
        Args:
            message (str): Alert message to send
        """
        # This would integrate with Hexa's UI/voice system
        print(f"ALERT: {message}")
        self.logger.warning(f"User Alert: {message}")
        
    def stop_monitoring(self, pid: int) -> None:
        """
        Stop monitoring a specific process
        
        Args:
            pid (int): Process ID to stop monitoring
        """
        if pid in self.monitoring_processes:
            self.monitoring_processes.remove(pid)
            
    def get_process_details(self, pid: int) -> Optional[Dict]:
        """
        Get detailed information about a process
        
        Args:
            pid (int): Process ID to get details for
            
        Returns:
            Optional[Dict]: Process details or None if process not found
        """
        try:
            if not psutil.pid_exists(pid):
                return None
                
            process = psutil.Process(pid)
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S'),
                "username": process.username(),
                "cmdline": process.cmdline(),
                "threads": process.num_threads(),
                "connections": [conn._asdict() for conn in process.connections()] if hasattr(process, 'connections') else [],
                "io_counters": process.io_counters()._asdict() if hasattr(process, 'io_counters') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting process details {pid}: {str(e)}")
            return None
