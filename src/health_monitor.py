"""
Health Monitoring and Self-Healing System
Monitors system health and attempts automatic recovery from common failures
"""

import os
import sys
import json
from datetime import datetime, timedelta
from logger import setup_logger, log_error_with_context
from alert_system import alert_system
from config import BASE_DIR

logger = setup_logger(__name__)

class HealthMonitor:
    """Monitor system health and trigger self-healing actions"""
    
    def __init__(self):
        self.health_file = os.path.join(BASE_DIR, 'reports', 'system_health.json')
        self.metrics = self.load_metrics()
    
    def load_metrics(self):
        """Load health metrics from file"""
        if os.path.exists(self.health_file):
            try:
                with open(self.health_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load health metrics: {e}")
        
        # Default metrics
        return {
            'last_successful_run': None,
            'last_failed_run': None,
            'consecutive_failures': 0,
            'consecutive_successes': 0,
            'total_runs': 0,
            'total_successes': 0,
            'total_failures': 0,
            'uptime_percentage': 100.0,
            'last_health_check': None,
            'issues': []
        }
    
    def save_metrics(self):
        """Save health metrics to file"""
        try:
            os.makedirs(os.path.dirname(self.health_file), exist_ok=True)
            with open(self.health_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save health metrics: {e}")
    
    def record_run_result(self, success, error_msg=None):
        """
        Record the result of a system run
        
        Args:
            success: Whether the run was successful
            error_msg: Error message if failed (optional)
        """
        now = datetime.now().isoformat()
        
        self.metrics['total_runs'] += 1
        
        if success:
            self.metrics['last_successful_run'] = now
            self.metrics['consecutive_successes'] += 1
            self.metrics['consecutive_failures'] = 0
            self.metrics['total_successes'] += 1
            
            # Clear issues on success
            self.metrics['issues'] = []
            
            logger.info("Run recorded as SUCCESS")
        else:
            self.metrics['last_failed_run'] = now
            self.metrics['consecutive_failures'] += 1
            self.metrics['consecutive_successes'] = 0
            self.metrics['total_failures'] += 1
            
            # Add issue
            if error_msg:
                self.metrics['issues'].append({
                    'timestamp': now,
                    'error': error_msg
                })
            
            logger.error(f"Run recorded as FAILURE: {error_msg}")
        
        # Calculate uptime
        if self.metrics['total_runs'] > 0:
            self.metrics['uptime_percentage'] = (
                self.metrics['total_successes'] / self.metrics['total_runs']
            ) * 100
        
        self.metrics['last_health_check'] = now
        self.save_metrics()
        
        # Check if self-healing is needed
        if self.metrics['consecutive_failures'] >= 3:
            logger.critical(f"System has failed {self.metrics['consecutive_failures']} times in a row")
            alert_system.send_critical_alert(
                f"System has failed {self.metrics['consecutive_failures']} consecutive times",
                {'last_error': error_msg}
            )
            self.attempt_self_heal()
    
    def check_health(self):
        """
        Perform comprehensive health check
        
        Returns:
            Dict with health status and issues
        """
        logger.info("Performing health check...")
        
        issues = []
        warnings = []
        
        # Check 1: Recent failures
        if self.metrics['consecutive_failures'] > 0:
            issues.append(f"System has {self.metrics['consecutive_failures']} consecutive failures")
        
        # Check 2: Uptime percentage
        if self.metrics['uptime_percentage'] < 90:
            issues.append(f"Uptime is low: {self.metrics['uptime_percentage']:.1f}%")
        elif self.metrics['uptime_percentage'] < 95:
            warnings.append(f"Uptime below target: {self.metrics['uptime_percentage']:.1f}%")
        
        # Check 3: Last successful run
        if self.metrics['last_successful_run']:
            last_success = datetime.fromisoformat(self.metrics['last_successful_run'])
            hours_since_success = (datetime.now() - last_success).total_seconds() / 3600
            
            if hours_since_success > 8:
                issues.append(f"No successful run in {hours_since_success:.1f} hours")
            elif hours_since_success > 6:
                warnings.append(f"Last successful run was {hours_since_success:.1f} hours ago")
        
        # Check 4: Required files exist
        required_files = [
            'eth_4h_data.csv',
            'eth_1m_data.csv'
        ]
        
        for filename in required_files:
            filepath = os.path.join(BASE_DIR, filename)
            if not os.path.exists(filepath):
                issues.append(f"Required file missing: {filename}")
        
        # Check 5: Logs directory exists and is writable
        logs_dir = os.path.join(BASE_DIR, 'logs')
        if not os.path.exists(logs_dir):
            warnings.append("Logs directory does not exist")
        elif not os.access(logs_dir, os.W_OK):
            issues.append("Logs directory is not writable")
        
        # Determine overall status
        if issues:
            status = 'critical'
        elif warnings:
            status = 'degraded'
        else:
            status = 'healthy'
        
        health_report = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'uptime': f"{self.metrics['uptime_percentage']:.1f}%",
                'total_runs': self.metrics['total_runs'],
                'consecutive_failures': self.metrics['consecutive_failures'],
                'last_success': self.metrics['last_successful_run']
            },
            'issues': issues,
            'warnings': warnings
        }
        
        logger.info(f"Health check complete: {status.upper()}")
        
        # Send health check alert
        alert_system.send_health_check_alert(status, health_report['metrics'])
        
        return health_report
    
    def attempt_self_heal(self):
        """
        Attempt to automatically fix common issues
        """
        logger.info("Attempting self-healing...")
        
        healing_actions = []
        
        # Action 1: Ensure required directories exist
        try:
            required_dirs = [
                os.path.join(BASE_DIR, 'logs'),
                os.path.join(BASE_DIR, 'reports'),
                os.path.join(BASE_DIR, 'reports', 'latest'),
                os.path.join(BASE_DIR, 'data')
            ]
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    healing_actions.append(f"Created missing directory: {dir_path}")
                    logger.info(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
        
        # Action 2: Check and fix file permissions
        try:
            logs_dir = os.path.join(BASE_DIR, 'logs')
            if os.path.exists(logs_dir) and not os.access(logs_dir, os.W_OK):
                os.chmod(logs_dir, 0o755)
                healing_actions.append(f"Fixed permissions for logs directory")
                logger.info("Fixed logs directory permissions")
        except Exception as e:
            logger.error(f"Failed to fix permissions: {e}")
        
        # Action 3: Clear old log files if disk space is an issue
        try:
            logs_dir = os.path.join(BASE_DIR, 'logs')
            if os.path.exists(logs_dir):
                # Remove logs older than 30 days
                cutoff_date = datetime.now() - timedelta(days=30)
                
                for filename in os.listdir(logs_dir):
                    filepath = os.path.join(logs_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            healing_actions.append(f"Removed old log file: {filename}")
                            logger.info(f"Removed old log: {filename}")
        except Exception as e:
            logger.error(f"Failed to clean old logs: {e}")
        
        # Action 4: Reset consecutive failure counter if we've taken action
        if healing_actions:
            logger.info(f"Self-healing completed {len(healing_actions)} actions")
            alert_system.send_info_alert(
                f"Self-healing performed {len(healing_actions)} actions:\n" +
                "\n".join(f"• {action}" for action in healing_actions)
            )
            
            # Give system one more chance before alerting again
            if self.metrics['consecutive_failures'] >= 5:
                self.metrics['consecutive_failures'] = 2
                self.save_metrics()
        else:
            logger.warning("Self-healing found no actions to perform")
            alert_system.send_warning_alert(
                "Self-healing attempted but found no fixable issues - manual intervention may be required"
            )
    
    def get_status_summary(self):
        """
        Get a human-readable status summary
        
        Returns:
            String with status summary
        """
        if not self.metrics['total_runs']:
            return "System has not run yet"
        
        summary = f"""System Health Summary:
        
Status: {'🟢 HEALTHY' if self.metrics['consecutive_failures'] == 0 else '🔴 UNHEALTHY'}
Uptime: {self.metrics['uptime_percentage']:.1f}%
Total Runs: {self.metrics['total_runs']} ({self.metrics['total_successes']} success, {self.metrics['total_failures']} failed)
Consecutive Failures: {self.metrics['consecutive_failures']}
Last Success: {self.metrics['last_successful_run'] or 'Never'}
Last Failure: {self.metrics['last_failed_run'] or 'Never'}
"""
        
        if self.metrics['issues']:
            summary += "\nRecent Issues:\n"
            for issue in self.metrics['issues'][-5:]:  # Last 5 issues
                summary += f"  - [{issue['timestamp']}] {issue['error']}\n"
        
        return summary

def main():
    """Main function for standalone health check"""
    logger.info("=" * 70)
    logger.info("SYSTEM HEALTH CHECK")
    logger.info("=" * 70)
    
    monitor = HealthMonitor()
    health_report = monitor.check_health()
    
    print("\n" + "=" * 70)
    print("HEALTH CHECK RESULTS")
    print("=" * 70)
    print(f"\nStatus: {health_report['status'].upper()}")
    print(f"\nMetrics:")
    for key, value in health_report['metrics'].items():
        print(f"  {key}: {value}")
    
    if health_report['issues']:
        print(f"\nIssues ({len(health_report['issues'])}):")
        for issue in health_report['issues']:
            print(f"  ❌ {issue}")
    
    if health_report['warnings']:
        print(f"\nWarnings ({len(health_report['warnings'])}):")
        for warning in health_report['warnings']:
            print(f"  ⚠️  {warning}")
    
    if not health_report['issues'] and not health_report['warnings']:
        print("\n✅ All checks passed!")
    
    print("\n" + monitor.get_status_summary())
    
    return 0 if health_report['status'] == 'healthy' else 1

if __name__ == '__main__':
    sys.exit(main())
