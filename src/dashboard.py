"""
Performance Dashboard Generator
Creates a comprehensive HTML dashboard showing system performance metrics
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
from config import BASE_DIR
from track_accuracy_enhanced import EnhancedAccuracyTracker
from health_monitor import HealthMonitor

def generate_dashboard_html():
    """Generate HTML dashboard with performance metrics"""
    
    # Load data
    tracker = EnhancedAccuracyTracker()
    health_monitor = HealthMonitor()
    
    # Get summary stats
    summary = tracker.history.get('summary', {})
    health_metrics = health_monitor.metrics
    
    # Calculate metrics
    total_predictions = summary.get('total_predictions', 0)
    total_validations = summary.get('total_validations', 0)
    ensemble_error = summary.get('ensemble_avg_error_pct', 0)
    direction_accuracy = summary.get('directional_accuracy', 0)
    
    # System health
    uptime = health_metrics.get('uptime_percentage', 0)
    consecutive_failures = health_metrics.get('consecutive_failures', 0)
    total_runs = health_metrics.get('total_runs', 0)
    
    # Per-model performance
    linear_error = summary.get('linear_avg_error_pct', 0)
    poly_error = summary.get('polynomial_avg_error_pct', 0)
    rf_error = summary.get('random_forest_avg_error_pct', 0)
    
    # Status indicators
    health_status = "🟢 HEALTHY" if consecutive_failures == 0 else "🔴 UNHEALTHY"
    accuracy_status = "🟢 GOOD" if direction_accuracy >= 60 else ("🟡 FAIR" if direction_accuracy >= 50 else "🔴 POOR")
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETH Prediction System Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
            color: #667eea;
        }}
        
        .header p {{
            color: #666;
            font-size: 14px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .metric-subtitle {{
            font-size: 12px;
            color: #999;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .status-good {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-fair {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            font-size: 24px;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .model-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        
        .model-name {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .model-error {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        
        .model-label {{
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
            transition: width 0.3s ease;
        }}
        
        .timestamp {{
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 14px;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 ETH Price Prediction System Dashboard</h1>
            <p>Real-time performance metrics and system health monitoring</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">System Health</div>
                <div class="metric-value">{health_status}</div>
                <div class="metric-subtitle">Uptime: {uptime:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {uptime}%">{uptime:.1f}%</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Directional Accuracy</div>
                <div class="metric-value">{direction_accuracy:.1f}%</div>
                <div class="metric-subtitle">Based on {total_validations} validations</div>
                <span class="status-badge {('status-good' if direction_accuracy >= 60 else 'status-fair' if direction_accuracy >= 50 else 'status-poor')}">{accuracy_status}</span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Price Accuracy</div>
                <div class="metric-value">{100-ensemble_error:.1f}%</div>
                <div class="metric-subtitle">Avg error: {ensemble_error:.2f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {100-ensemble_error if ensemble_error < 100 else 0}%">{100-ensemble_error:.1f}%</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{total_predictions}</div>
                <div class="metric-subtitle">{total_validations} validated</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">System Runs</div>
                <div class="metric-value">{total_runs}</div>
                <div class="metric-subtitle">Consecutive failures: {consecutive_failures}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Last Success</div>
                <div class="metric-value">
                    {health_metrics.get('last_successful_run', 'Never')[:10] if health_metrics.get('last_successful_run') else 'Never'}
                </div>
                <div class="metric-subtitle">
                    {health_metrics.get('last_successful_run', 'Never')[11:16] if health_metrics.get('last_successful_run') else ''}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Model Performance Comparison</h2>
            <div class="model-comparison">
                <div class="model-card">
                    <div class="model-name">Linear Regression</div>
                    <div class="model-error">{100-linear_error:.1f}%</div>
                    <div class="model-label">Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {100-linear_error if linear_error < 100 else 0}%"></div>
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-name">Polynomial</div>
                    <div class="model-error">{100-poly_error:.1f}%</div>
                    <div class="model-label">Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {100-poly_error if poly_error < 100 else 0}%"></div>
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-name">Random Forest</div>
                    <div class="model-error">{100-rf_error:.1f}%</div>
                    <div class="model-label">Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {100-rf_error if rf_error < 100 else 0}%"></div>
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-name">Ensemble</div>
                    <div class="model-error">{100-ensemble_error:.1f}%</div>
                    <div class="model-label">Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {100-ensemble_error if ensemble_error < 100 else 0}%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>System Status</h2>
            <p style="margin-bottom: 15px;">
                <strong>Health Status:</strong> {health_status}<br>
                <strong>Accuracy Status:</strong> {accuracy_status}<br>
                <strong>Total Runs:</strong> {total_runs} ({health_metrics.get('total_successes', 0)} success, {health_metrics.get('total_failures', 0)} failed)<br>
                <strong>Consecutive Failures:</strong> {consecutive_failures}<br>
                <strong>Last Successful Run:</strong> {health_metrics.get('last_successful_run', 'Never')}<br>
                <strong>Last Failed Run:</strong> {health_metrics.get('last_failed_run', 'Never')}
            </p>
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>"""
    
    return html

def main():
    """Generate and save dashboard"""
    print("Generating performance dashboard...")
    
    html = generate_dashboard_html()
    
    # Save to reports directory
    dashboard_path = os.path.join(BASE_DIR, 'reports', 'dashboard.html')
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
    
    with open(dashboard_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Dashboard generated: {dashboard_path}")
    print(f"\nOpen in browser: file://{dashboard_path}")
    
    return dashboard_path

if __name__ == '__main__':
    main()
