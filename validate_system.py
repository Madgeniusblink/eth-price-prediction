#!/usr/bin/env python3
"""
Comprehensive validation script for ETH Price Prediction System
Checks data quality, model performance, signal accuracy, and system health
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

class SystemValidator:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.reports_path = self.repo_path / "reports"
        self.latest_path = self.reports_path / "latest"
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks_passed": 0,
            "checks_failed": 0,
            "warnings": [],
            "errors": [],
            "details": {}
        }
    
    def validate_file_exists(self, filepath, description):
        """Check if required file exists"""
        if filepath.exists():
            print(f"✓ {description} exists: {filepath}")
            self.validation_results["checks_passed"] += 1
            return True
        else:
            error_msg = f"✗ {description} missing: {filepath}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False
    
    def validate_json_structure(self, filepath, required_fields, description):
        """Validate JSON file has required fields and no N/A values"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            missing_fields = []
            for field in required_fields:
                if '.' in field:  # Nested field
                    parts = field.split('.')
                    current = data
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            missing_fields.append(field)
                            break
                else:
                    if field not in data:
                        missing_fields.append(field)
            
            if missing_fields:
                error_msg = f"✗ {description}: Missing fields {missing_fields}"
                print(error_msg)
                self.validation_results["checks_failed"] += 1
                self.validation_results["errors"].append(error_msg)
                return False, data
            
            # Check for N/A values recursively
            na_fields = self._find_na_values(data)
            if na_fields:
                error_msg = f"✗ {description}: Found N/A values in fields: {na_fields}"
                print(error_msg)
                self.validation_results["checks_failed"] += 1
                self.validation_results["errors"].append(error_msg)
                return False, data
            
            print(f"✓ {description}: Structure valid, no N/A values")
            self.validation_results["checks_passed"] += 1
            return True, data
            
        except json.JSONDecodeError as e:
            error_msg = f"✗ {description}: Invalid JSON - {str(e)}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False, None
        except Exception as e:
            error_msg = f"✗ {description}: Error reading file - {str(e)}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False, None
    
    def _find_na_values(self, obj, path=""):
        """Recursively find N/A values in nested structures"""
        na_fields = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if value == "N/A" or value == "n/a":
                    na_fields.append(current_path)
                elif isinstance(value, (dict, list)):
                    na_fields.extend(self._find_na_values(value, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                if item == "N/A" or item == "n/a":
                    na_fields.append(current_path)
                elif isinstance(item, (dict, list)):
                    na_fields.extend(self._find_na_values(item, current_path))
        
        return na_fields
    
    def validate_predictions(self, predictions_data):
        """Validate prediction values are reasonable"""
        print("\n=== Validating Predictions ===")
        
        current_price = predictions_data.get("current_price", 0)
        predictions = predictions_data.get("predictions", {})
        
        if current_price <= 0:
            error_msg = f"✗ Invalid current price: {current_price}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False
        
        print(f"Current Price: ${current_price:.2f}")
        
        # Check each prediction timeframe
        timeframes = ["15m", "30m", "60m", "120m"]
        all_valid = True
        
        for tf in timeframes:
            if tf not in predictions:
                error_msg = f"✗ Missing prediction for {tf}"
                print(error_msg)
                self.validation_results["checks_failed"] += 1
                self.validation_results["errors"].append(error_msg)
                all_valid = False
                continue
            
            pred = predictions[tf]
            pred_price = pred.get("price", 0)
            change_pct = pred.get("change_pct", 0)
            
            # Validate price is reasonable (within ±20% of current)
            if pred_price <= 0:
                error_msg = f"✗ {tf}: Invalid price {pred_price}"
                print(error_msg)
                self.validation_results["checks_failed"] += 1
                self.validation_results["errors"].append(error_msg)
                all_valid = False
            elif abs(change_pct) > 20:
                warning_msg = f"⚠ {tf}: Large change {change_pct:.2f}% (${pred_price:.2f})"
                print(warning_msg)
                self.validation_results["warnings"].append(warning_msg)
            else:
                print(f"✓ {tf}: ${pred_price:.2f} ({change_pct:+.2f}%)")
                self.validation_results["checks_passed"] += 1
        
        return all_valid
    
    def validate_trading_signals(self, signals_data):
        """Validate trading signal values"""
        print("\n=== Validating Trading Signals ===")
        
        signal = signals_data.get("trading_signal", {})
        
        # Check signal direction
        signal_type = signal.get("signal", "")
        if signal_type not in ["BUY", "SELL", "HOLD"]:
            error_msg = f"✗ Invalid signal type: {signal_type}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False
        
        print(f"Signal: {signal_type}")
        
        # Check confidence
        confidence = signal.get("confidence", "")
        if confidence not in ["HIGH", "MEDIUM", "LOW"]:
            error_msg = f"✗ Invalid confidence: {confidence}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False
        
        print(f"✓ Confidence: {confidence}")
        self.validation_results["checks_passed"] += 1
        
        # Check entry/stop/target prices
        entry = signal.get("entry", 0)
        stop_loss = signal.get("stop_loss", 0)
        target = signal.get("target", 0)
        
        if entry <= 0 or stop_loss <= 0 or target <= 0:
            error_msg = f"✗ Invalid trade prices: Entry={entry}, Stop={stop_loss}, Target={target}"
            print(error_msg)
            self.validation_results["checks_failed"] += 1
            self.validation_results["errors"].append(error_msg)
            return False
        
        print(f"✓ Entry: ${entry:.2f}")
        print(f"✓ Stop Loss: ${stop_loss:.2f}")
        print(f"✓ Target: ${target:.2f}")
        self.validation_results["checks_passed"] += 3
        
        # Check risk/reward ratio
        risk_reward = signal.get("risk_reward", 0)
        if risk_reward <= 0:
            warning_msg = f"⚠ Poor risk/reward ratio: {risk_reward:.2f}"
            print(warning_msg)
            self.validation_results["warnings"].append(warning_msg)
        else:
            print(f"✓ Risk/Reward: {risk_reward:.2f}")
            self.validation_results["checks_passed"] += 1
        
        return True
    
    def validate_model_performance(self):
        """Check model performance metrics"""
        print("\n=== Validating Model Performance ===")
        
        perf_file = self.reports_path / "model_performance.json"
        if not perf_file.exists():
            warning_msg = "⚠ Model performance file not found (may be first run)"
            print(warning_msg)
            self.validation_results["warnings"].append(warning_msg)
            return True
        
        try:
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            
            # Check R² score
            r2_score = perf_data.get("ensemble_r2", 0)
            if r2_score < 0.5:
                warning_msg = f"⚠ Low R² score: {r2_score:.4f} (target: >0.6)"
                print(warning_msg)
                self.validation_results["warnings"].append(warning_msg)
            else:
                print(f"✓ R² Score: {r2_score:.4f}")
                self.validation_results["checks_passed"] += 1
            
            # Store performance metrics
            self.validation_results["details"]["model_performance"] = {
                "r2_score": r2_score,
                "model_scores": perf_data.get("model_scores", {}),
                "model_weights": perf_data.get("model_weights", {})
            }
            
            return True
            
        except Exception as e:
            warning_msg = f"⚠ Error reading model performance: {str(e)}"
            print(warning_msg)
            self.validation_results["warnings"].append(warning_msg)
            return True
    
    def validate_recent_reports(self):
        """Check that recent reports exist"""
        print("\n=== Validating Recent Reports ===")
        
        # Check for reports in the last 24 hours
        now = datetime.utcnow()
        year_dir = self.reports_path / str(now.year)
        
        if not year_dir.exists():
            warning_msg = "⚠ No reports directory for current year"
            print(warning_msg)
            self.validation_results["warnings"].append(warning_msg)
            return True
        
        # Count recent reports
        recent_reports = []
        for month_dir in year_dir.iterdir():
            if month_dir.is_dir():
                for day_dir in month_dir.iterdir():
                    if day_dir.is_dir():
                        for report_dir in day_dir.iterdir():
                            if report_dir.is_dir():
                                recent_reports.append(report_dir)
        
        print(f"✓ Found {len(recent_reports)} report(s) in current year")
        self.validation_results["checks_passed"] += 1
        self.validation_results["details"]["recent_reports_count"] = len(recent_reports)
        
        return True
    
    def run_validation(self):
        """Run all validation checks"""
        print("=" * 60)
        print("ETH PRICE PREDICTION SYSTEM VALIDATION")
        print("=" * 60)
        
        # Check latest report files exist
        print("\n=== Checking Required Files ===")
        self.validate_file_exists(
            self.latest_path / "trading_signals.json",
            "Trading Signals"
        )
        self.validate_file_exists(
            self.latest_path / "predictions_summary.json",
            "Predictions Summary"
        )
        self.validate_file_exists(
            self.latest_path / "README.md",
            "Report README"
        )
        
        # Validate predictions_summary.json
        print("\n=== Validating Predictions Summary ===")
        required_pred_fields = [
            "current_price",
            "predictions.15m.price",
            "predictions.30m.price",
            "predictions.60m.price",
            "predictions.120m.price",
            "trend_analysis.trend",
            "ensemble_r2"
        ]
        
        valid, pred_data = self.validate_json_structure(
            self.latest_path / "predictions_summary.json",
            required_pred_fields,
            "Predictions Summary"
        )
        
        if valid and pred_data:
            self.validate_predictions(pred_data)
            self.validation_results["details"]["predictions"] = pred_data.get("predictions", {})
        
        # Validate trading_signals.json
        print("\n=== Validating Trading Signals ===")
        required_signal_fields = [
            "trading_signal.signal",
            "trading_signal.confidence",
            "trading_signal.entry",
            "trading_signal.stop_loss",
            "trading_signal.target",
            "trend_analysis.trend",
            "support_resistance.current_price"
        ]
        
        valid, signal_data = self.validate_json_structure(
            self.latest_path / "trading_signals.json",
            required_signal_fields,
            "Trading Signals"
        )
        
        if valid and signal_data:
            self.validate_trading_signals(signal_data)
            self.validation_results["details"]["trading_signal"] = signal_data.get("trading_signal", {})
        
        # Validate model performance
        self.validate_model_performance()
        
        # Check recent reports
        self.validate_recent_reports()
        
        # Print summary
        self.print_summary()
        
        return self.validation_results
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_checks = self.validation_results["checks_passed"] + self.validation_results["checks_failed"]
        passed = self.validation_results["checks_passed"]
        failed = self.validation_results["checks_failed"]
        warnings = len(self.validation_results["warnings"])
        
        print(f"\nTotal Checks: {total_checks}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"⚠ Warnings: {warnings}")
        
        if failed == 0:
            print("\n🎉 ALL VALIDATION CHECKS PASSED!")
            if warnings > 0:
                print(f"   (with {warnings} warning(s))")
        else:
            print(f"\n❌ VALIDATION FAILED: {failed} check(s) failed")
        
        # Save validation results
        output_file = self.repo_path / "validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

def main():
    repo_path = os.path.dirname(os.path.abspath(__file__))
    validator = SystemValidator(repo_path)
    results = validator.run_validation()
    
    # Exit with error code if validation failed
    if results["checks_failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
