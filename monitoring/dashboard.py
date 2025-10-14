"""
MLOps Monitoring Dashboard for Product Recognition
Analyzes prediction logs and displays metrics
"""

import json
import os
from datetime import datetime, timedelta
from collections import Counter

def load_logs(log_file="../logs/predictions.jsonl"):
    """Load prediction logs"""
    logs = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    except FileNotFoundError:
        print("❌ No log file found. Make some predictions first!")
        print(f"   Looking for: {log_file}")
        return []
    return logs

def analyze_predictions():
    """Analyze prediction logs and display dashboard"""
    logs = load_logs()
    if not logs:
        return
    
    # Convert timestamps
    for log in logs:
        log['timestamp'] = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
    
    print()
    print("=" * 70)
    print("📊 PRODUCT RECOGNITION - MLOPS DASHBOARD")
    print("=" * 70)
    print()
    
    # Overall metrics
    print("📈 OVERALL METRICS")
    print("-" * 70)
    total_predictions = len(logs)
    total_detections = sum(log['num_predictions'] for log in logs)
    
    all_confidences = []
    for log in logs:
        all_confidences.extend(log.get('confidence_scores', []))
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    avg_processing = sum(log.get('processing_time_ms', 0) for log in logs) / total_predictions
    
    print(f"  Total Predictions:        {total_predictions:,}")
    print(f"  Total Detections:         {total_detections:,}")
    print(f"  Avg Detections/Image:     {total_detections/total_predictions:.2f}")
    print(f"  Avg Confidence:           {avg_confidence:.1%}")
    print(f"  Avg Processing Time:      {avg_processing:.2f}ms")
    print()
    
    # Time range
    print("📅 TIME RANGE")
    print("-" * 70)
    timestamps = [log['timestamp'] for log in logs]
    first_pred = min(timestamps)
    last_pred = max(timestamps)
    duration = last_pred - first_pred
    
    print(f"  First Prediction:         {first_pred.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Last Prediction:          {last_pred.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:                 {duration}")
    print(f"  Predictions/Hour:         {total_predictions / max(duration.total_seconds() / 3600, 1):.1f}")
    print()
    
    # Model info
    if logs:
        print("🤖 MODEL INFORMATION")
        print("-" * 70)
        print(f"  Model Version:            {logs[0].get('model_version', 'Unknown')}")
        print()
    
    # Class distribution
    print("🏷️  TOP DETECTED CLASSES")
    print("-" * 70)
    all_classes = []
    for log in logs:
        all_classes.extend(log.get('classes_detected', []))
    
    class_counts = Counter(all_classes)
    for i, (cls, count) in enumerate(class_counts.most_common(15), 1):
        percentage = (count / len(all_classes)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {i:2}. {cls:20} {count:4} ({percentage:5.1f}%) {bar}")
    print()
    
    # Confidence distribution
    print("🎯 CONFIDENCE DISTRIBUTION")
    print("-" * 70)
    if all_confidences:
        # Sort confidences
        sorted_conf = sorted(all_confidences)
        n = len(sorted_conf)
        
        print(f"  Minimum:                  {min(all_confidences):.1%}")
        print(f"  25th Percentile:          {sorted_conf[n//4]:.1%}")
        print(f"  Median:                   {sorted_conf[n//2]:.1%}")
        print(f"  75th Percentile:          {sorted_conf[3*n//4]:.1%}")
        print(f"  Maximum:                  {max(all_confidences):.1%}")
        
        # Histogram
        print()
        print("  Distribution:")
        ranges = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for low, high in ranges:
            count = sum(1 for c in all_confidences if low <= c < high)
            percentage = (count / len(all_confidences)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {low:.1f}-{high:.1f}  {count:4} ({percentage:5.1f}%) {bar}")
    print()
    
    # Performance metrics
    print("⏱️  PERFORMANCE METRICS")
    print("-" * 70)
    processing_times = [log.get('processing_time_ms', 0) for log in logs]
    if processing_times:
        sorted_times = sorted(processing_times)
        n = len(sorted_times)
        
        print(f"  Fastest:                  {min(processing_times):.2f}ms")
        print(f"  25th Percentile:          {sorted_times[n//4]:.2f}ms")
        print(f"  Median:                   {sorted_times[n//2]:.2f}ms")
        print(f"  75th Percentile:          {sorted_times[3*n//4]:.2f}ms")
        print(f"  Slowest:                  {max(processing_times):.2f}ms")
    print()
    
    # Recent activity (last 24 hours)
    print("📊 RECENT ACTIVITY (Last 24 hours)")
    print("-" * 70)
    now = datetime.utcnow()
    recent_logs = [log for log in logs if (now - log['timestamp']).total_seconds() < 86400]
    
    if recent_logs:
        recent_detections = sum(log['num_predictions'] for log in recent_logs)
        recent_confidences = []
        for log in recent_logs:
            recent_confidences.extend(log.get('confidence_scores', []))
        recent_avg_conf = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0
        
        print(f"  Predictions:              {len(recent_logs):,}")
        print(f"  Detections:               {recent_detections:,}")
        print(f"  Avg Confidence:           {recent_avg_conf:.1%}")
    else:
        print("  No predictions in last 24 hours")
    print()
    
    print("=" * 70)
    print()
    print("💡 TIP: Run 'python dashboard.py' regularly to monitor your model's performance")
    print()

if __name__ == "__main__":
    analyze_predictions()
