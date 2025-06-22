#!/usr/bin/env python3
"""
Script to query and display multi-model performance results from PostgreSQL.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime

# PostgreSQL configuration
PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

def get_model_performance_summary():
    """
    Get a summary of model performance across all weeks.
    """
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to get average performance by model
        query = """
        SELECT 
            model_name,
            COUNT(*) as weeks_evaluated,
            AVG(accuracy) as avg_accuracy,
            AVG(macro_f1) as avg_macro_f1,
            AVG(weighted_f1) as avg_weighted_f1,
            MIN(macro_f1) as min_macro_f1,
            MAX(macro_f1) as max_macro_f1,
            STDDEV(macro_f1) as std_macro_f1,
            SUM(total_samples) as total_samples
        FROM model_performance_metrics 
        GROUP BY model_name
        ORDER BY avg_macro_f1 DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("❌ No performance data found in database")
            return None
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(results)
        
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for _, row in df.iterrows():
            print(f"🏆 {row['model_name'].upper()}")
            print(f"   Weeks Evaluated: {row['weeks_evaluated']}")
            print(f"   Average Macro F1: {row['avg_macro_f1']:.4f}")
            print(f"   Average Accuracy: {row['avg_accuracy']:.4f}")
            print(f"   Macro F1 Range: {row['min_macro_f1']:.4f} - {row['max_macro_f1']:.4f}")
            print(f"   Std Dev: {row['std_macro_f1']:.4f}")
            print(f"   Total Samples: {row['total_samples']:,}")
            print()
        
        return df
        
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_weekly_comparison(limit_weeks=10):
    """
    Get a comparison of model performance for the most recent weeks.
    """
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to get recent weekly performance
        query = """
        SELECT 
            week_date,
            model_name,
            accuracy,
            macro_f1,
            weighted_f1,
            total_samples
        FROM model_performance_metrics 
        WHERE week_date IN (
            SELECT DISTINCT week_date 
            FROM model_performance_metrics 
            ORDER BY week_date DESC 
            LIMIT %s
        )
        ORDER BY week_date DESC, macro_f1 DESC
        """
        
        cursor.execute(query, (limit_weeks,))
        results = cursor.fetchall()
        
        if not results:
            print("❌ No weekly performance data found")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        print(f"📈 WEEKLY MODEL COMPARISON (Last {limit_weeks} weeks)")
        print("=" * 80)
        print()
        
        # Group by week and show model rankings
        for week_date in df['week_date'].unique():
            week_data = df[df['week_date'] == week_date]
            print(f"📅 Week: {week_date}")
            
            for i, (_, row) in enumerate(week_data.iterrows(), 1):
                print(f"   {i}. {row['model_name']}: Macro F1 = {row['macro_f1']:.4f} (Acc: {row['accuracy']:.4f})")
            print()
        
        return df
        
    except Exception as e:
        print(f"❌ Error querying weekly data: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_best_model_by_week():
    """
    Show which model performed best each week.
    """
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to find the best model each week
        query = """
        WITH ranked_models AS (
            SELECT 
                week_date,
                model_name,
                macro_f1,
                accuracy,
                ROW_NUMBER() OVER (PARTITION BY week_date ORDER BY macro_f1 DESC) as rank
            FROM model_performance_metrics
        )
        SELECT 
            week_date,
            model_name,
            macro_f1,
            accuracy
        FROM ranked_models 
        WHERE rank = 1
        ORDER BY week_date DESC
        LIMIT 20
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("❌ No data found for best model analysis")
            return None
        
        print("🏆 BEST MODEL BY WEEK")
        print("=" * 80)
        print()
        
        # Count wins by model
        model_wins = {}
        
        for row in results:
            model_name = row['model_name']
            if model_name not in model_wins:
                model_wins[model_name] = 0
            model_wins[model_name] += 1
            
            print(f"📅 {row['week_date']}: {model_name} (F1: {row['macro_f1']:.4f}, Acc: {row['accuracy']:.4f})")
        
        print()
        print("📊 WIN COUNT BY MODEL:")
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {wins} wins")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing best models: {e}")
        return None
    finally:
        if conn:
            conn.close()

def main():
    """
    Main function to display all performance analyses.
    """
    print("🔍 MULTI-MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # 1. Overall performance summary
    summary_df = get_model_performance_summary()
    
    if summary_df is not None:
        print("\n" + "="*80)
        
        # 2. Weekly comparison
        weekly_df = get_weekly_comparison()
        
        if weekly_df is not None:
            print("\n" + "="*80)
            
            # 3. Best model by week
            best_models = get_best_model_by_week()
    
    print("\n" + "="*80)
    print("💡 TIPS:")
    print("   - Use Grafana to create visualizations of these metrics")
    print("   - Set up alerts for significant performance degradation")
    print("   - Consider model retraining when performance drops consistently")
    print("   - The best overall model can be used for production deployment")

if __name__ == "__main__":
    main() 