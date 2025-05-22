"""
GPS Data Analysis with NumPy
============================

This script demonstrates various NumPy operations on a dummy GPS dataset
containing 1000 GPS coordinates with additional metadata.

Author: Paul Stobbe
Date: May 22, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Set random seed for reproducible results
np.random.seed(42)

class GPSDataAnalyzer:
    """A class to handle GPS data generation and analysis using NumPy."""
    
    def __init__(self, num_points=1000):
        """Initialize with specified number of GPS points."""
        self.num_points = num_points
        self.gps_data = None
        self.generate_dummy_data()
    
    def generate_dummy_data(self):
        """Generate dummy GPS dataset with realistic coordinates and metadata."""
        print(f"🌍 Generating {self.num_points} dummy GPS points...")
        
        # Generate GPS coordinates (centered around Munich, Germany)
        center_lat = 48.1351  # Munich latitude
        center_lon = 11.5820  # Munich longitude
        
        # Create realistic GPS scatter around Munich (±0.5 degrees)
        latitudes = np.random.normal(center_lat, 0.2, self.num_points)
        longitudes = np.random.normal(center_lon, 0.3, self.num_points)
        
        # Generate timestamps (last 30 days)
        base_time = datetime.now() - timedelta(days=30)
        timestamps = np.array([
            base_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            ) for _ in range(self.num_points)
        ])
        
        # Generate additional metadata
        altitudes = np.random.normal(500, 100, self.num_points)  # Altitude in meters
        speeds = np.random.exponential(25, self.num_points)      # Speed in km/h
        accuracies = np.random.uniform(1, 50, self.num_points)   # GPS accuracy in meters
        
        # Create structured array
        self.gps_data = np.array(list(zip(
            latitudes, longitudes, altitudes, speeds, accuracies, timestamps
        )), dtype=[
            ('latitude', 'f8'),
            ('longitude', 'f8'),
            ('altitude', 'f8'),
            ('speed', 'f8'),
            ('accuracy', 'f8'),
            ('timestamp', 'O')
        ])
        
        print(f"✅ Generated GPS dataset with shape: {self.gps_data.shape}")
        print(f"📊 Data types: {self.gps_data.dtype}")
    
    def basic_statistics(self):
        """Calculate and display basic statistics using NumPy."""
        print("\n📈 BASIC STATISTICS")
        print("=" * 50)
        
        # Latitude statistics
        lat_mean = np.mean(self.gps_data['latitude'])
        lat_std = np.std(self.gps_data['latitude'])
        lat_min = np.min(self.gps_data['latitude'])
        lat_max = np.max(self.gps_data['latitude'])
        
        print(f"🌐 Latitude:")
        print(f"   Mean: {lat_mean:.6f}°")
        print(f"   Std:  {lat_std:.6f}°")
        print(f"   Range: {lat_min:.6f}° to {lat_max:.6f}°")
        
        # Longitude statistics
        lon_mean = np.mean(self.gps_data['longitude'])
        lon_std = np.std(self.gps_data['longitude'])
        lon_min = np.min(self.gps_data['longitude'])
        lon_max = np.max(self.gps_data['longitude'])
        
        print(f"🌐 Longitude:")
        print(f"   Mean: {lon_mean:.6f}°")
        print(f"   Std:  {lon_std:.6f}°")
        print(f"   Range: {lon_min:.6f}° to {lon_max:.6f}°")
        
        # Speed statistics
        speed_mean = np.mean(self.gps_data['speed'])
        speed_median = np.median(self.gps_data['speed'])
        speed_max = np.max(self.gps_data['speed'])
        
        print(f"🚗 Speed:")
        print(f"   Mean: {speed_mean:.2f} km/h")
        print(f"   Median: {speed_median:.2f} km/h")
        print(f"   Max: {speed_max:.2f} km/h")
        
        # Altitude statistics
        alt_mean = np.mean(self.gps_data['altitude'])
        alt_std = np.std(self.gps_data['altitude'])
        
        print(f"⛰️  Altitude:")
        print(f"   Mean: {alt_mean:.1f} m")
        print(f"   Std: {alt_std:.1f} m")
    
    def advanced_numpy_operations(self):
        """Perform advanced NumPy operations on the GPS data."""
        print("\n🔬 ADVANCED NUMPY OPERATIONS")
        print("=" * 50)
        
        # 1. Calculate distances from center point using vectorized operations
        center_lat, center_lon = 48.1351, 11.5820
        
        # Haversine formula using NumPy
        lat1, lon1 = np.radians(self.gps_data['latitude']), np.radians(self.gps_data['longitude'])
        lat2, lon2 = np.radians(center_lat), np.radians(center_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        distances = 6371 * 2 * np.arcsin(np.sqrt(a))  # Earth radius = 6371 km
        
        print(f"📏 Distance calculations:")
        print(f"   Mean distance from Munich center: {np.mean(distances):.2f} km")
        print(f"   Max distance from Munich center: {np.max(distances):.2f} km")
        print(f"   Points within 10km: {np.sum(distances < 10)} ({np.sum(distances < 10)/len(distances)*100:.1f}%)")
        
        # 2. Find clusters using coordinate binning
        lat_bins = np.linspace(np.min(self.gps_data['latitude']), 
                              np.max(self.gps_data['latitude']), 10)
        lon_bins = np.linspace(np.min(self.gps_data['longitude']), 
                              np.max(self.gps_data['longitude']), 10)
        
        hist, _, _ = np.histogram2d(self.gps_data['latitude'], 
                                  self.gps_data['longitude'], 
                                  bins=[lat_bins, lon_bins])
        
        max_cluster_idx = np.unravel_index(np.argmax(hist), hist.shape)
        max_cluster_count = hist[max_cluster_idx]
        
        print(f"🎯 Clustering analysis:")
        print(f"   Densest grid cell contains: {max_cluster_count} points")
        print(f"   Grid cell location: [{lat_bins[max_cluster_idx[0]]:.4f}, {lon_bins[max_cluster_idx[1]]:.4f}]")
        
        # 3. Speed analysis and filtering
        high_speed_mask = self.gps_data['speed'] > 50  # Above 50 km/h
        high_speed_count = np.sum(high_speed_mask)
        
        print(f"🏎️  Speed analysis:")
        print(f"   High-speed points (>50 km/h): {high_speed_count} ({high_speed_count/len(self.gps_data)*100:.1f}%)")
        
        if high_speed_count > 0:
            high_speed_locations = self.gps_data[high_speed_mask]
            print(f"   Average speed of high-speed points: {np.mean(high_speed_locations['speed']):.2f} km/h")
        
        # 4. Accuracy filtering
        accurate_mask = self.gps_data['accuracy'] < 10  # Better than 10m accuracy
        accurate_count = np.sum(accurate_mask)
        
        print(f"🎯 Accuracy analysis:")
        print(f"   High-accuracy points (<10m): {accurate_count} ({accurate_count/len(self.gps_data)*100:.1f}%)")
        print(f"   Mean accuracy: {np.mean(self.gps_data['accuracy']):.2f} m")
        
        return distances, high_speed_mask, accurate_mask
    
    def time_series_analysis(self):
        """Analyze temporal patterns in the GPS data."""
        print("\n⏰ TIME SERIES ANALYSIS")
        print("=" * 50)
        
        # Convert timestamps to hours for analysis
        hours = np.array([ts.hour for ts in self.gps_data['timestamp']])
        
        # Activity by hour
        hour_counts, hour_bins = np.histogram(hours, bins=24, range=(0, 24))
        peak_hour = np.argmax(hour_counts)
        
        print(f"📅 Temporal patterns:")
        print(f"   Most active hour: {peak_hour}:00 ({hour_counts[peak_hour]} points)")
        print(f"   Least active hour: {np.argmin(hour_counts)}:00 ({np.min(hour_counts)} points)")
        
        # Weekend vs weekday analysis
        weekdays = np.array([ts.weekday() for ts in self.gps_data['timestamp']])
        weekend_mask = weekdays >= 5  # Saturday=5, Sunday=6
        
        weekend_count = np.sum(weekend_mask)
        weekday_count = len(self.gps_data) - weekend_count
        
        print(f"   Weekend points: {weekend_count} ({weekend_count/len(self.gps_data)*100:.1f}%)")
        print(f"   Weekday points: {weekday_count} ({weekday_count/len(self.gps_data)*100:.1f}%)")
        
        # Speed patterns by time
        weekend_speeds = self.gps_data['speed'][weekend_mask]
        weekday_speeds = self.gps_data['speed'][~weekend_mask]
        
        if len(weekend_speeds) > 0 and len(weekday_speeds) > 0:
            print(f"   Average weekend speed: {np.mean(weekend_speeds):.2f} km/h")
            print(f"   Average weekday speed: {np.mean(weekday_speeds):.2f} km/h")
    
    def geometric_transformations(self):
        """Apply geometric transformations using NumPy."""
        print("\n🔄 GEOMETRIC TRANSFORMATIONS")
        print("=" * 50)
        
        # Create coordinate arrays
        coords = np.column_stack((self.gps_data['latitude'], self.gps_data['longitude']))
        
        # 1. Translation (shift coordinates)
        translation_vector = np.array([0.01, 0.01])  # Shift by ~1km
        translated_coords = coords + translation_vector
        
        print(f"📍 Translation:")
        print(f"   Shifted all points by {translation_vector}")
        print(f"   Original center: [{np.mean(coords[:, 0]):.6f}, {np.mean(coords[:, 1]):.6f}]")
        print(f"   Translated center: [{np.mean(translated_coords[:, 0]):.6f}, {np.mean(translated_coords[:, 1]):.6f}]")
        
        # 2. Scaling (zoom in/out)
        center = np.mean(coords, axis=0)
        scale_factor = 0.5  # Scale down by 50%
        scaled_coords = center + (coords - center) * scale_factor
        
        print(f"🔍 Scaling:")
        print(f"   Scale factor: {scale_factor}")
        print(f"   Original spread (std): [{np.std(coords[:, 0]):.6f}, {np.std(coords[:, 1]):.6f}]")
        print(f"   Scaled spread (std): [{np.std(scaled_coords[:, 0]):.6f}, {np.std(scaled_coords[:, 1]):.6f}]")
        
        # 3. Rotation (rotate around center)
        angle = np.pi / 6  # 30 degrees in radians
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        centered_coords = coords - center
        rotated_coords = np.dot(centered_coords, rotation_matrix.T) + center
        
        print(f"🔄 Rotation:")
        print(f"   Rotation angle: {np.degrees(angle):.1f} degrees")
        print(f"   Rotation matrix applied to {len(coords)} points")
        
        return translated_coords, scaled_coords, rotated_coords
    
    def statistical_analysis(self):
        """Perform statistical analysis using NumPy."""
        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Correlation analysis
        speed_alt_corr = np.corrcoef(self.gps_data['speed'], self.gps_data['altitude'])[0, 1]
        speed_acc_corr = np.corrcoef(self.gps_data['speed'], self.gps_data['accuracy'])[0, 1]
        
        print(f"🔗 Correlations:")
        print(f"   Speed vs Altitude: {speed_alt_corr:.3f}")
        print(f"   Speed vs Accuracy: {speed_acc_corr:.3f}")
        
        # Percentile analysis
        speed_percentiles = np.percentile(self.gps_data['speed'], [10, 25, 50, 75, 90])
        altitude_percentiles = np.percentile(self.gps_data['altitude'], [10, 25, 50, 75, 90])
        
        print(f"📈 Speed percentiles (km/h): 10%={speed_percentiles[0]:.1f}, 25%={speed_percentiles[1]:.1f}, 50%={speed_percentiles[2]:.1f}, 75%={speed_percentiles[3]:.1f}, 90%={speed_percentiles[4]:.1f}")
        print(f"⛰️  Altitude percentiles (m): 10%={altitude_percentiles[0]:.1f}, 25%={altitude_percentiles[1]:.1f}, 50%={altitude_percentiles[2]:.1f}, 75%={altitude_percentiles[3]:.1f}, 90%={altitude_percentiles[4]:.1f}")
        
        # Outlier detection using IQR method
        Q1_speed = np.percentile(self.gps_data['speed'], 25)
        Q3_speed = np.percentile(self.gps_data['speed'], 75)
        IQR_speed = Q3_speed - Q1_speed
        
        outlier_mask = (self.gps_data['speed'] < Q1_speed - 1.5 * IQR_speed) | \
                      (self.gps_data['speed'] > Q3_speed + 1.5 * IQR_speed)
        outlier_count = np.sum(outlier_mask)
        
        print(f"🚨 Outlier detection (IQR method):")
        print(f"   Speed outliers: {outlier_count} ({outlier_count/len(self.gps_data)*100:.1f}%)")
        
        if outlier_count > 0:
            outlier_speeds = self.gps_data['speed'][outlier_mask]
            print(f"   Outlier speed range: {np.min(outlier_speeds):.1f} - {np.max(outlier_speeds):.1f} km/h")
    
    def save_results(self):
        """Save processed data to files."""
        print("\n💾 SAVING RESULTS")
        print("=" * 50)
        
        # Save original data as CSV-like format
        np.savetxt('gps_data_original.csv',
                   np.column_stack((
                       self.gps_data['latitude'],
                       self.gps_data['longitude'],
                       self.gps_data['altitude'],
                       self.gps_data['speed'],
                       self.gps_data['accuracy']
                   )),
                   delimiter=',',
                   header='latitude,longitude,altitude,speed,accuracy',
                   comments='',
                   fmt='%.6f,%.6f,%.2f,%.2f,%.2f')
        
        # Save summary statistics
        with open('gps_analysis_summary.txt', 'w') as f:
            f.write("GPS Data Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total GPS points: {len(self.gps_data)}\n")
            f.write(f"Latitude range: {np.min(self.gps_data['latitude']):.6f} to {np.max(self.gps_data['latitude']):.6f}\n")
            f.write(f"Longitude range: {np.min(self.gps_data['longitude']):.6f} to {np.max(self.gps_data['longitude']):.6f}\n")
            f.write(f"Speed statistics: Mean={np.mean(self.gps_data['speed']):.2f} km/h, Max={np.max(self.gps_data['speed']):.2f} km/h\n")
            f.write(f"Altitude statistics: Mean={np.mean(self.gps_data['altitude']):.1f} m, Std={np.std(self.gps_data['altitude']):.1f} m\n")
        
        print("✅ Files saved:")
        print("   📄 gps_data_original.csv - Raw GPS data")
        print("   📄 gps_analysis_summary.txt - Analysis summary")

def main():
    """Main function to run the GPS data analysis."""
    print("🚀 GPS DATA ANALYSIS WITH NUMPY")
    print("=" * 60)
    print(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = GPSDataAnalyzer(num_points=1000)
    
    # Run all analyses
    analyzer.basic_statistics()
    distances, high_speed_mask, accurate_mask = analyzer.advanced_numpy_operations()
    analyzer.time_series_analysis()
    translated_coords, scaled_coords, rotated_coords = analyzer.geometric_transformations()
    analyzer.statistical_analysis()
    analyzer.save_results()
    
    print(f"\n🎉 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📁 Check your current directory for the generated files!")

if __name__ == "__main__":
    main()
