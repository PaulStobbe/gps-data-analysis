# 🌍 GPS Data Analysis with NumPy

A comprehensive Python script that demonstrates various NumPy operations on GPS coordinate data. This project generates a dummy dataset of 1000 GPS points and performs extensive analysis using NumPy's powerful array operations.

## 🎯 **Features**

### **Data Generation**
- Creates 1000 realistic GPS coordinates centered around Munich, Germany
- Includes latitude, longitude, altitude, speed, accuracy, and timestamp data
- Uses NumPy's random functions for realistic data distribution

### **Analysis Capabilities**

#### 📊 **Basic Statistics**
- Mean, standard deviation, min/max calculations
- Comprehensive statistics for all GPS parameters
- Median and range analysis

#### 🔬 **Advanced NumPy Operations**
- **Vectorized distance calculations** using the Haversine formula
- **Clustering analysis** with 2D histogram binning
- **Boolean masking** for data filtering (speed, accuracy)
- **Array indexing** and conditional operations

#### ⏰ **Time Series Analysis**
- Activity patterns by hour of day
- Weekend vs weekday behavior analysis
- Temporal speed pattern detection

#### 🔄 **Geometric Transformations**
- **Translation**: Coordinate shifting operations
- **Scaling**: Zoom in/out transformations
- **Rotation**: Matrix-based coordinate rotation

#### 📈 **Statistical Analysis**
- Correlation analysis between variables
- Percentile calculations (10th, 25th, 50th, 75th, 90th)
- Outlier detection using IQR (Interquartile Range) method

#### 💾 **Data Export**
- CSV export of processed GPS data
- Summary statistics report generation

## 🚀 **Quick Start**

### **Requirements**
```bash
pip install numpy matplotlib
```

### **Usage**
```bash
python gps_data_analysis.py
```

## 📁 **Output Files**

The script generates two output files:
- `gps_data_original.csv` - Raw GPS dataset with coordinates and metadata
- `gps_analysis_summary.txt` - Statistical summary of the analysis

## 🔍 **Example Output**

```
🚀 GPS DATA ANALYSIS WITH NUMPY
============================================================
📅 Analysis started at: 2025-05-22 17:30:15

🌍 Generating 1000 dummy GPS points...
✅ Generated GPS dataset with shape: (1000,)
📊 Data types: [('latitude', '<f8'), ('longitude', '<f8'), ('altitude', '<f8'), ('speed', '<f8'), ('accuracy', '<f8'), ('timestamp', 'O')]

📈 BASIC STATISTICS
==================================================
🌐 Latitude:
   Mean: 48.135234°
   Std:  0.198765°
   Range: 47.456789° to 48.823456°

🌐 Longitude:
   Mean: 11.582134°
   Std:  0.298432°
   Range: 10.234567° to 12.456789°
```

## 🧮 **NumPy Operations Demonstrated**

### **Array Creation & Manipulation**
- `np.random.normal()` - Normal distribution generation
- `np.random.exponential()` - Exponential distribution
- `np.column_stack()` - Combining arrays
- Structured arrays with custom data types

### **Mathematical Operations**
- `np.mean()`, `np.std()`, `np.min()`, `np.max()` - Basic statistics
- `np.radians()`, `np.sin()`, `np.cos()` - Trigonometric functions
- `np.sqrt()`, `np.arcsin()` - Mathematical functions
- `np.dot()` - Matrix multiplication

### **Advanced Features**
- `np.histogram2d()` - 2D histogram generation
- `np.corrcoef()` - Correlation coefficient calculation
- `np.percentile()` - Percentile calculations
- Boolean indexing and masking
- Vectorized operations for performance

## 🌍 **GPS Data Structure**

The generated dataset includes:
- **Latitude**: GPS latitude coordinates (degrees)
- **Longitude**: GPS longitude coordinates (degrees)
- **Altitude**: Elevation above sea level (meters)
- **Speed**: Movement speed (km/h)
- **Accuracy**: GPS accuracy (meters)
- **Timestamp**: Date and time of GPS reading

## 📚 **Learning Objectives**

This project demonstrates:
- Practical NumPy usage for geospatial data
- Vectorized operations for performance
- Statistical analysis techniques
- Data transformation and manipulation
- File I/O operations with NumPy
- Real-world data processing workflows

## 🤝 **Contributing**

Feel free to fork this repository and submit pull requests for improvements or additional analysis features!

## 📄 **License**

This project is open source and available under the MIT License.

---

**Author**: Paul Stobbe  
**Date**: May 22, 2025  
**Technology**: Python, NumPy, Matplotlib
