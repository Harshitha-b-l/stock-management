The Stock Management System is an efficient real-time inventory tracking application designed for small businesses, institutions, or personal use. Leveraging computer vision with the YOLOv8 model, this app detects and categorizes stock items via webcam images, automatically logs them with detailed timestamps, and provides user-friendly visual summaries for inventory monitoring.

This project demonstrates practical integration of AI-powered object detection, data logging, and interactive dashboards to streamline stock management workflows.

Key Features
Real-time Object Detection: Uses YOLOv8 to identify objects captured through a webcam.

Automated Categorization: Maps detected items to predefined categories (e.g., Electronics, Stationery, Beverages).

Inventory Logging: Automatically records each detection to a CSV file (stock_log.csv) with product name, category, timestamp, and quantity.

Interactive Dashboard: Utilizes Streamlit for a clean interface to capture images, display detection results, and present inventory summaries.

Visual Analytics: Displays recent detection history and category-wise counts with charts for quick stock assessment.

Customizable Category Mapping: Easily update or expand the product-to-category mappings in the code.

Technology Stack
Programming Language: Python

Frameworks and Libraries:

Streamlit (UI and dashboard)

Ultralytics YOLO (YOLOv8 object detection)

OpenCV (image processing)

Pandas (data handling)

Data Storage: CSV file (stock_log.csv) for recording stock logs
