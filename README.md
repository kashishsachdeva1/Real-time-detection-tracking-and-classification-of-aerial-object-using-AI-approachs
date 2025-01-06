# Real-time-detection-tracking-and-classification-of-aerial-object-using-AI-approachs

This project leverages a combination of YOLO object detection, Random Forest classification, and tracking algorithms to detect, classify, and analyze aerial objects such as drones, eagles, and birds in real-time. 
---

### Tech Stack
- **Programming Language:** Python  
- **Computer Vision Libraries:** OpenCV, NumPy  
- **Machine Learning Frameworks:** YOLO (Ultralytics), scikit-learn (Random Forest)  
- **Utilities:** joblib for model persistence  
- **Hardware:** Laptop camera or video files as input  

---

### Input and Output
- **Input:**  
  - Video stream from the laptop camera or pre-recorded video (`D4.mp4`).  
- **Output:**  
  - Real-time display of tracked objects with bounding boxes, object IDs, classification labels (drone, eagle, or bird), and tracking lines.  

---

### **Working**  
1. **Detection:**  
   - YOLO detects objects in each frame and extracts bounding boxes, centroids, and confidence scores.  

2. **Tracking:**  
   - Tracked objects are associated with detections using the Euclidean distance.  
   - Each object is assigned a unique ID for persistent tracking across frames.  

3. **Feature Extraction:**  
   - Features like velocity, acceleration, and frequency are calculated for each tracked object.  

4. **Classification:**  
   - A Random Forest model classifies objects (drone, eagle, or bird) based on extracted features.  
   - The classification result is displayed with the object’s ID.  

5. **Visualization:**  
   - Real-time visualization includes bounding boxes, tracking lines, and classification labels for all tracked objects.  

---

### **Challenges Faced**  
1. **Object Occlusion:**  
   - Maintaining consistent IDs for objects that move out of view or overlap with others.  

2. **Feature Normalization:**  
   - Ensuring accurate scaling of features for consistent classification performance.  

3. **Frame-to-Frame Association:**  
   - Optimizing the matching algorithm for efficiency without compromising accuracy.  

4. **Real-Time Performance:**  
   - Balancing the computational cost of YOLO detection and Random Forest classification to ensure smooth frame rates.  

---

### **Future Enhancements**  
- Implement a deep learning-based tracker for improved tracking under occlusion.  
- Expand the classification model to include additional aerial object classes.  
- Optimize the pipeline for deployment on edge devices or drones.  

This project demonstrates an end-to-end pipeline for real-time detection, tracking, and classification, ideal for surveillance or wildlife monitoring applications.
