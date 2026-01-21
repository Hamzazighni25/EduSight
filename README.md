# EduSight – Data-Driven Educational Analytics Platform

📌 **Overview**

EduSight is an end-to-end **Data Engineering, Business Intelligence, and Machine Learning** project designed to transform heterogeneous educational data into actionable insights.  
The platform centralizes educational data, structures it into a data warehouse, and delivers interactive dashboards and predictive models to support data-driven decision-making in the education sector.

EduSight targets educational decision-makers such as **teachers, school directors, and educational authorities** by providing unified access to performance indicators, enrollment trends, and predictive analytics.

---

## 🚀 Key Features

### 🔹 Centralized Educational Data Platform
- Consolidation of heterogeneous educational data from official open-data sources.
- Enriched datasets including student enrollment, success rates, honors rates, and socioeconomic indicators (IPS).

### 🔹 Interactive Business Intelligence Dashboards
- **Power BI dashboards** for:
  - School structure and regional distribution
  - Academic performance and success rates
  - High schools, academic tracks, and socioeconomic impact
- Dynamic filters (region, school status, academic track, service type).
- Geographic visualizations using **Azure Maps**.

### 🔹 Machine Learning Integration
- Classification models for institutional profiling.
- Regression models for academic performance prediction.
- Clustering models to identify homogeneous school profiles.
- Predictive insights integrated into BI workflows.

### 🔹 End-to-End Deployment
- Machine learning models integrated with Power BI.
- Unified analytics workflow from data ingestion to visualization and prediction.

---

## 🛠️ Data Engineering Process

### ETL Pipeline
- Implemented using **SSIS (SQL Server Integration Services)**.
- Key steps:
  - Data cleaning (missing values, duplicates, format normalization)
  - Data transformation (standardization, encoding, reshaping)
  - Data loading into a structured data warehouse

### Data Warehouse Design
- **Star schema architecture**:
  - Central fact table containing educational indicators
  - Shared dimensions (institution, geography, service, section, track, time, institution type)
- Optimized for analytical querying and BI performance.

---

## 📊 Dashboards & Analytics

### School Structure & Distribution Dashboard
- Number of schools by region and postal code
- Public vs private school distribution
- Student population mapping
- Long-term trends in school infrastructure

### Performance & Success Analysis
- Global success rate
- Honors rate by region
- High-performing schools identification
- Socioeconomic index vs success rate analysis

### Academic Tracks & High Schools Dashboard
- Distribution of high schools by status and academic track
- Enrollment patterns by section and status
- Service type distribution

---

## 🤖 Machine Learning Models

### Classification

#### Institution Status Prediction
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Task:** Public vs Private classification

#### Institution Type/Level Prediction
- **Algorithm:** Support Vector Classifier (SVC)
- **Task:** Multiclass classification of educational levels

#### Enrollment Demand Prediction
- **Algorithm:** Logistic Regression
- **Task:** High vs Low enrollment classification

### Regression

#### Mention Rate Prediction (Taux_Mentions)
- **Algorithm:** XGBoost Regressor

#### Success Rate Prediction (Taux_Reussite)
- Continuous performance estimation

### Clustering

#### School Profiling
- **Algorithm:** K-Means
- Identification of homogeneous school clusters based on performance and context

---

## 🚀 Deployment
- Trained models integrated into **Power BI** for predictive analytics.
- End-to-end analytical workflow:

- 
- Web interface showcasing dashboards and predictive modules.

---

## 🔮 Future Plans
- Expand datasets to include additional educational regions, multi-year indicators, and enriched demographic variables.  
- Optimize machine learning models using advanced algorithms and feature engineering strategies.  
- Refine the user interface and dashboard interactivity for a seamless experience.  
- Real-world deployment and collaboration with educational institutions and authorities to enable continuous monitoring and actionable insights.

---

## 👥 Team
**Developed by Team METAFLOW:**  
- Hamza Zighni  
- Jihed Bakalti  
- Zeineb Moujehed  
- Meryem Bennani  
- Melek Amimi  
- Med Malek Manai  

**Guided by Professors:**  
- Mr. Ridha Berrahal  
- Mme. Jihene Jebri  

**Institution:** ESPRIT – School of Engineering and Technology  
**Academic Year:** 2025–2026

---

## 📚 References
- French Open Education Data (fr-en-annuaire-education)  
- Web-scraped indicators from official education platforms  
- CRISP-DM methodology  
- Academic references on Data Engineering, Business Intelligence, and Machine Learning