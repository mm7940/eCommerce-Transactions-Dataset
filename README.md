
# E-Commerce Transactions Dataset Analysis

This project involves analyzing an e-commerce transactions dataset by performing Exploratory Data Analysis (EDA), building a Lookalike Model, and applying Customer Segmentation using clustering techniques. Below are the tasks, methodologies, and outputs for the project.

---

## Task 1: Exploratory Data Analysis (EDA) and Business Insights

### Objective:

Perform EDA on the provided dataset and derive at least 5 business insights.

### Methodology:

- **EDA Techniques Used:**
  - Data visualization using histograms, bar charts, and heatmaps.
  - Statistical summaries and distribution analysis of transaction amounts and customer behavior.

### Output:

- Visualizations: [Screenshot of Transaction Amount Distribution](Screenshot%20(158).png).
- Visualizations: [Screenshot of Top Customers by Transaction Amount](Screenshot%20(159).png).
- Visualizations: [Screenshot of Top Products by Sales](Screenshot%20(160).png).

---

## Task 2: Lookalike Model

### Objective:

Build a model that takes a user’s profile and transaction history as input to recommend 3 similar customers, with similarity scores.

### Methodology:

- **Features Used:**
  - Customer demographic details and transaction behavior.
  - Product purchase patterns.
- **Modeling Approach:**
  - Similarity computation using techniques like cosine similarity or distance-based metrics.
  - Profile enrichment with product interactions.

### Output:

- Recommended similar customers with assigned similarity scores.
- Visualization: [Screenshot](Screenshot%20(161).png).

---

## Task 3: Customer Segmentation / Clustering

### Objective:

Segment customers into clusters using profile and transaction data to identify behavioral patterns.

### Methodology:

- **Data Preparation:**
  - Combined `Customers.csv` and `Transactions.csv` for enriched segmentation.
- **Clustering Algorithm:**
  - K-Means clustering with optimal clusters determined between 2 and 10.
  - Evaluation performed using DB Index.
- **Visualization:**
  - Cluster scatter plots for better interpretability.

### Metrics:

- **DB Index:** Used as the evaluation metric to validate cluster quality.

### Output:

- Clustering results in both CSV format and visual plots.
- Visualization: [Screenshot](Screenshot%20(162).png).

---

## Project Deliverables

1. **Code and Notebooks:**
   - All scripts used for data preprocessing, modeling, and visualization.
2. **Outputs:**
   - Screenshots and CSV files for task-specific outputs.
3. **Documentation:**
   - This README provides a concise overview of the project and its tasks.

---


## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For any questions or feedback, please contact [hassamahmad9934@gmail.com].
