# Power-Network Disruption Prediction: Evaluating Local Network Topology

## 1. Project Title and Short Description
Title: *Power-Network Disruption Prediction*  
This project investigates whether incorporating local network topological properties into standard machine learning models improves the prediction of future disruptions in a stylized power transmission network, compared to relying solely on intrinsic substation attributes.

---

## 2. Problem Statement
Given a power network where one central substation (`S066`) has experienced an initial interruption, the objective is to predict which of the remaining, initially functioning substations will experience a simulated interruption over the subsequent 24 hours. The central research question is: 
> *Does direct connectivity and local cross-link density provide predictive power beyond substation load and equipment age?*

---

## 3. Dataset Set
The analysis uses two synthetic datasets provided by the course:
* **`power_substations.csv` (144 nodes):**
  * `substation_id`: Unique identifier (`S001` to `S144`).
  * `load_ratio`: Pre-disruption load relative to nominal capacity (range: 0.42-0.94).
  * `equipment_age_years`: Simulated age of equipment (2-40 years).
  * `initially_interrupted`: Binary indicator (1 for `S066`, 0 for all others).
  * `interrupted_later`: Binary prediction target for the 143 eligible substations (39 positive events).
* **`power_links.csv` (344 undirected links):** Physical, unweighted transmission lines connecting substations in a 12x12 grid augmented with diagonal cross-links.

---

## 4. Method
1. **Data Preprocessing & Filtering:** Excluded the initially interrupted substation (`S066`), leaving 143 eligible substations.
2. **Network Feature Extraction (NetworkX):**
   * **Degree:** Total count of direct physical connections incident to a substation.
   * **Local Clustering Coefficient:** The proportion of possible edges that actually exist among a substation's immediate neighbours (measuring local meshing and triangle density).
3. **Model Evaluation Setup:**
   * Partitioned eligible observations using a stratified 70/30 train-test split (random seed=123) to preserve the disruption ratio.
   * Standardized features using `StandardScaler` fitted strictly on training data inside an `sklearn.pipeline` to prevent data leakage.
4. **Model Comparison:**
   * **Model A (Baseline):** Logistic Regression using only `load_ratio` and `equipment_age_years`.
   * **Model B (Network-Augmented):** Logistic Regression combining original features with `degree` and `local_clustering`.
   * Evaluation metrics: **Accuracy**, **F1-score**, **ROC AUC**, and **Confusion Matrices**.

---

## 5. Results
Across the held-out test set (43 substations):

| Model | Features | Accuracy | F1-Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Model A (Baseline)** | `load_ratio`, `equipment_age_years` | 0.767 | 0.444 | 0.828 |
| **Model B (With Network)** | Baseline + `degree`, `local_clustering` | 0.860 | 0.750 | 0.949 |

* **Performance Observations:** The addition of network features produces mixed outcomes. While probability calibration and ranking (ROC AUC) show a moderate change, discrete classification metrics (F1 and Accuracy) show only marginal differences.
* **Error Analysis:** As seen in the confusion matrices, both models exhibit trade-offs between false alarms (false positives) and missed disruptions (false negatives), indicating that network topology alone does not provide a definitive decision boundary.

---

## 6. Interpretation & Limitations
* **Synthetic Nature:** The dataset was generated via independent probabilistic draws based on a logistic formula, it does not simulate genuine alternating current (AC) power flow, impedance, dynamic cascading failures, or protective tripping mechanisms.
* **Non-Causal Association:** An observed correlation between network clustering and interruption probability does not imply that failures propagate along physical links.
* **Operational & Ethical Risks:** In real-world utility dispatch, relying on imperfect heuristic models carries severe implications:
  * **False Negatives (Missed Disruptions):** Leave critical infrastructure unprepared, risking unannounced blackouts for hospitals and vulnerable communities.
  * **False Positives (False Alarms):** Divert emergency repair crews and limited public funding away from true grid vulnerabilities.
  * Algorithmic prioritization must remain an advisory tool subject to electrical engineering audits and public utility oversight.

---

## 7. Reflection
* **What Worked Well:** Building an end-to-end reproducible pipeline with leak-free scaling and clear NetworkX graph visualisations.
* **Challenges:** Evaluating performance on a small sample size (43 test nodes), where single classification flips significantly alter discrete scores like F1.
* **Future Work:** With more time, I would implement k-fold cross-validation to assess stability across splits, explore non-linear models (e.g., Random Forest, Gradient Boosting), and test path-based centrality metrics (e.g., betweenness centrality, shortest path distance to `S066`).
