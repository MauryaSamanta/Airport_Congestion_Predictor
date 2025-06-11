# Airport Departure Congestion Prediction

This project analyzes and predicts **airport departure congestion** using flight operations data, focusing on delays caused by ground surface congestion (mainly taxiways and runway queues). The goal is to build an interpretable machine learning model that predicts whether a departing flight will experience congestion.

---

## Project Steps & Rationale

### 1. **Data Collection**
- Used publicly available flight operations data (e.g., US DOT, FAA, or similar sources).
- Key fields: `TaxiOut`, `DepTime`, `ArrDelay`, `NASDelay`, `Origin`, and others relevant to departure operations.

### 2. **Why TaxiOut?**
- **TaxiOut time** measures how long an aircraft spends taxiing from gate to runway before takeoff.
- Research and industry standards (FAA, Eurocontrol) use **TaxiOut** as a primary indicator of departure surface congestion, as high TaxiOut times reflect ground bottlenecks and runway queues.

### 3. **Labeling Congestion**
- For each airport, calculated the **75th percentile** of historical TaxiOut times as a dynamic, airport-specific congestion threshold.
    - *Why 75th percentile?*  
      - Referenced in aviation research and operational dashboards as a robust threshold for identifying unusually high taxi times and surface congestion events.
      - See: FAA ATADS documentation and surface operations studies.
- Labeled a flight as **"congested"** (`1`) if its TaxiOut exceeded the 75th percentile for its airport; otherwise, labeled as **"not congested"** (`0`).

### 4. **Feature Engineering**
- Created features such as hour of day, day of week, past congestion history, and rolling 1 hour average taxi out times

### 5. **Modeling**
- Trained a model using both **Random Forest Classifier** as well as **Logistic Regression**


### 6. **Evaluation**
- Evaluated the model using accuracy, precision, recall, and confusion matrix.
- Obtained highest F1 score of 84% using **Random Forest Classifier** 

---

## Data Sources

- [US DOT Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
- [FAA ATADS](https://aspm.faa.gov/)
- Academic literature on airport surface operations and congestion metrics

---

## References

- FAA, "ATADS: Airport Departure and Arrival Surface Metrics"
- Idris et al., "Analysis of Airport Surface Operations," Transportation Research Record
- Simaiakis & Balakrishnan, "Airport Departure Queues: Estimation and Control," Transportation Research

---


