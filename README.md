# PG&E Energy Analytics Challenge 2025
This repository contains my solution to the 2025 Electricity Load Forecasting Challenge, focused on predicting hourly electricity demand in the California Independent System Operator (CAISO) energy market.

## Competition Overview
The challenge, organized by PG&E, engages students and researchers in developing accurate electricity load forecasts for a specific (undisclosed) region within the CAISO system. These forecasts play a key role in maintaining a reliable and stable power grid across California.


## Objective
The goal is to build a model that predicts hourly electricity load for an entire year using historical data. The catch: models must be causal — i.e., no peeking into the future. For any prediction at a certain date and hour, only past and present data can be used.
## Code Structure
- **PK_Model.py**
- **SA_Model.py**
- **NU_Model.py**
- **BY_Model.py**
- **NG_Model.py**
  - Each of these files trains three different models and generates six Excel files (insample and outsample predictions).
  - Training and testing splits:
    1. Train on Year 1, test on Year 2
    2. Train on Year 2, test on Year 1
    3. Train on Year 1 + Year 2, test on Year 3

- **Hybrid_Code_test_3.ipynb**
  - Runs all the individual models.
  - Trains four types of hybrid models using the six generated Excel files.
  - The final submission is based on **Hybrid Model 1**.

## How to Run
1. Open the **Hybrid_Code_test_3.ipynb**.
2. Run all cells.
3. The final results will be generated automatically.


