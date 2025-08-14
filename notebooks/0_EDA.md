# Exploratory Data Analysis - DQN Traffic Signal Control

This notebook provides exploratory data analysis for the traffic simulation data and environment setup.

## Contents

1. **Environment Configuration Analysis**
   - Display traffic environment configuration
   - DQN agent configuration
   - Traffic generation configuration

2. **State Space Analysis**
   - Analyze state representation
   - Feature descriptions
   - State normalization

3. **Action Space Analysis**
   - Available actions
   - Action descriptions

4. **Traffic Pattern Analysis**
   - Daily traffic patterns
   - Peak vs off-peak analysis

5. **Vehicle Type Distribution**
   - Vehicle type probabilities
   - Speed characteristics

6. **Reward Function Analysis**
   - Reward components
   - Weight analysis

7. **Simulation Environment Test**
   - Environment testing
   - Basic metrics collection

8. **Metrics Analysis**
   - Performance metrics
   - Time series analysis

9. **Summary and Insights**
   - Key findings
   - Next steps

## Key Findings

- State Space: 20-dimensional normalized state representation
- Action Space: 4 discrete actions for traffic signal control
- Traffic Patterns: Peak hours show 2-6x higher volume than off-peak
- Vehicle Types: 70% passenger, 20% truck, 10% bus
- Reward Function: Multi-objective optimization with weighted components

## Next Steps

1. Proceed to model training (notebook 1)
2. Analyze training results (notebook 2)
3. Evaluate model performance and compare with baselines
