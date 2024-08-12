# Fleet Decarbonization Optimization

## Overview

This project contains the solution for the Shell.ai Hackathon for Sustainable and Affordable Energy 2024 challenge on fleet decarbonization optimization. The goal is to develop a strategy for transitioning a vehicle fleet to lower-emission alternatives while balancing operational costs, meeting demand, and adhering to emission limits over a 16-year period (2023-2038).

## Problem Description

The challenge involves optimizing a fleet composition over 16 years (2023-2038) to meet supply-chain demand while adhering to carbon emission limits and minimizing overall cost. Key aspects include:

- Multiple vehicle types (Diesel, LNG, BEV) with different characteristics
- Varying fuel types, costs, and associated emissions
- Yearly demand satisfaction for different vehicle sizes and distance requirements
- Decreasing carbon emission limits over time
- Vehicle lifecycle constraints (10-year lifespan, purchase/sell timing)

## Approach

I used linear programming to model and solve this optimization problem. The key components of our approach are:

1. **Decision Variables**:
   - $buy_{y,v}$: Number of vehicles of type $v$ bought in year $y$
   - $use_{y,v}$: Number of vehicles of type $v$ used in year $y$
   - $sell_{y,v}$: Number of vehicles of type $v$ sold in year $y$

2. **Objective Function**:
   Minimize total cost over the planning horizon:
   $$ \text{Minimize } C_{total} = \sum_{y=2023}^{2038} (C_{buy}^y + C_{ins}^y + C_{mnt}^y + C_{fuel}^y - C_{sell}^y) $$

3. **Key Constraints**:
   - Demand Satisfaction
   - Carbon Emissions Limits
   - Vehicle Lifecycle

## Implementation

The optimization model is implemented in Python using the PuLP library. The main script `opt_model.py` in the `scripts/` directory contains the model formulation and solver configuration.

## Data

The `data/` directory should contain all the necessary input files for the optimization model:

- `carbon_emissions.csv`: Yearly carbon emission limits
- `cost_profiles.csv`: Vehicle cost profiles over time
- `demand.csv`: Yearly demand for different vehicle sizes and distances
- `fuels.csv`: Fuel types, costs, and emission factors
- `vehicles.csv`: Vehicle specifications
- `vehicles_fuels.csv`: Vehicle fuel consumption data

## Results

The optimization results are stored in the `data/results/` directory. 

## Running the Model

To run the optimization model:

1. Ensure you have Python 3.x installed and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the optimization script:
   ```
   python scripts/opt_model.py
   ```

## Future Work

Potential areas for improvement and exploration include:
- Incorporating uncertainty in fuel prices and demand
- Exploring multi-objective optimization to balance cost and emissions explicitly
- Investigating heuristic approaches for larger problem instances

## Acknowledgments

This project was developed as part of the Shell.ai Hackathon for Sustainable and Affordable Energy 2024. We thank Shell for providing this challenging and relevant problem in the field of sustainability and optimization.

