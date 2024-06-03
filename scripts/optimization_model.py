# scripts/optimization_model.py

import os
import pandas as pd
import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpInteger

# Define the data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load datasets
carbon_emissions = pd.read_csv(os.path.join(data_dir, 'carbon_emissions.csv'))
cost_profiles = pd.read_csv(os.path.join(data_dir, 'cost_profiles.csv'))
demand = pd.read_csv(os.path.join(data_dir, 'demand.csv'))
fuels = pd.read_csv(os.path.join(data_dir, 'fuels.csv'))
vehicles = pd.read_csv(os.path.join(data_dir, 'vehicles.csv'))
vehicles_fuels = pd.read_csv(os.path.join(data_dir, 'vehicles_fuels.csv'))

# Check if data is loaded correctly
print(carbon_emissions.head())
print(cost_profiles.head())
print(demand.head())
print(fuels.head())
print(vehicles.head())
print(vehicles_fuels.head())

# Print column names to verify
print("Vehicles columns:", vehicles.columns)

# Initialize the optimization problem
model = LpProblem(name="fleet-optimization", sense=LpMinimize)

# Define decision variables
years = carbon_emissions['Year']
vehicle_ids = vehicles['ID']

x_buy = LpVariable.dicts("Buy", (vehicle_ids, years), lowBound=0, cat=LpInteger)
x_use = LpVariable.dicts("Use", (vehicle_ids, years), lowBound=0, cat=LpInteger)
x_sell = LpVariable.dicts("Sell", (vehicle_ids, years), lowBound=0, cat=LpInteger)

# Map years to indices for cost profiles
year_to_index = {year: idx for idx, year in enumerate(years)}

# Correct column names based on the print statement
vehicle_size_col = 'Size'  # Update based on actual column name

# Objective Function Components
purchase_cost = lpSum([vehicles.loc[vehicles['ID'] == i, 'Cost ($)'].values[0] * x_buy[i][yr]
                       for i in vehicle_ids for yr in years])

insurance_cost = lpSum([vehicles.loc[vehicles['ID'] == i, 'Cost ($)'].values[0] * 
                        cost_profiles.loc[min(year_to_index[yr], len(cost_profiles) - 1), 'Insurance Cost %'] / 100 * x_use[i][yr]
                        for i in vehicle_ids for yr in years])

maintenance_cost = lpSum([vehicles.loc[vehicles['ID'] == i, 'Cost ($)'].values[0] * 
                          cost_profiles.loc[min(year_to_index[yr], len(cost_profiles) - 1), 'Maintenance Cost %'] / 100 * x_use[i][yr]
                          for i in vehicle_ids for yr in years])

fuel_cost = lpSum([vehicles_fuels.loc[vehicles_fuels['ID'] == i, 'Consumption (unit_fuel/km)'].values[0] * 
                   fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == i, 'Fuel'].values[0]) & (fuels['Year'] == yr), 'Cost ($/unit_fuel)'].values[0] * 
                   demand.loc[(demand['Year'] == yr) & (demand['Size'] == vehicles.loc[vehicles['ID'] == i, vehicle_size_col].values[0]), 'Demand (km)'].values[0] * 
                   x_use[i][yr] for i in vehicle_ids for yr in years])

resale_value = lpSum([vehicles.loc[vehicles['ID'] == i, 'Cost ($)'].values[0] * 
                      cost_profiles.loc[min(year_to_index[yr], len(cost_profiles) - 1), 'Resale Value %'] / 100 * x_sell[i][yr]
                      for i in vehicle_ids for yr in years])

# Total Cost
total_cost = purchase_cost + insurance_cost + maintenance_cost + fuel_cost - resale_value

model += total_cost

# Constraints

# Demand Fulfillment
for yr in years:
    for size in demand['Size'].unique():
        for dist in demand['Distance'].unique():
            model += lpSum([x_use[i][yr] * vehicles.loc[vehicles['ID'] == i, 'Yearly range (km)'].values[0]
                            for i in vehicle_ids if vehicles.loc[vehicles['ID'] == i, vehicle_size_col].values[0] == size]) >= \
                     demand.loc[(demand['Year'] == yr) & (demand['Size'] == size) & (demand['Distance'] == dist), 'Demand (km)'].values[0]

# Emission Limits
for yr in years:
    model += lpSum([vehicles_fuels.loc[vehicles_fuels['ID'] == i, 'Consumption (unit_fuel/km)'].values[0] *
                    fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == i, 'Fuel'].values[0]) & (fuels['Year'] == yr), 'Emissions (CO2/unit_fuel)'].values[0] *
                    demand.loc[(demand['Year'] == yr) & (demand['Size'] == vehicles.loc[vehicles['ID'] == i, vehicle_size_col].values[0]), 'Demand (km)'].values[0] *
                    x_use[i][yr] for i in vehicle_ids]) <= carbon_emissions.loc[carbon_emissions['Year'] == yr, 'Carbon emission CO2/kg'].values[0]

# Solve the model
model.solve()

# Print the results
for v in model.variables():
    print(v.name, "=", v.varValue)

# Save the results to a CSV file
results = []
for yr in years:
    for i in vehicle_ids:
        results.append([yr, i, x_buy[i][yr].varValue, x_use[i][yr].varValue, x_sell[i][yr].varValue])

results_df = pd.DataFrame(results, columns=['Year', 'ID', 'Num_Buy', 'Num_Use', 'Num_Sell'])
results_df.to_csv(os.path.join(data_dir, 'optimization_results.csv'), index=False)
