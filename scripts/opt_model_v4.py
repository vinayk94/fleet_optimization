# Import required libraries
import os
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpInteger, PULP_CBC_CMD, LpStatus

# Define the data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load datasets
carbon_emissions = pd.read_csv(os.path.join(data_dir, 'carbon_emissions.csv'))
cost_profiles = pd.read_csv(os.path.join(data_dir, 'cost_profiles.csv'))
demand = pd.read_csv(os.path.join(data_dir, 'demand.csv'))
fuels = pd.read_csv(os.path.join(data_dir, 'fuels.csv'))
vehicles = pd.read_csv(os.path.join(data_dir, 'vehicles.csv'))
vehicles_fuels = pd.read_csv(os.path.join(data_dir, 'vehicles_fuels.csv'))

# Preprocess data for cost profiles and fuels
max_lifecycle = cost_profiles['End of Year'].max()
full_years = range(1, max_lifecycle + 1)
full_cost_profiles = pd.DataFrame({'End of Year': full_years})
cost_profiles = full_cost_profiles.merge(cost_profiles, on='End of Year', how='left').fillna(0)

fuel_types = fuels['Fuel'].unique()
full_years = range(2023, 2039)
full_fuel_profiles = pd.MultiIndex.from_product([full_years, fuel_types], names=['Year', 'Fuel']).to_frame(index=False)
fuels = full_fuel_profiles.merge(fuels, on=['Year', 'Fuel'], how='left').fillna(0)

# Initialize the optimization problem
model = LpProblem(name="fleet-optimization", sense=LpMinimize)

# Define decision variables for buying, using, and selling vehicles
buy_vars = LpVariable.dicts("buy", 
                            [(year, v_id) for year in range(2023, 2039) for v_id in vehicles['ID']], 
                            lowBound=0, cat=LpInteger)

use_vars = LpVariable.dicts("use", 
                            [(year, v_id) for year in range(2023, 2039) for v_id in vehicles['ID']], 
                            lowBound=0, cat=LpInteger)

sell_vars = LpVariable.dicts("sell", 
                             [(year, v_id) for year in range(2023, 2039) for v_id in vehicles['ID']], 
                             lowBound=0, cat=LpInteger)

# Fleet size calculation
fleet_size = {}
for year in range(2023, 2039):
    for v_id in vehicles['ID']:
        purchase_year = int(v_id.split('_')[-1])
        if purchase_year <= year < purchase_year + 10:
            fleet_size[(year, v_id)] = (
                lpSum([buy_vars[(y, v_id)] for y in range(purchase_year, year + 1)]) - 
                lpSum([sell_vars[(y, v_id)] for y in range(purchase_year, year + 1) if (y, v_id) in sell_vars])
            )
        else:
            fleet_size[(year, v_id)] = 0

# Ensure vehicle usage within 10-year lifespan
for v_id in vehicles['ID']:
    purchase_year = int(v_id.split('_')[-1])
    for year in range(2023, 2039):
        if year < purchase_year or year >= purchase_year + 10:
            model += use_vars[(year, v_id)] == 0
            model += sell_vars[(year, v_id)] == 0

# Ensure vehicles are only bought in their designated purchase year
for v_id in vehicles['ID']:
    purchase_year = int(v_id.split('_')[-1])
    for year in range(2023, 2039):
        if year != purchase_year:
            model += buy_vars[(year, v_id)] == 0

# Ensure vehicles are sold by the end of their 10-year lifespan
for v_id in vehicles['ID']:
    purchase_year = int(v_id.split('_')[-1])
    model += lpSum([sell_vars[(year, v_id)] for year in range(purchase_year, purchase_year + 10) if (year, v_id) in sell_vars]) == lpSum([buy_vars[(year, v_id)] for year in range(purchase_year, purchase_year + 10) if (year, v_id) in buy_vars])

# Define the objective function components
def get_cost_profile_value(year, purchase_year, cost_type):
    year_offset = year - purchase_year
    value = cost_profiles.loc[cost_profiles['End of Year'] == year_offset, cost_type]
    if not value.empty:
        return value.values[0] / 100
    return 0

buying_costs = lpSum([
    buy_vars[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0]
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

insurance_costs = lpSum([
    fleet_size[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(v_id.split('_')[-1]), 'Insurance Cost %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

maintenance_costs = lpSum([
    fleet_size[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(v_id.split('_')[-1]), 'Maintenance Cost %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

fuel_costs = lpSum([
    use_vars[(year, v_id)] * vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Consumption (unit_fuel/km)'].values[0] *
    fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]) & (fuels['Year'] == year), 'Cost ($/unit_fuel)'].values[0] *
    vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
    if (year, v_id) in use_vars
])

resale_value = lpSum([
    sell_vars[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(v_id.split('_')[-1]), 'Resale Value %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
    if (year, v_id) in sell_vars
])

# Set the objective function
model += buying_costs + insurance_costs + maintenance_costs + fuel_costs - resale_value

# Demand Satisfaction Constraint
for _, row in demand.iterrows():
    model += lpSum([
        use_vars[(row['Year'], v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
        for v_id in vehicles['ID']
        if (row['Year'], v_id) in use_vars and
           vehicles.loc[vehicles['ID'] == v_id, 'Size'].values[0] == row['Size'] and 
           vehicles.loc[vehicles['ID'] == v_id, 'Distance'].values[0] >= row['Distance']
    ]) >= row['Demand (km)']

# Carbon Emissions Constraint
for year in carbon_emissions['Year']:
    model += lpSum([
        use_vars[(year, v_id)] * vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Consumption (unit_fuel/km)'].values[0] * 
        fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]) & (fuels['Year'] == year), 'Emissions (CO2/unit_fuel)'].values[0] * 
        vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
        for v_id in vehicles['ID']
        if (year, v_id) in use_vars
    ]) <= carbon_emissions.loc[carbon_emissions['Year'] == year, 'Carbon emission CO2/kg'].values[0]

# Vehicle Sale Constraint
for year in range(2023, 2039):
    for v_id in vehicles['ID']:
        if (year, v_id) in sell_vars:
            model += sell_vars[(year, v_id)] <= 0.2 * lpSum([
                buy_vars[(y, v_id)] - sell_vars[(y, v_id)] for y in range(2023, year + 1)
                if (y, v_id) in sell_vars and (y, v_id) in buy_vars
            ])

# Ensure the results directory exists
results_dir = os.path.join(data_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Solve the problem
solver = PULP_CBC_CMD(msg=True, timeLimit=1800, gapRel=0.01)
model.solve(solver)


def find_infeasible_constraints(model):
    # Create a new model
    relaxed_model = LpProblem("Relaxed_" + model.name, model.sense)

    # Add all variables from the original model
    for v in model.variables():
        relaxed_model.addVariable(v)

    # Dictionary to store slack variables
    slack_vars = {}

    # Add constraints with slack variables
    for name, constraint in model.constraints.items():
        slack_var = LpVariable(f"Slack_{name}", lowBound=0)
        slack_vars[name] = slack_var
        relaxed_constraint = constraint.copy()
        relaxed_constraint += slack_var
        relaxed_model += relaxed_constraint

    # Set objective to minimize sum of slack variables
    relaxed_model += lpSum(slack_vars.values())

    # Solve the relaxed model
    relaxed_model.solve()

    # Print constraints with non-zero slack
    for name, slack_var in slack_vars.items():
        if slack_var.varValue > 1e-6:  # Using a small threshold to account for numerical precision
            print(f"Constraint {name} is violated by {slack_var.varValue}")



# Add this right after solving your original model
print(f"Model status: {model.status}")
print(f"Model status string: {LpStatus[model.status]}")

if model.status == -1:
    print("Model is infeasible. Constraints cannot be satisfied with current parameters.")
    
    # Find infeasible constraints
    print("\nFinding infeasible constraints:")
    find_infeasible_constraints(model)
    
    """
    # Print variable bounds
    print("\nVariable bounds:")
    for v in model.variables():
        print(f"{v.name}: {v.lowBound} <= {v.varValue} <= {v.upBound}")
    
    # Print objective function
    print("\nObjective function:")
    print(model.objective)
    """

# Create lists to store variable info and results
variable_info = []
results = []

# Append Buy, Use, and Sell operations to the results
for year in range(2023, 2039):
    for v_id in vehicles['ID']:
        # Extract purchase year and vehicle attributes
        purchase_year = int(v_id.split('_')[-1])
        size = vehicles.loc[vehicles['ID'] == v_id, 'Size'].values[0]
        distance_bucket = vehicles.loc[vehicles['ID'] == v_id, 'Distance'].values[0]

        # Buy operations
        if (year, v_id) in buy_vars:
            num_bought = int(buy_vars[(year, v_id)].varValue)
            variable_info.append([year, v_id, 'Buy', num_bought])
            if num_bought > 0:
                results.append([year, v_id, num_bought, 'Buy', '', '', ''])

        # Use operations
        if (year, v_id) in use_vars:
            num_used = int(use_vars[(year, v_id)].varValue)
            variable_info.append([year, v_id, 'Use', num_used])
            if num_used > 0:
                fuel_type = vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]
                yearly_range = vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
                results.append([year, v_id, num_used, 'Use', fuel_type, distance_bucket, yearly_range])

        # Sell operations
        if (year, v_id) in sell_vars:
            num_sold = int(sell_vars[(year, v_id)].varValue)
            variable_info.append([year, v_id, 'Sell', num_sold])
            if num_sold > 0:
                results.append([year, v_id, num_sold, 'Sell', '', '', ''])

# Create DataFrames for variable info and results
variable_info_df = pd.DataFrame(variable_info, columns=[
    'Year', 'ID', 'Type', 'Num_Vehicles'
])

results_df = pd.DataFrame(results, columns=[
    'Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'
])

def validate_and_correct_submission(submission_df, vehicles, vehicles_fuels, fuels):
    errors = []
    print("Starting validation and correction...")

    # Check and correct data types
    print("Validating data types...")
    if not all(submission_df['Year'].apply(lambda x: isinstance(x, int))):
        errors.append(1)
        submission_df['Year'] = submission_df['Year'].astype(int)

    if not all(submission_df['ID'].apply(lambda x: isinstance(x, str))):
        errors.append(2)
        submission_df['ID'] = submission_df['ID'].astype(str)

    if not all(submission_df['Num_Vehicles'].apply(lambda x: isinstance(x, int))):
        errors.append(3)
        submission_df['Num_Vehicles'] = submission_df['Num_Vehicles'].astype(int)

    if not all(submission_df['Type'].apply(lambda x: isinstance(x, str))):
        errors.append(4)
        submission_df['Type'] = submission_df['Type'].astype(str)

    if not all(submission_df['Fuel'].apply(lambda x: isinstance(x, str))):
        errors.append(5)
        submission_df['Fuel'] = submission_df['Fuel'].astype(str)

    if not all(submission_df['Distance_bucket'].apply(lambda x: isinstance(x, str))):
        errors.append(6)
        submission_df['Distance_bucket'] = submission_df['Distance_bucket'].astype(str)

    # Clean and convert Distance_per_vehicle(km) to float
    submission_df['Distance_per_vehicle(km)'] = pd.to_numeric(submission_df['Distance_per_vehicle(km)'], errors='coerce').fillna(0.0)
    if not all(submission_df['Distance_per_vehicle(km)'].apply(lambda x: isinstance(x, (int, float)))):
        errors.append(7)
        submission_df['Distance_per_vehicle(km)'] = submission_df['Distance_per_vehicle(km)'].astype(float)

    print("Validating ranges and constraints...")
    # Check and correct value ranges and constraints
    if not all((2023 <= submission_df['Year']) & (submission_df['Year'] <= 2038)):
        errors.append(8)

    if not all(submission_df['ID'].isin(vehicles['ID'])):
        errors.append(9)

    if not all(submission_df['Num_Vehicles'] > 0):
        errors.append(10)

    valid_types = {"Buy", "Sell", "Use"}
    if not all(submission_df['Type'].isin(valid_types)):
        errors.append(11)

    valid_fuels = {"Electricity", "LNG", "BioLNG", "HVO", "B20"}
    if not all(submission_df['Fuel'].isin(valid_fuels)):
        errors.append(12)

    valid_distance_buckets = {"D1", "D2", "D3", "D4"}
    if not all(submission_df['Distance_bucket'].isin(valid_distance_buckets)):
        errors.append(13)

    # Check distance driven range
    for _, row in submission_df.iterrows():
        v_id = row['ID']
        yearly_range = vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
        if not (0 <= row['Distance_per_vehicle(km)'] <= yearly_range):
            errors.append(14)
            break

    # Check if each vehicle is used within its 10-year lifespan and bought in the correct year
    for _, row in submission_df.iterrows():
        year = row['Year']
        v_id = row['ID']
        purchase_year = int(v_id.split('_')[-1])
        if row['Type'] == 'Buy' and year != purchase_year:
            errors.append(16)
            break
        if row['Type'] in ['Use', 'Sell']:
            if year < purchase_year or year >= purchase_year + 10:
                errors.append(17)
                break

    print("Checking demand satisfaction...")
    # Check if the sum of distance traveled by each vehicle type meets the demand in demand.csv
    demand_met = {}
    for _, row in submission_df.iterrows():
        if row['Type'] == 'Use':
            year = row['Year']
            size = vehicles.loc[vehicles['ID'] == row['ID'], 'Size'].values[0]
            distance_bucket = row['Distance_bucket']
            key = (year, size, distance_bucket)
            demand_met[key] = demand_met.get(key, 0) + row['Distance_per_vehicle(km)'] * row['Num_Vehicles']

    for _, row in demand.iterrows():
        key = (row['Year'], row['Size'], row['Distance'])
        if key not in demand_met or demand_met[key] < row['Demand (km)']:
            print(f"Demand not met for year {row['Year']}, size {row['Size']}, distance {row['Distance']}")
            errors.append(19)
            break

    # Check if at most 20% of the fleet is sold each year
    for year in range(2023, 2039):
        fleet_size = {}
        for _, row in submission_df[submission_df['Year'] <= year].iterrows():
            v_id = row['ID']
            if row['Type'] == 'Buy':
                fleet_size[v_id] = fleet_size.get(v_id, 0) + row['Num_Vehicles']
            elif row['Type'] == 'Sell':
                fleet_size[v_id] = fleet_size.get(v_id, 0) - row['Num_Vehicles']
        
        total_fleet = sum(fleet_size.values())
        sold_this_year = submission_df[(submission_df['Year'] == year) & (submission_df['Type'] == 'Sell')]['Num_Vehicles'].sum()
        
        if sold_this_year > 0.2 * total_fleet:
            errors.append(21)
            break

    # Check if you only "Use" and "Sell" vehicle IDs that you have in the fleet
    fleet_size = {}
    for year in range(2023, 2039):
        for _, row in submission_df[submission_df['Year'] == year].iterrows():
            v_id = row['ID']
            if row['Type'] == 'Buy':
                fleet_size[v_id] = fleet_size.get(v_id, 0) + row['Num_Vehicles']
            elif row['Type'] == 'Sell':
                if v_id not in fleet_size or fleet_size[v_id] < row['Num_Vehicles']:
                    errors.append(24)
                    break
                fleet_size[v_id] -= row['Num_Vehicles']
            elif row['Type'] == 'Use':
                if v_id not in fleet_size or fleet_size[v_id] < row['Num_Vehicles']:
                    errors.append(22)
                    print(f"Using more vehicles than available in fleet for ID {v_id} in year {year}")
                    break
        if 22 in errors or 24 in errors:
            break

    # Check if the right type of fuel is used for each vehicle
    for _, row in submission_df[submission_df['Type'] == 'Use'].iterrows():
        v_id = row['ID']
        correct_fuel = vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]
        if row['Fuel'] != correct_fuel:
            errors.append(20)
            break

    print(f"Validation errors found: {errors}")
    return submission_df, errors

# Validate and correct the submission
results_df, validation_errors = validate_and_correct_submission(results_df, vehicles, vehicles_fuels, fuels)

# Print validation errors if any
if validation_errors:
    print(f"Validation errors found: {validation_errors}")
else:
    print("No validation errors found.")

# Calculate total cost
total_cost = model.objective.value()
print(f"Total cost: ${total_cost:,.2f}")

# Save the variable info to a CSV file
variable_info_df.to_csv(os.path.join(results_dir, 'variable_info.csv'), index=False)

# Save the results to a CSV file
results_df.to_csv(os.path.join(results_dir, 'submission.csv'), index=False)

print("Optimization completed and results saved.")

