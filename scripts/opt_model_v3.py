import os
import pandas as pd
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


### PREPROCESSING FUEL AND VEHICLE COSTS 

# Create a full range of years and vehicle lifecycle stages

# Determine the maximum lifecycle based on the available data
max_lifecycle = cost_profiles['End of Year'].max()
full_years = range(1, max_lifecycle+1)  
full_cost_profiles = pd.DataFrame({'End of Year': full_years})
# Merge with existing cost_profiles and fill missing values with defaults
cost_profiles = full_cost_profiles.merge(cost_profiles, on='End of Year', how='left').fillna(0)


# Get all unique fuel types
fuel_types = fuels['Fuel'].unique()
# Create a full range of years and fuel types
full_years = range(2023, 2039)
full_fuel_profiles = pd.MultiIndex.from_product([full_years, fuel_types], names=['Year', 'Fuel']).to_frame(index=False)
# Merge with existing fuels and fill missing values with defaults
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
        purchase_year = int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0])
        fleet_size[(year, v_id)] = lpSum([
            buy_vars[(y, v_id)] for y in range(purchase_year, year + 1)
        ]) - lpSum([
            sell_vars[(y, v_id)] for y in range(purchase_year, year + 1)
        ])

# Define the objective function

# Sum of the costs of all vehicles purchased each year
buying_costs = lpSum([
    buy_vars[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0]
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

# Ensure we are correctly calculating the year offsets for insurance and maintenance costs
def get_cost_profile_value(year, purchase_year, cost_type):
    year_offset = year - purchase_year
    value = cost_profiles.loc[cost_profiles['End of Year'] == year_offset, cost_type]
    if not value.empty:
        return value.values[0] / 100
    return 0

# Sum of the insurance costs for all vehicles in the fleet each year
# calculated as a percentage of the vehicle's purchase cost based on the number of years since the vehicle was purchased
insurance_costs = lpSum([
    fleet_size[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0]), 'Insurance Cost %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

# Sum of the maintenance costs for all vehicles in the fleet each year
# Sum of the maintenance costs for all vehicles used each year, calculated similarly to insurance costs.
maintenance_costs = lpSum([
    fleet_size[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0]), 'Maintenance Cost %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])
# Sum of the fuel costs for all vehicles used each year, 
# calculated based on the vehicle's fuel consumption rate, the cost of fuel per unit, and the vehicle's yearly range.

fuel_costs = lpSum([
    use_vars[(year, v_id)] * vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Consumption (unit_fuel/km)'].values[0] *
    fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]) & (fuels['Year'] == year), 'Cost ($/unit_fuel)'].values[0] *
    vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

# Sum of the resale values of all vehicles sold each year, 
# calculated as a percentage of the vehicle's purchase cost based on the number of years since the vehicle was purchased.

resale_value = lpSum([
    sell_vars[(year, v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0] *
    get_cost_profile_value(year, int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0]), 'Resale Value %')
    for year in range(2023, 2039)
    for v_id in vehicles['ID']
])

model += buying_costs + insurance_costs + maintenance_costs + fuel_costs - resale_value


# Demand Satisfaction Constraint
for _, row in demand.iterrows():
    model += lpSum([
        use_vars[(row['Year'], v_id)] * vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
        for v_id in vehicles['ID']
        if vehicles.loc[vehicles['ID'] == v_id, 'Size'].values[0] == row['Size'] and 
           vehicles.loc[vehicles['ID'] == v_id, 'Distance'].values[0] >= row['Distance']
    ]) >= row['Demand (km)']

# Carbon Emissions Constraint
for year in carbon_emissions['Year']:
    model += lpSum([
        use_vars[(year, v_id)] * vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Consumption (unit_fuel/km)'].values[0] * 
        fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]) & (fuels['Year'] == year), 'Emissions (CO2/unit_fuel)'].values[0] * 
        vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
        for v_id in vehicles['ID']
    ]) <= carbon_emissions.loc[carbon_emissions['Year'] == year, 'Carbon emission CO2/kg'].values[0]

# Vehicle Purchase and Usage Constraint
for v_id in vehicles['ID']:
    purchase_year = int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0])
    for year in range(purchase_year, min(purchase_year + 10, 2039)):
        model += lpSum([
            buy_vars[(y, v_id)] for y in range(purchase_year, min(year + 1, 2039))
        ]) - lpSum([
            sell_vars[(y, v_id)] for y in range(purchase_year, min(year + 1, 2039))
        ]) >= use_vars[(year, v_id)]

# Vehicle Sale Constraint
for year in range(2023, 2039):
    for v_id in vehicles['ID']:
        model += sell_vars[(year, v_id)] <= 0.2 * lpSum([
            buy_vars[(y, v_id)] - sell_vars[(y, v_id)] for y in range(2023, year)
        ])

# Ensure the results directory exists
results_dir = os.path.join(data_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Solve the problem
model.solve()

# Create lists to store variable info and results
variable_info = []
results = []

# Append Buy, Use, and Sell operations to the results
for year in range(2023, 2039):
    for v_id in vehicles['ID']:
        # Extract purchase year and vehicle attributes
        purchase_year = int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0])
        size = vehicles.loc[vehicles['ID'] == v_id, 'Size'].values[0]
        distance_bucket = vehicles.loc[vehicles['ID'] == v_id, 'Distance'].values[0]

        # Buy operations
        num_bought = int(buy_vars[(year, v_id)].varValue)
        variable_info.append([year, v_id, 'Buy', num_bought])
        if num_bought > 0:
            results.append([year, v_id, num_bought, 'Buy', '', '', ''])

        # Use operations
        num_used = int(use_vars[(year, v_id)].varValue)
        variable_info.append([year, v_id, 'Use', num_used])
        if num_used > 0:
            fuel_type = vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]
            yearly_range = vehicles.loc[vehicles['ID'] == v_id, 'Yearly range (km)'].values[0]
            results.append([year, v_id, num_used, 'Use', fuel_type, distance_bucket, yearly_range])
 
        # Sell operations
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

# Save the variable info to a CSV file
variable_info_df.to_csv(os.path.join(results_dir, 'variable_info.csv'), index=False)

# Save the results to a CSV file
#results_df.to_csv(os.path.join(results_dir, 'submission.csv'), index=False)



def validate_and_correct_submission(submission_df, vehicles, vehicles_fuels, fuels):
    errors = []

    # Check and correct data types
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

    # Check if each vehicle is used within its 10-year lifespan
    for _, row in submission_df.iterrows():
        year = row['Year']
        v_id = row['ID']
        purchase_year = int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0])
        if (year < purchase_year) or (year >= purchase_year + 10):
            errors.append(17)
            break

    # Check that vehicle bought in year YYYY should have YYYY in its ID
    for _, row in submission_df.iterrows():
        year = row['Year']
        v_id = row['ID']
        if not str(year) in v_id:
            errors.append(16)
            break

    # Check if the sum of distance traveled by each vehicle type meets the demand in demand.csv
    demand_met = {}
    for _, row in submission_df.iterrows():
        year = row['Year']
        size = vehicles.loc[vehicles['ID'] == row['ID'], 'Size'].values[0]
        distance_bucket = row['Distance_bucket']
        key = (year, size, distance_bucket)
        demand_met[key] = demand_met.get(key, 0) + row['Distance_per_vehicle(km)'] * row['Num_Vehicles']

    for _, row in demand.iterrows():
        key = (row['Year'], row['Size'], row['Distance'])
        if key not in demand_met or demand_met[key] < row['Demand (km)']:
            errors.append(19)
            break

    # Check if at most 20% of the fleet is sold each year
    for year in range(2023, 2039):
        for v_id in vehicles['ID']:
            total_bought = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Buy')
            ]['Num_Vehicles'].sum()

            total_sold = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Sell')
            ]['Num_Vehicles'].sum()

            if total_sold > 0.2 * total_bought:
                errors.append(21)
                break

    # Check if you only "Use" and "Sell" vehicle IDs that you have in the fleet
    for year in range(2023, 2039):
        for v_id in vehicles['ID']:
            total_bought = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Buy')
            ]['Num_Vehicles'].sum()

            total_sold = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Sell')
            ]['Num_Vehicles'].sum()

            total_used = submission_df[
                (submission_df['Year'] == year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Use')
            ]['Num_Vehicles'].sum()

            if total_used > (total_bought - total_sold):
                errors.append(23)
                break

            if total_sold > (total_bought - total_sold):
                errors.append(25)
                break

    return submission_df, errors


# Validate and correct the submission
results_df, validation_errors = validate_and_correct_submission(results_df, vehicles, vehicles_fuels, fuels)

# Print validation errors if any
if validation_errors:
    print(f"Validation errors found: {validation_errors}")

# Save the corrected results to a CSV file
results_df.to_csv(os.path.join(results_dir, 'submission.csv'), index=False)



