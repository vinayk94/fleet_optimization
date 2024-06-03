import os
import pandas as pd

# Define the data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load datasets
carbon_emissions = pd.read_csv(os.path.join(data_dir, 'carbon_emissions.csv'))
cost_profiles = pd.read_csv(os.path.join(data_dir, 'cost_profiles.csv'))
demand = pd.read_csv(os.path.join(data_dir, 'demand.csv'))
fuels = pd.read_csv(os.path.join(data_dir, 'fuels.csv'))
vehicles = pd.read_csv(os.path.join(data_dir, 'vehicles.csv'))
vehicles_fuels = pd.read_csv(os.path.join(data_dir, 'vehicles_fuels.csv'))

# Load the optimization results
results = pd.read_csv(os.path.join(data_dir, 'optimization_results.csv'))

# Prepare data for verification
vehicles_dict = vehicles.set_index('ID').to_dict('index')
fuels_dict = fuels.set_index(['Fuel', 'Year']).to_dict('index')

# Group vehicles_fuels by ID and sum the consumption values to handle duplicates
vehicles_fuels_grouped = vehicles_fuels.groupby('ID').sum()
vehicles_fuels_dict = vehicles_fuels_grouped.to_dict('index')

# Verify constraints
constraints_satisfied = True
messages = []

# Verify demand fulfillment
for _, row in demand.iterrows():
    year, size, distance, demand_km = row['Year'], row['Size'], row['Distance'], row['Demand (km)']
    supply_km = 0
    
    for _, res_row in results[(results['Year'] == year)].iterrows():
        vehicle_id = res_row['ID']
        num_use = res_row['Num_Use']
        
        if vehicles_dict[vehicle_id]['Size'] == size:
            vehicle_range = vehicles_dict[vehicle_id]['Yearly range (km)']
            supply_km += num_use * vehicle_range
    
    if supply_km < demand_km:
        constraints_satisfied = False
        messages.append(f"Demand not met for year {year}, size {size}, distance {distance}: supply {supply_km} < demand {demand_km}")

# Verify carbon emission limits
for year in carbon_emissions['Year']:
    total_emissions = 0
    
    for _, res_row in results[(results['Year'] == year)].iterrows():
        vehicle_id = res_row['ID']
        num_use = res_row['Num_Use']
        
        if vehicle_id in vehicles_fuels_dict:
            fuel_type = vehicles_fuels_dict[vehicle_id]['Fuel']
            consumption = vehicles_fuels_dict[vehicle_id]['Consumption (unit_fuel/km)']
            
            if (fuel_type, year) in fuels_dict:
                emission_factor = fuels_dict[(fuel_type, year)]['Emissions (CO2/unit_fuel)']
                total_emissions += num_use * consumption * emission_factor
            else:
                messages.append(f"Skipping missing fuel data for vehicle {vehicle_id}, fuel type {fuel_type}, year {year}")
        else:
            messages.append(f"Vehicle {vehicle_id} fuel data is missing.")
    
    emission_limit = carbon_emissions[carbon_emissions['Year'] == year]['Carbon emission CO2/kg'].values[0]
    
    if total_emissions > emission_limit:
        constraints_satisfied = False
        messages.append(f"Emissions exceeded for year {year}: emissions {total_emissions} > limit {emission_limit}")

# Display available fuel type and year combinations
available_fuel_year_combinations = list(fuels_dict.keys())
print("Available fuel type and year combinations:")
for combo in available_fuel_year_combinations:
    print(combo)

print("Constraints Satisfied:", constraints_satisfied)
print("Messages:")
for message in messages:
    print(message)
