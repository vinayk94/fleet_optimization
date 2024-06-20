import pandas as pd
import os

# Define the data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load datasets
carbon_emissions = pd.read_csv(os.path.join(data_dir, 'carbon_emissions.csv'))
cost_profiles = pd.read_csv(os.path.join(data_dir, 'cost_profiles.csv'))
demand = pd.read_csv(os.path.join(data_dir, 'demand.csv'))
fuels = pd.read_csv(os.path.join(data_dir, 'fuels.csv'))
vehicles = pd.read_csv(os.path.join(data_dir, 'vehicles.csv'))
vehicles_fuels = pd.read_csv(os.path.join(data_dir, 'vehicles_fuels.csv'))

# Load the files
submission_df = pd.read_csv(os.path.join(data_dir, 'results','submission.csv'))
sample_submission_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))


def get_cost_profile_value(year, purchase_year, cost_type):
    year_offset = year - purchase_year
    value = cost_profiles.loc[cost_profiles['End of Year'] == year_offset, cost_type]
    if not value.empty:
        return value.values[0] / 100
    return 0

def calculate_total_cost(submission_df):
    total_cost = 0
    fleet = {}  # Fleet size by year and vehicle ID

    for year in range(2023, 2039):
        fleet[year] = {v_id: 0 for v_id in vehicles['ID']}
        for _, row in submission_df[submission_df['Year'] == year].iterrows():
            v_id = row['ID']
            num_vehicles = row['Num_Vehicles']
            op_type = row['Type']
            vehicle_cost = vehicles.loc[vehicles['ID'] == v_id, 'Cost ($)'].values[0]

            if op_type == 'Buy':
                fleet[year][v_id] += num_vehicles
                total_cost += num_vehicles * vehicle_cost

            elif op_type == 'Use':
                fuel_type = vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Fuel'].values[0]
                consumption = vehicles_fuels.loc[vehicles_fuels['ID'] == v_id, 'Consumption (unit_fuel/km)'].values[0]
                fuel_cost_per_unit = fuels.loc[(fuels['Fuel'] == fuel_type) & (fuels['Year'] == year), 'Cost ($/unit_fuel)'].values[0]
                yearly_range = row['Distance_per_vehicle(km)']
                total_cost += num_vehicles * yearly_range * consumption * fuel_cost_per_unit

            elif op_type == 'Sell':
                fleet[year][v_id] -= num_vehicles
                purchase_year = vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0]
                resale_value = get_cost_profile_value(year, purchase_year, 'Resale Value %')
                total_cost -= num_vehicles * vehicle_cost * resale_value

        # Apply insurance and maintenance costs to the entire fleet
        for v_id, num_vehicles in fleet[year].items():
            if num_vehicles > 0:
                purchase_year = vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0]
                insurance_cost = get_cost_profile_value(year, purchase_year, 'Insurance Cost %')
                maintenance_cost = get_cost_profile_value(year, purchase_year, 'Maintenance Cost %')
                total_cost += num_vehicles * vehicle_cost * (insurance_cost + maintenance_cost)

    return total_cost

"""
# Calculate total costs for both submissions
total_cost_submission = calculate_total_cost(submission_df)
total_cost_sample_submission = calculate_total_cost(sample_submission_df)

# Calculate the percentage improvement
percentage_improvement = ((total_cost_sample_submission - total_cost_submission) / total_cost_sample_submission) * 100

print(f"Total cost of submission: {total_cost_submission}")
print(f"Total cost of sample submission: {total_cost_sample_submission}")

if total_cost_submission < total_cost_sample_submission:
    print("Your solution has a lower total cost.")
else:
    print("Sample submission has a lower total cost.")
print(f"Percentage improvement: {percentage_improvement:.2f}%")

"""


def validate_demand_satisfaction(submission_df, demand, vehicles, name="Submission"):
    for _, row in demand.iterrows():
        year_demand = row['Year']
        size = row['Size']
        distance = row['Distance']
        total_demand = row['Demand (km)']

        satisfied_demand = submission_df[
            (submission_df['Year'] == year_demand) & 
            (submission_df['Type'] == 'Use')
        ].apply(lambda x: (
            vehicles.loc[vehicles['ID'] == x['ID'], 'Size'].values[0] == size and 
            vehicles.loc[vehicles['ID'] == x['ID'], 'Distance'].values[0] >= distance
        ) * x['Num_Vehicles'] * vehicles.loc[vehicles['ID'] == x['ID'], 'Yearly range (km)'].values[0], axis=1).sum()

        if satisfied_demand < total_demand:
            print(f"{name}: Demand not satisfied for year {year_demand}, size {size}, distance {distance}")
            return False
    return True


def validate_carbon_emissions(submission_df, vehicles_fuels, fuels, carbon_emissions, vehicles, name="Submission"):
    for year in carbon_emissions['Year']:
        total_emissions = submission_df[
            (submission_df['Year'] == year) & 
            (submission_df['Type'] == 'Use')
        ].apply(lambda x: (
            vehicles_fuels.loc[vehicles_fuels['ID'] == x['ID'], 'Consumption (unit_fuel/km)'].values[0] *
            fuels.loc[(fuels['Fuel'] == vehicles_fuels.loc[vehicles_fuels['ID'] == x['ID'], 'Fuel'].values[0]) & (fuels['Year'] == year), 'Emissions (CO2/unit_fuel)'].values[0] *
            x['Num_Vehicles'] * vehicles.loc[vehicles['ID'] == x['ID'], 'Yearly range (km)'].values[0]
        ), axis=1).sum()

        emission_limit = carbon_emissions.loc[carbon_emissions['Year'] == year, 'Carbon emission CO2/kg'].values[0]
        if total_emissions > emission_limit:
            print(f"{name}: Carbon emissions exceeded for year {year}")
            return False
    return True


def validate_vehicle_purchase_usage(submission_df, vehicles, name="Submission"):
    fleet_size = {}
    for year in range(2023, 2039):
        for v_id in vehicles['ID']:
            purchase_year = int(vehicles.loc[vehicles['ID'] == v_id, 'Year'].values[0])
            if purchase_year <= year < purchase_year + 10:
                bought = submission_df[
                    (submission_df['Year'] <= year) & 
                    (submission_df['ID'] == v_id) & 
                    (submission_df['Type'] == 'Buy')
                ]['Num_Vehicles'].sum()

                sold = submission_df[
                    (submission_df['Year'] <= year) & 
                    (submission_df['ID'] == v_id) & 
                    (submission_df['Type'] == 'Sell')
                ]['Num_Vehicles'].sum()

                used = submission_df[
                    (submission_df['Year'] == year) & 
                    (submission_df['ID'] == v_id) & 
                    (submission_df['Type'] == 'Use')
                ]['Num_Vehicles'].sum()

                if used > (bought - sold):
                    print(f"{name}: More vehicles used than bought for vehicle ID {v_id} in year {year}")
                    return False
    return True


def validate_vehicle_sale(submission_df, vehicles, name="Submission"):
    for year in range(2023, 2039):
        for v_id in vehicles['ID']:
            bought = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Buy')
            ]['Num_Vehicles'].sum()

            sold = submission_df[
                (submission_df['Year'] <= year) & 
                (submission_df['ID'] == v_id) & 
                (submission_df['Type'] == 'Sell')
            ]['Num_Vehicles'].sum()

            if sold > 0.2 * bought:
                print(f"{name}: More than 20% of vehicles sold for vehicle ID {v_id} in year {year}")
                return False
    return True


def validate_constraints(submission_df, demand, carbon_emissions, vehicles, vehicles_fuels, fuels, name="Submission"):
    if not validate_demand_satisfaction(submission_df, demand, vehicles, name):
        return False
    if not validate_carbon_emissions(submission_df, vehicles_fuels, fuels, carbon_emissions, vehicles, name):
        return False
    if not validate_vehicle_purchase_usage(submission_df, vehicles, name):
        return False
    if not validate_vehicle_sale(submission_df, vehicles, name):
        return False
    return True


# Validate both submissions
valid_submission = validate_constraints(submission_df, demand, carbon_emissions, vehicles, vehicles_fuels, fuels, name="Your Submission")
valid_sample_submission = validate_constraints(sample_submission_df, demand, carbon_emissions, vehicles, vehicles_fuels, fuels, name="Sample Submission")

# Calculate total costs for both submissions
total_cost_submission = calculate_total_cost(submission_df)
total_cost_sample_submission = calculate_total_cost(sample_submission_df)

# Calculate the percentage improvement
percentage_improvement = ((total_cost_sample_submission - total_cost_submission) / total_cost_sample_submission) * 100

print(f"Total cost of your submission: {total_cost_submission}")
print(f"Total cost of sample submission: {total_cost_sample_submission}")
print(f"Percentage improvement: {percentage_improvement:.2f}%")

# Print validation results
if valid_submission:
    print("Your submission satisfies all constraints.")
else:
    print("Your submission does not satisfy all constraints.")

if valid_sample_submission:
    print("Sample submission satisfies all constraints.")
else:
    print("Sample submission does not satisfy all constraints.")






