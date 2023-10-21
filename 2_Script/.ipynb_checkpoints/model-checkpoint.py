import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

###########################################
###########################################
# MANIPULATING CARBON INTENSITY (CI) DATA
###########################################
###########################################

# Get a data point from the Ember dataset. 
def get_region_data_point(df, region, variable, unit, year):
    
    value = (
        df.copy()
        .query(f'Area == "{region}" & Variable == "{variable}" & Unit == "{unit}" & Year == {year}')
        .reset_index()
        .aggregate({'Value':'sum'})
        ._get_value(0, 'Value')
    )
    
    return value

# Return IPCC 2015 CI score for a specific asset class. 
def return_CI_by_source(source):
    
    # CI data from Ember. Based on IPCC 2015. https://ember-climate.org/app/uploads/2022/03/GER22-Methodology.pdf
    CI_table = [
        ['Coal', 820], 
        ['Gas', 490], 
        ['Other Fossil', 700],
        ['Wind', 11],
        ['Solar', 48],
        ['Bioenergy', 230],
        ['Hydro', 24],
        ['Other Renewables', 38], # Based on IPCC's geothermal
        ['Nuclear', 12]
    ]
    
    # Turn into Dataframe
    CI_table = pd.DataFrame(CI_table, columns=['Source', 'CI_IPCC'])
    
    # Extract value based on "source" input. 
    CI_value = (
        CI_table.copy()
        .query(f"Source == '{source}'")
        .reset_index()
        .aggregate({'CI_IPCC':'sum'})
        ._get_value(0, 'CI_IPCC')
    )
    
    return(CI_value)


# Get the power sector emissions, nuclear generation, and nuclear deficit for Scenario 1
def get_emissions_S1(df, region, year):
    
    var1, unit1 = ('Total Generation', 'TWh')
    var2, unit2 = ('CO2 intensity', 'gCO2/kWh')
    
    twh = get_region_data_point(df=df, region=region, variable=var1, unit=unit1, year=year)
    CI = get_region_data_point(df=df, region=region, variable=var2, unit=unit2, year=year)
    
    # TWh x gCO2/kWh = ktCO2. Divided by 1 000 000 yield GtCO2
    gtCO2_S1 = twh * CI / 1000000
    
    # Get how much nuclear needs to produce in this scenario
    nuclear_needed = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=year)
    
    # 
    nuclear_deficit = 0
    
    return(gtCO2_S1, nuclear_needed, nuclear_deficit)


# Get the power sector emissions, nuclear generation, and nuclear deficit for Scenario 2
def get_emissions_S2(df, region, current_year, baseline_year):
    
    # Get current TWh generation
    generation_current = get_region_data_point(df=df, region=region, variable='Total Generation', unit='TWh', year=current_year)
        
    # Get baseline CI
    CI_baseline = get_region_data_point(df=df, region=region, variable='CO2 intensity', unit='gCO2/kWh', year=baseline_year)
    
    # Calculate current emissions with baseline CI 
    gtCO2_S2 = generation_current * CI_baseline / 1000000 # TWh x gCO2/kWh = ktCO2. Divided by 1 000 000 yield GtCO2
    
    # Calculate necessary growth in nuclear generation
    nuclear_generation_baseline = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=baseline_year)
    nuclear_generation_share = get_region_data_point(df=df, region=region, variable='Nuclear', unit='%', year=baseline_year) / 100 # Percentage
    nuclear_generation_current = generation_current * nuclear_generation_share
    nuclear_deficit = nuclear_generation_current - nuclear_generation_baseline
    if nuclear_deficit < 0:
        nuclear_deficit = 0    
        
    # Get how much nuclear needs to produce in this scenario
    nuclear_needed = nuclear_generation_current
    
    return(gtCO2_S2, nuclear_needed, nuclear_deficit)


# Get the power sector emissions, nuclear generation, and nuclear deficit for Scenarios 3-5
def get_emissions_S345(df, region, current_year, baseline_year, res_growth_reduction): #no_renewables
    
    # Get current TWh generation
    generation_current = get_region_data_point(df=df, region=region, variable='Total Generation', unit='TWh', year=current_year)
        
    # Get current TWh demand
    demand_current = get_region_data_point(df=df, region=region, variable='Demand', unit='TWh', year=current_year)
    

    # Get baseline and current renewables TWh
    wind_generation_baseline = get_region_data_point(df=df, region=region, variable='Wind', unit='TWh', year=baseline_year)
    wind_generation_current = get_region_data_point(df=df, region=region, variable='Wind', unit='TWh', year=current_year)
    solar_generation_baseline = get_region_data_point(df=df, region=region, variable='Solar', unit='TWh', year=baseline_year)
    solar_generation_current = get_region_data_point(df=df, region=region, variable='Solar', unit='TWh', year=current_year)

    # Get growth and adjust generation
    wind_growth = wind_generation_current - wind_generation_baseline
    solar_growth = solar_generation_current - solar_generation_baseline
    
    
    # Adjust amount of RES in model
    wind_growth = wind_growth * res_growth_reduction # res_growth_reduction: 0 = S1 scenario pace, 0.2 = 20% less than S1, 0.4 = 40% less than S1, ... 
    solar_growth = solar_growth * res_growth_reduction
    ws_emissions = wind_growth * return_CI_by_source('Wind') + solar_growth * return_CI_by_source('Solar')
    
    
    # Get baseline and current nuclear TWh
    nuclear_generation_baseline = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=baseline_year)
    nuclear_generation_current = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=current_year)
    
    # Get current fossils TWh
    coal_generation_current = get_region_data_point(df=df, region=region, variable='Coal', unit='TWh', year=current_year)
    other_fossil_generation_current = get_region_data_point(df=df, region=region, variable='Other Fossil', unit='TWh', year=current_year)
    gas_generation_current = get_region_data_point(df=df, region=region, variable='Gas', unit='TWh', year=current_year)
    
    # Calculate lost nuclear TWh (through decommissioning)
    lost_nuclear = nuclear_generation_baseline - nuclear_generation_current
    
    
    # Fill potential demand gap before decommissioning fossils
    # How should we account for net imports here?
    # unserved_demand = demand_current - generation_current
    demand_deficit = solar_growth + wind_growth
    nuclear_deficit = 0
    
    if demand_deficit > 0:
        nuclear_surplus = lost_nuclear - demand_deficit
        if nuclear_surplus < 0:
            nuclear_deficit = nuclear_surplus * (-1)
    else:
        nuclear_surplus = lost_nuclear
    
    # Keep lost nuclear and decommission fossils instead
    replaced_coal = 0
    replaced_other_fossil = 0
    replaced_gas = 0
    
    if nuclear_surplus > 0:
        
        remaining_coal = coal_generation_current - nuclear_surplus
        if remaining_coal < 0:
            
            replaced_coal = coal_generation_current
            nuclear_surplus = nuclear_surplus - coal_generation_current
            
        else:
            
            replaced_coal = nuclear_surplus
            nuclear_surplus = 0
       
    if nuclear_surplus > 0:
        
        remaining_other_fossil = other_fossil_generation_current - nuclear_surplus
        if remaining_other_fossil < 0:
            
            replaced_other_fossil = other_fossil_generation_current
            nuclear_surplus = nuclear_surplus - other_fossil_generation_current
            
        else:
            
            replaced_other_fossil = nuclear_surplus
            nuclear_surplus = 0
    
    if nuclear_surplus > 0:
        
        remaining_gas = gas_generation_current - nuclear_surplus
        if remaining_gas < 0:
            
            replaced_gas = gas_generation_current
            nuclear_surplus = nuclear_surplus - gas_generation_current
            
        else:
            
            replaced_gas = nuclear_surplus
            nuclear_surplus = 0
    
    # ... Make into function. 
    
    if nuclear_surplus > 0:
        nuclear_deficit = 0 - nuclear_surplus
        
    # Calculate avoided emissions
    # TWh x gCO2/kWh = ktCO2. Divided by 1 000 000 yield GtCO2
    avoided_emissions = (
        replaced_coal * return_CI_by_source('Coal') 
        + replaced_other_fossil * return_CI_by_source('Other Fossil') 
        + replaced_gas * return_CI_by_source('Gas') 
        + ws_emissions 
        - (lost_nuclear + nuclear_deficit) * return_CI_by_source('Nuclear')
    )
    avoided_emissions_gtCO2 = avoided_emissions / 1000000
    
    gtCO2_S34 = get_emissions_S1(df=df, region=region, year=current_year)[0] # Get only the emissions
    gtCO2_S34 = gtCO2_S34 - avoided_emissions_gtCO2
    
    # Get how much nuclear needs to produce in this scenario
    nuclear_needed = nuclear_generation_baseline + nuclear_deficit
    
    return(gtCO2_S34, nuclear_needed, nuclear_deficit)


# Generate the minimum RES growth rate (as a percentage of actual growth rate) for nuclear to off-set fossil-based power generation. 
def get_res_equivalent(df, region, current_year, baseline_year): 
    
   # Get baseline and current renewables TWh
    wind_generation_baseline = get_region_data_point(df=df, region=region, variable='Wind', unit='TWh', year=baseline_year)
    wind_generation_current = get_region_data_point(df=df, region=region, variable='Wind', unit='TWh', year=current_year)
    solar_generation_baseline = get_region_data_point(df=df, region=region, variable='Solar', unit='TWh', year=baseline_year)
    solar_generation_current = get_region_data_point(df=df, region=region, variable='Solar', unit='TWh', year=current_year)

    # Get baseline and current nuclear TWh
    nuclear_generation_baseline = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=baseline_year)
    nuclear_generation_current = get_region_data_point(df=df, region=region, variable='Nuclear', unit='TWh', year=current_year)
    
    # Calculate lost nuclear TWh (through decommissioning)
    lost_nuclear = nuclear_generation_baseline - nuclear_generation_current
    
    # Calculate the gained RES TWh (through expansion)
    demand_deficit = (wind_generation_current + solar_generation_current) - (wind_generation_baseline + solar_generation_baseline)
    if demand_deficit < 0 or lost_nuclear <= 0:
        return(0)
    
    # Comparing the TWh of nuclear we have saved with the TWh of RES we have lost in a S5 scenario. How many times over does the nuclear amount cover the RES?
    res_minimum_growth = (demand_deficit - lost_nuclear) / demand_deficit # That is the minimum tolerable growth rate from the S1 baseline.
    
    return(res_minimum_growth)


# Run the data model.
def run_model(df, country, baseline_year, res_reduction_S4):
    
    RES_REDUCTION_S3 = 1 # 100%
    RES_REDUCTION_S4 = res_reduction_S4 # TBD
    RES_REDUCTION_S5 = 0 # 0%
    
    # Set model parameters
    year_list = np.arange(baseline_year+1, 2023)
    source_list = df.query("Subcategory == 'Fuel'").Variable.unique().tolist()
    
    gtCO2_S1_list = []
    gtCO2_S2_list = []
    gtCO2_S3_list = []
    gtCO2_S4_list = []
    gtCO2_S5_list = []
    
    nuc_needed_S1_list = []
    nuc_needed_S2_list = []
    nuc_needed_S3_list = []
    nuc_needed_S4_list = []
    nuc_needed_S5_list = []
    
    nuc_deficit_S1_list = []
    nuc_deficit_S2_list = []
    nuc_deficit_S3_list = []
    nuc_deficit_S4_list = []
    nuc_deficit_S5_list = []
    
    min_res_growth_rate_list = []
    
    for year in year_list:
        
        print(f"Starting year {year}...")
        
        #######################
        # CALCULATE S1 (BASELINE) EMISSIONS
        #######################
        
        gtCO2_S1, nuc_needed, nuc_deficit = get_emissions_S1(df=df, region=country, year=year)
        gtCO2_S1_list.append(gtCO2_S1)
        nuc_needed_S1_list.append(nuc_needed)
        nuc_deficit_S1_list.append(nuc_deficit)
        
        #######################
        # CALCULATE S2 (STATUS QUO) EMISSIONS
        #######################
        
        gtCO2_S2, nuc_needed, nuc_deficit = get_emissions_S2(df=df, region=country, current_year=year, baseline_year=baseline_year)
        gtCO2_S2_list.append(gtCO2_S2)
        nuc_needed_S2_list.append(nuc_needed)
        nuc_deficit_S2_list.append(nuc_deficit) # In this scenario, nuclear will grow its generation when total generation increases (since nuclear retains its mix share).
        
        #######################
        # CALCULATE S3 (NUCLEAR-FIRST) EMISSIONS
        #######################
        
        gtCO2_S3, nuc_needed, nuc_deficit = get_emissions_S345(df=df, region=country, current_year=year, baseline_year=baseline_year, res_growth_reduction=RES_REDUCTION_S3)
        gtCO2_S3_list.append(gtCO2_S3)
        nuc_needed_S3_list.append(nuc_needed)
        nuc_deficit_S3_list.append(nuc_deficit) # In this scenario, there could be a generation deficiency that needs to be filled by expansion of nuclear generation. 
        
        #######################
        # CALCULATE S4 (BALANCED) EMISSIONS
        #######################
        
        gtCO2_S4, nuc_needed, nuc_deficit = get_emissions_S345(df=df, region=country, current_year=year, baseline_year=baseline_year, res_growth_reduction=RES_REDUCTION_S4)
        gtCO2_S4_list.append(gtCO2_S4)
        nuc_needed_S4_list.append(nuc_needed)
        nuc_deficit_S4_list.append(nuc_deficit) # In this scenario, there could be a generation deficiency that needs to be filled by expansion of nuclear generation. 
        
        
        #######################
        # CALCULATE S5 (UTOPIAN) EMISSIONS
        #######################
        
        gtCO2_S5, nuc_needed, nuc_deficit = get_emissions_S345(df=df, region=country, current_year=year, baseline_year=baseline_year, res_growth_reduction=RES_REDUCTION_S5)
        gtCO2_S5_list.append(gtCO2_S5)
        nuc_needed_S5_list.append(nuc_needed)
        nuc_deficit_S5_list.append(nuc_deficit)
        
        #######################
        # CALCULATE MINIMUM RES GROWTH RATE NEEDED TO REPLACE FOSSILS IN MODEL 4
        #######################
        
        res_growth_rate = get_res_equivalent(df=df, region=country, current_year=year, baseline_year=baseline_year) * 100 # Percentages
        min_res_growth_rate_list.append(res_growth_rate)
        
        
    #######################
    # ADD TO DATAFRAME
    #######################
        
    data_scenarios = pd.DataFrame(
        {
            "Year": year_list,
            "GtCO2_S1": gtCO2_S1_list,
            "GtCO2_S2": gtCO2_S2_list,
            "GtCO2_S3": gtCO2_S3_list,
            "GtCO2_S4": gtCO2_S4_list,
            "GtCO2_S5": gtCO2_S5_list,
            
            "TWh_nuc_S1": nuc_needed_S1_list,
            "TWh_nuc_S2": nuc_needed_S2_list,
            "TWh_nuc_S3": nuc_needed_S3_list,
            "TWh_nuc_S4": nuc_needed_S4_list,
            "TWh_nuc_S5": nuc_needed_S5_list,
            
            'TWh_nuc_deficit_S1': nuc_deficit_S1_list,
            'TWh_nuc_deficit_S2': nuc_deficit_S2_list,
            'TWh_nuc_deficit_S3': nuc_deficit_S3_list,
            'TWh_nuc_deficit_S4': nuc_deficit_S4_list,
            'TWh_nuc_deficit_S5': nuc_deficit_S5_list,
            
            '%_RES_growth_needed': min_res_growth_rate_list
        }
    )
    
    return(data_scenarios)


# Visualize the results. 
def vizualize_models(model, data_type, country_name):
    
    search = (
        {
            'emissions':'GtCO2',
            'generation':'TWh_nuc',
            'deficit': 'TWh_nuc_deficit'
        }
    )
    
    try:
        if search[data_type] == 'GtCO2':
            y_label = 'Power sector emissions (GtCO2 / year)'
        elif search[data_type] == 'TWh_nuc':
            y_label = 'Nuclear generation (TWh)'
        elif search[data_type] == 'TWh_nuc_deficit':
            y_label = 'Necessary nuclear expansion (TWh)'
        else:
            return('Wrong data_type. Enter one of the following: emissions | generation | deficit')
    except KeyError:
           return('Wrong data_type. Enter one of the following: emissions | generation | deficit') 
    
    data_viz = model[['Year', 
                      f'{search[data_type]}_S1', 
                      f'{search[data_type]}_S2', 
                      f'{search[data_type]}_S3', 
                      f'{search[data_type]}_S4',
                      f'{search[data_type]}_S5'
                     ]]

    data_viz = (
        data_viz.rename(columns={f'{search[data_type]}_S1':'Baseline (S1)', 
                                 f'{search[data_type]}_S2':'Status Quo (S2)', 
                                 f'{search[data_type]}_S3':'Nuclear-First (S3)', 
                                 f'{search[data_type]}_S4':'Balanced (S4)',
                                 f'{search[data_type]}_S5':'Utopian (S5)'
                                })
        .melt(id_vars=['Year'])
    )
    
    sns.set_context("talk")
    sns.set_style("ticks", {'axes.grid' : True})

    plt.figure(figsize=(15,10))

    ax = sns.pointplot(
        data=data_viz, 
        x="Year", 
        y="value", 
        hue="variable",
        palette="Dark2"
    )
    
    sns.despine()
    
    plt.ylabel(y_label)
    plt.xlabel('Year')
    #plt.suptitle(f'')
    plt.title(f'Model scenarios for {country_name}')

    plt.xticks(rotation=45)
    
    ax.legend(title='Scenarios')
    
    plt.show()
    

# Visualize the RES growth limit for fossil-based power displacement.     
def vizualize_RES(model, country_name):

    data_viz = model[['Year', 
                      '%_RES_growth_needed'
                     ]]

    data_viz = (
        data_viz.rename(columns={f'%_RES_growth_needed':'Necessary RES growth from S1 (baseline)'
                                })
    )

    sns.set_context("talk")
    sns.set_style("ticks", {'axes.grid' : True})

    plt.figure(figsize=(15,10))

    ax = sns.pointplot(
        data=data_viz, 
        x="Year", 
        y="Necessary RES growth from S1 (baseline)",
        color="Tomato"
    )

    sns.despine()

    plt.ylabel('Share of RES growth from S1 (Baseline) scenario')
    plt.xlabel('Year')
    plt.title(f'Minimum cumulative RES growth from S1 baseline to ensure \nthat nuclear generation replaces fossil generation in {country_name}')

    plt.xticks(rotation=45)
    plt.ylim(-20, 100)

    plt.show()
    
    
# Quickly compare certain Ember data points over time and between countries
# Not necessarily for the modelling
def compare_regions(data, reg1, reg2, variable, unit, year1, year2):
    
    reg1_year1 = get_region_data_point(df=data, region=reg1, variable=variable, unit=unit, year=year1)
    reg1_year2 = get_region_data_point(df=data, region=reg1, variable=variable, unit=unit, year=year2)
    reg2_year1 = get_region_data_point(df=data, region=reg2, variable=variable, unit=unit, year=year1)
    reg2_year2 = get_region_data_point(df=data, region=reg2, variable=variable, unit=unit, year=year2)
    
    print('--------------------------------------')
    
    print(f'{reg1}|{year1}: {reg1_year1.round()}')
    print(f'{reg1}|{year2}: {reg1_year2.round()}')
    
    print('\n')
    
    print(f'{reg2}|{year1}: {reg2_year1.round()}')
    print(f'{reg2}|{year2}: {reg2_year2.round()}')
    
    print('\n')
    
    print(f'{reg1}|{year1}-{year2}: {round(reg1_year2/reg1_year1-1, 2)}')
    print(f'{reg2}|{year1}-{year2}: {round(reg2_year2/reg2_year1-1, 2)}')
    
    print('\n')
    
    print(f'{reg1} vs {reg2}|{year1}: {round(reg1_year1/reg2_year1-1, 2)}')
    print(f'{reg1} vs {reg2}|{year2}: {round(reg1_year2/reg2_year2-1, 2)}')
    
    print('--------------------------------------')