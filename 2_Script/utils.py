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

# Recalculate the carbon intensity (CI, gCO2e/kWh) of a region based on select countries you wish to exclude from said region. 
# E.g., if you wish to compare Germany's CI performance vs the EU, you need to exclude Germany from the EU group average. 
def calc_new_CI(data, region, country_list, year):
    
    # Get one or more countries' power sector emissions (MtCO2/yr)
    country_emissions = (
        data.copy()
        .query(f'Area in ({country_list}) & Variable == "Total emissions" & Year == {year}')
        .reset_index()
        .aggregate({'Value':'sum'})
        ._get_value(0, 'Value')
    )
    
    # Get the region's power sector emissions (MtCO2/yr)
    region_emissions = (
        data.copy()
        .query(f'Area == "{region}" & Variable == "Total emissions" & Year == {year}')
        .reset_index()
        ._get_value(0, 'Value')
    )
    
    # Get one or more countries' power sector generation (TWh)
    country_generation = (
        data.copy()
        .query(f'Area in ({country_list}) & Category == "Electricity generation" & Variable == "Total Generation" & Year == {year}')
        .reset_index()
        .aggregate({'Value':'sum'})
        ._get_value(0, 'Value')
    )
    
    # Get the region's power sector generation (TWh)
    region_generation = (
        data.copy()
        .query(f'Area == "{region}" & Category == "Electricity generation" & Variable == "Total Generation" & Year == {year}')
        .reset_index()
        ._get_value(0, 'Value')
    )
    
    # Calculate select countries' share of region's emissions (%)
    country_emissions_share = (region_emissions - country_emissions) / region_emissions
    #print(f'Emission share excl. {country_list}: {country_emissions_share.round(2)}')
    
    # Calculate select countries' share of region's generation (%)
    country_generation_share = (region_generation - country_generation) / region_generation
    #print(f'Generation share excl. {country_list}: {country_generation_share.round(2)}')
    
    # Calculate the constant by which regional CI needs to be adjusted
    country_adjuster = country_emissions_share / country_generation_share
    #print(f'{country_list} CI adjuster: {country_adjuster.round(2)}')
    
    # Calculate CI by dividing total regional emissions with generation and multiplying with the CI adjuster
    new_CI = (region_emissions / region_generation) * country_adjuster * 1000
    
    old_CI = (
        data.copy()
        .query(f'Area == "{region}" & Variable == "CO2 intensity" & Year == {year}')
        .reset_index()
        ._get_value(0, 'Value')
    )
    
    print(f'{year} | {region} CI: {new_CI.round(1)} <-- {old_CI.round(1)}')
    
    return new_CI



# Update the carbon intensity (CI, gCO2e/kWh) of a region based on select countries you wish to exclude from said region. 
def update_CI(data, region, country_list, year):    
    data.Value = (
        np.where(
            ((data.Variable=="CO2 intensity") & (data.Area==region) & (data.Year==year)), 
            calc_new_CI(data=data, region=region, country_list=country_list, year=year), 
            data.Value)
    )
    

# Change the country list to exclude based on which regions to update. 
def adapt_country_list(region):
    
    if region in ['Asia']:
        country_list = ['Japan']
        
    elif region in ['Europe', 'EU']:
        country_list = ['Germany', 'Lithuania']
                
    else:
        country_list = ['Germany', 'Japan', 'Lithuania']
    
    print(f'{region} --> {country_list}')
    return country_list


###########################################
###########################################
# HELP FUNCTIONS
###########################################
###########################################

def available_info(df):
    
    print("The following exists described in the data:\n")
    
    print("AREAS:")
    print(f"{df.Area.unique().tolist()}")
    
    print("\nVARIABLES")
    print(f"{df.Variable.unique().tolist()}")
    
    print("\nUNITS")
    print(f"{df.Unit.unique().tolist()}")


###########################################
###########################################
# CREATING MASTER FILE
###########################################
###########################################

def get_region_data(df, region_list, variable, unit):
    
    output = (
        df.copy()
        .query(f'Area in ({region_list}) & Variable == "{variable}" & Unit == "{unit}"')
        .reset_index()
    )
    
    output = output[['Area', 'Year', 'Category', 'Subcategory', 'Variable', 'Unit', 'Value']]
    
    return output.round(3)

def get_region_data_point(df, region, variable, unit, year):
    
    value = (
        df.copy()
        .query(f'Area == "{region}" & Variable == "{variable}" & Unit == "{unit}" & Year == {year}')
        .reset_index()
        .aggregate({'Value':'sum'})
        ._get_value(0, 'Value')
    )
    
    return value

def create_df_twh_by_year(df, year):

    filter_years=[year]
    filter_variables=['Hydro', 'Wind and Solar', 'Bioenergy', 'Other Renewables', 'Nuclear', 'Coal', 'Gas', 'Other Fossil', 'Net Imports', 'Demand'] # 'Renewables', 'Demand per capita' 
    filter_units=['TWh', 'mtCO2']
    #filter_area_types=['Country']

    data_twh = (
        df.copy()
        .query('Year == @filter_years')
        .query('Variable == @filter_variables')
        .query('Unit == @filter_units')
        .rename(columns={'Area type':'Area_type'})
        #.query('Area_type == @filter_area_types')
    )

    data_twh["Year"] = data_twh["Year"].apply(str)
    data_twh["New Variable"] = data_twh[["Category", "Variable", "Unit"]].apply(lambda x: " - ".join(x), axis=1)
    data_twh = data_twh[["Area", "Year", "New Variable", "Value"]] # "YoY % change", "YoY absolute change"

    data_twh = (
        data_twh.pivot(index=['Area'], 
                   columns='New Variable',
                   values=['Value']) # "YoY % change", "YoY absolute change"
        .reset_index()
        .droplevel(0, axis=1) 
        #.fillna(0) # Have or not have?
    )

    data_twh.columns.values[0] = 'Area'

    data_twh = (
        data_twh.rename(columns={'Electricity demand - Demand - TWh': f'Demand_TWh{year}', 
                                 'Electricity generation - Bioenergy - TWh': f'Bioenergy_TWh{year}',
                                 'Electricity generation - Coal - TWh': f'Coal_TWh{year}',
                                 'Electricity generation - Gas - TWh': f'Gas_TWh{year}',
                                 'Electricity generation - Hydro - TWh':  f'Hydro_TWh{year}',
                                 'Electricity generation - Nuclear - TWh': f'Nuclear_TWh{year}',
                                 'Electricity generation - Other Fossil - TWh': f'Other_Fossil_TWh{year}',
                                 'Electricity generation - Other Renewables - TWh': f'Other_Renewables_TWh{year}', 
                                 'Electricity generation - Wind and Solar - TWh': f'Wind_and_Solar_TWh{year}',
                                 'Power sector emissions - Bioenergy - mtCO2': f'Bioenergy_mtCO2{year}',
                                 'Power sector emissions - Coal - mtCO2': f'Coal_mtCO2{year}',
                                 'Power sector emissions - Gas - mtCO2': f'Gas_mtCO2{year}',
                                 'Power sector emissions - Hydro - mtCO2':  f'Hydro_mtCO2{year}',
                                 'Power sector emissions - Nuclear - mtCO2': f'Nuclear_mtCO2{year}',
                                 'Power sector emissions - Other Fossil - mtCO2': f'Other_Fossil_mtCO2{year}',
                                 'Power sector emissions - Other Renewables - mtCO2': f'Other_Renewables_mtCO2{year}', 
                                 'Power sector emissions - Wind and Solar - mtCO2': f'Wind_and_Solar_mtCO2{year}',
                                 'Electricity imports - Net Imports - TWh': f'Net_Imports_TWh{year}'
                            })
    )
    
    return(data_twh)



def create_df_mix_by_year(df, year):
    
    filter_years=[year]
    filter_variables=['Hydro', 'Wind and Solar', 'Bioenergy', 'Other Renewables', 'Nuclear', 'Coal', 'Gas', 'Other Fossil']
    filter_units=['%'] 

    data = (
        df.copy()
        .query('Year == @filter_years')
        .query('Variable == @filter_variables')
        .query('Unit == @filter_units')
        .rename(columns={'Area type':'Area_type'})
    )

    data["Year"] = data["Year"].apply(str)
    data["New Variable"] = data[["Category", "Variable", "Unit"]].apply(lambda x: " - ".join(x), axis=1)
    data = data[["Area", "Year", "New Variable", "Value"]] # "YoY % change", "YoY absolute change"


    data = (
        data.pivot(index=['Area'], 
                   columns='New Variable',
                   values=['Value']) # "YoY % change", "YoY absolute change"
        .reset_index()
        .droplevel(0, axis=1) 
    )

    data.columns.values[0] = 'Area'

    data = (
        data.rename(columns={'Electricity generation - Bioenergy - %': f'Bioenergy_%mix{year}',
                                         'Electricity generation - Coal - %':f'Coal_%mix{year}',
                                         'Electricity generation - Gas - %':f'Gas_%mix{year}',
                                         'Electricity generation - Hydro - %': f'Hydro_%mix{year}',
                                         'Electricity generation - Nuclear - %': f'Nuclear_%mix{year}',
                                         'Electricity generation - Other Fossil - %': f'Other_Fossil_%mix{year}',
                                         'Electricity generation - Other Renewables - %': f'Other_Renewables_%mix{year}', 
                                         'Electricity generation - Wind and Solar - %':f'Wind_and_Solar_%mix{year}'
                            })
    )
    
    return data



def create_df_mix_shift_between_years(df, year_list): 

    filter_years=year_list
    max_year=str(max(year_list))
    min_year= str(min(year_list))
    filter_variables=['Hydro', 'Wind and Solar', 'Bioenergy', 'Other Renewables', 'Nuclear', 'Coal', 'Gas', 'Other Fossil', 'CO2 intensity']
    filter_units=['%', 'gCO2/kWh']

    data = (
        df.copy()
        .query('Year == @filter_years')
        .query('Variable == @filter_variables')
        .query('Unit == @filter_units')
        .rename(columns={'Area type':'Area_type'})
    )

    data["Year"] = data["Year"].apply(str)
    data["New Variable"] = data[["Category", "Variable", "Unit"]].apply(lambda x: " - ".join(x), axis=1)
    data = data[["Area", "Year", "New Variable", "Value"]] # "YoY % change", "YoY absolute change"

    data = (
        data.melt(id_vars=['Area', 'Year', 'New Variable'], value_vars=['Value'])
        .drop(columns={'variable'})
        .pivot(index=['Area', 'New Variable'], 
                   columns='Year',
                   values=['value']) # "YoY % change", "YoY absolute change"
        .reset_index()
        .droplevel(0, axis=1) 
    )

    data.columns.values[0] = 'Area'
    data.columns.values[1] = 'Variable'

    data = (
        data.assign(Change=lambda x: x[max_year] - x[min_year]) # Because % shares are in absolute.
        .drop(columns={min_year, max_year})
        .pivot(index=['Area'],
              columns='Variable', 
              values='Change')
        .reset_index()
        .round(5)
        .replace([np.inf, -np.inf], np.nan)
    )
    
    separator='_'
    id_tag=separator.join([min_year, max_year])

    # ppmix = percentage point mix change
    data = (
        data.rename(columns={'Electricity generation - Bioenergy - %': f'Bioenergy_ppmix{id_tag}',
                                        'Electricity generation - Coal - %': f'Coal_ppmix{id_tag}',
                                        'Electricity generation - Gas - %': f'Gas_ppmix{id_tag}',
                                        'Electricity generation - Hydro - %': f'Hydro_ppmix{id_tag}',
                                        'Electricity generation - Nuclear - %': f'Nuclear_ppmix{id_tag}',
                                        'Electricity generation - Other Fossil - %': f'Other_Fossil_ppmix{id_tag}',
                                        'Electricity generation - Other Renewables - %': f'Other_Renewables_ppmix{id_tag}', 
                                        'Electricity generation - Wind and Solar - %': f'Wind_and_Solar_ppmix{id_tag}',
                                        'Power sector emissions - CO2 intensity - gCO2/kWh': f'CI_change{id_tag}'
                            })
    )
    
    return(data)


def create_df_percentage_shift_between_years(df, unit, year_list): 

    filter_years=year_list
    max_year=str(max(year_list))
    min_year= str(min(year_list))
    filter_variables=['Hydro', 'Wind and Solar', 'Bioenergy', 'Other Renewables', 'Nuclear', 'Coal', 'Gas', 'Other Fossil', 'CO2 intensity']
    filter_units=[f'{unit}']
    #filter_area_types=['Country']

    data = (
        df.copy()
        .query('Year == @filter_years')
        .query('Variable == @filter_variables')
        .query('Unit == @filter_units')
        .rename(columns={'Area type':'Area_type'})
        #.query('Area_type == @filter_area_types')
    )

    data["Year"] = data["Year"].apply(str)
    data["New Variable"] = data[["Category", "Variable", "Unit"]].apply(lambda x: " - ".join(x), axis=1)
    data = data[["Area", "Year", "New Variable", "Value"]] # "YoY % change", "YoY absolute change"

    data = (
        data.melt(id_vars=['Area', 'Year', 'New Variable'], value_vars=['Value'])
        .drop(columns={'variable'})
        .pivot(index=['Area', 'New Variable'], 
                   columns='Year',
                   values=['value']) # "YoY % change", "YoY absolute change"
        .reset_index()
        .droplevel(0, axis=1) 
        #.fillna(0) # Have or not have?
    )

    data.columns.values[0] = 'Area'
    data.columns.values[1] = 'Variable'

    data = (
        data.assign(Change=lambda x: x[max_year] / x[min_year] - 1) 
        .drop(columns={min_year, max_year})
        .pivot(index=['Area'],
              columns='Variable', 
              values='Change')
        .reset_index()
        .round(5)
        .replace([np.inf, -np.inf], np.nan)
    )
    
    separator='_'
    id_tag=separator.join([min_year, max_year])

    data = (
        data.rename(columns={f'Electricity generation - Bioenergy - {unit}': f'Bioenergy_%{unit}{id_tag}',
                                        f'Electricity generation - Coal - {unit}': f'Coal_%{unit}{id_tag}',
                                        f'Electricity generation - Gas - {unit}': f'Gas_%{unit}{id_tag}',
                                        f'Electricity generation - Hydro - {unit}': f'Hydro_%{unit}{id_tag}',
                                        f'Electricity generation - Nuclear - {unit}': f'Nuclear_%{unit}{id_tag}',
                                        f'Electricity generation - Other Fossil - {unit}': f'Other_Fossil_%{unit}{id_tag}',
                                        f'Electricity generation - Other Renewables - {unit}': f'Other_Renewables_%{unit}{id_tag}', 
                                        f'Electricity generation - Wind and Solar - {unit}': f'Wind_and_Solar_%{unit}{id_tag}'
                            })
    )
    
    return(data)



def create_df_ci_data(df):

    #filter_years=[2000, 2009, 2019, 2020, 2021, 2022]

    data_CI_indices = (
        df.copy()
        .query('Category == "Power sector emissions" & Variable == "CO2 intensity"')
        #.query('Year == @filter_years')
        .groupby(['Area', 'Year', 'Category', 'Subcategory', 'Variable', 'Unit'])
        .aggregate({'Value':'sum'})
        .reset_index()
        .drop(columns=['Category', 'Subcategory', 'Unit'], axis=1)
        .pivot(index='Area', columns='Year', values='Value')
        .assign(CI_index_2000=1)
        .assign(CI_index_2001=lambda x: x[2001] / x[2000])
        .assign(CI_index_2002=lambda x: x[2002] / x[2000])
        .assign(CI_index_2003=lambda x: x[2003] / x[2000])
        .assign(CI_index_2004=lambda x: x[2004] / x[2000])
        .assign(CI_index_2005=lambda x: x[2005] / x[2000])
        .assign(CI_index_2006=lambda x: x[2006] / x[2000])
        .assign(CI_index_2007=lambda x: x[2007] / x[2000])
        .assign(CI_index_2008=lambda x: x[2008] / x[2000])
        .assign(CI_index_2009=lambda x: x[2009] / x[2000])
        .assign(CI_index_2010=lambda x: x[2010] / x[2000])
        .assign(CI_index_2011=lambda x: x[2011] / x[2000])
        .assign(CI_index_2012=lambda x: x[2012] / x[2000])
        .assign(CI_index_2013=lambda x: x[2013] / x[2000])
        .assign(CI_index_2014=lambda x: x[2014] / x[2000])
        .assign(CI_index_2015=lambda x: x[2015] / x[2000])
        .assign(CI_index_2016=lambda x: x[2016] / x[2000])
        .assign(CI_index_2017=lambda x: x[2017] / x[2000])
        .assign(CI_index_2018=lambda x: x[2018] / x[2000])
        .assign(CI_index_2019=lambda x: x[2019] / x[2000])
        .assign(CI_index_2020=lambda x: x[2020] / x[2000])
        .assign(CI_index_2021=lambda x: x[2021] / x[2000])
        .assign(CI_index_2022=lambda x: x[2022] / x[2000])
        .rename(columns={2022: 'CI_2022', 
                         2021: 'CI_2021',
                         2020: 'CI_2020',
                         2019: 'CI_2019',
                         2018: 'CI_2018',
                         2017: 'CI_2017',
                         2016: 'CI_2016',
                         2015: 'CI_2015',
                         2014: 'CI_2014',
                         2013: 'CI_2013',
                         2012: 'CI_2012',
                         2011: 'CI_2011',
                         2010: 'CI_2010',
                         2009: 'CI_2009',
                         2008: 'CI_2008',
                         2007: 'CI_2007',
                         2006: 'CI_2006',
                         2005: 'CI_2005',
                         2004: 'CI_2004',
                         2003: 'CI_2003',
                         2002: 'CI_2002',
                         2001: 'CI_2001',
                         2000: 'CI_2000'})
        .round(2)
        .replace([np.inf, -np.inf], np.nan)
        .reset_index()
    )
    
    data_CI_categories = data_CI_indices[['Area', 'CI_2022', 'CI_2021', 'CI_index_2022', 'CI_index_2021']]

    return data_CI_indices, data_CI_categories




def create_df_other_traits(df):
    
    info_table = (
        df.copy()
        .drop(columns=['Year', 'Category', 'Subcategory', 'Variable', 'Unit', 'Value', 'YoY absolute change', 'YoY % change'])
        .groupby(['Area', 'Country code', 'Area type', 'Continent', 'Ember region'])
        .aggregate({'EU':'mean', 'OECD':'mean','G20':'mean','G7':'mean','ASEAN':'mean'})
        .reset_index()
        .rename(columns={'Country code':'country_code', 'Area type':'area_type','Ember region':'ember_region'})        
    )
    return info_table



def create_df_mix_shift_all_years(df):
    filter_variables=['Hydro', 'Wind and Solar', 'Bioenergy', 'Other Renewables', 'Nuclear', 'Coal', 'Gas', 'Other Fossil', 'CO2 intensity']
    filter_units=['%', 'gCO2/kWh']

    data = (
        df.copy()
        .query('Variable == @filter_variables')
        .query('Unit == @filter_units')
        .rename(columns={'Area type':'area_type'})
        .query('area_type == "Country"')
    )

    data["New Variable"] = data[["Category", "Variable", "Unit"]].apply(lambda x: " - ".join(x), axis=1)
    data = data[["Area", "Year", "New Variable", "Value"]] # "YoY % change", "YoY absolute change"

    data = data.sort_values(by=['Area', 'New Variable', 'Year'])
    data['diffs'] = data['Value'].diff()
    mask = data['New Variable'] != data['New Variable'].shift(1)
    data['diffs'][mask] = np.nan

    data = (
        data.pivot(index=['Area', 'Year'], 
                   columns='New Variable',
                   values=['diffs'])
        .reset_index()
        .droplevel(0, axis=1) 
    )

    data.columns.values[0] = 'Area'
    data.columns.values[1] = 'Year'

    tag = "pp_change"
    data = (
            data.rename(columns={f'Electricity generation - Bioenergy - %': f'Bioenergy_{tag}',
                                 f'Electricity generation - Coal - %': f'Coal_{tag}',
                                 f'Electricity generation - Gas - %': f'Gas_{tag}',
                                 f'Electricity generation - Hydro - %': f'Hydro_{tag}',
                                 f'Electricity generation - Nuclear - %': f'Nuclear_{tag}',
                                 f'Electricity generation - Other Fossil - %': f'Other_Fossil_{tag}',
                                 f'Electricity generation - Other Renewables - %': f'Other_Renewables_{tag}', 
                                 f'Electricity generation - Wind and Solar - %': f'Wind_and_Solar_{tag}',
                                 'Power sector emissions - CO2 intensity - gCO2/kWh': 'CI_change'
                                }
                       )
        )
    
    return(data)



def create_df_master(df):
    
    ##############################
    # Create multiple dataframes with various columns based on manipulations of available Ember data.
    ##############################
    data_twh_2000 = create_df_twh_by_year(df=df, year=2000)
    data_twh_2021 = create_df_twh_by_year(df=df, year=2021)
    data_twh_2022 = create_df_twh_by_year(df=df, year=2022)

    data_mix_states_2000 = create_df_mix_by_year(df=df, year=2000)
    data_mix_states_2021 = create_df_mix_by_year(df=df, year=2021)
    data_mix_states_2022 = create_df_mix_by_year(df=df, year=2022)

    data_mix_shifts_2000_2021 = create_df_mix_shift_between_years(df=df, year_list=[2000, 2021])
    data_mix_shifts_2000_2022 = create_df_mix_shift_between_years(df=df, year_list=[2000, 2022])

    data_twh_shifts = create_df_percentage_shift_between_years(df=df, unit='TWh', year_list=[2000, 2021])

    data_CI_indices, data_CI_categories = create_df_ci_data(df=df)

    data_other_traits = create_df_other_traits(df=df)
    
    ##############################
    # Merge all dataframes into single dataframe.
    ##############################
    data_master = (
        data_twh_2000.copy()
        .merge(data_twh_2021, how="outer", on=['Area'])
        .merge(data_twh_2022, how="outer", on=['Area'])
        .merge(data_mix_states_2000, how="outer", on=['Area'])
        .merge(data_mix_states_2021, how="outer", on=['Area'])
        .merge(data_mix_states_2022, how="outer", on=['Area'])
        .merge(data_twh_shifts, how="outer", on=['Area'])
        .merge(data_mix_shifts_2000_2021, how="outer", on=['Area'])
        .merge(data_mix_shifts_2000_2022, how="outer", on=['Area'])
        .merge(data_CI_indices, how="outer", on=['Area']) #data_CI_categories
        .merge(data_other_traits, how="outer", on=['Area'])
        #.merge(green_parties, how="outer", on=['Area'])
    )
    
    ##############################
    # CI index categorization
    ##############################
    conditions = [
        (data_master['CI_index_2021'] <= 0.75),
        (data_master['CI_index_2021'] > 0.75) & (data_master['CI_index_2021'] <= 0.95),
        (data_master['CI_index_2021'] > 0.95) & (data_master['CI_index_2021'] <= 1.05),
        (data_master['CI_index_2021'] > 1.05) & (data_master['CI_index_2021'] < 1.25),
        (data_master['CI_index_2021'] >= 1.25)
    ]

    values = [
        'Major Decrease', 
        'Minor Decrease', 
        'Limited Change', 
        'Minor Increase', 
        'Major Increase'
    ]

    data_master['CI_index_2021_cat'] = np.select(conditions, values)

    ##############################
    # CI score categorization
    ##############################
    q1 = data_master.CI_2021.quantile(0.05)
    q2 = data_master.CI_2021.quantile(0.30)
    q3 = data_master.CI_2021.quantile(0.70)
    q4 = data_master.CI_2021.quantile(0.95)

    conditions = [
        (data_master['CI_2021'] <= q1),
        (data_master['CI_2021'] > q1) & (data_master['CI_2021'] <= q2),
        (data_master['CI_2021'] > q2) & (data_master['CI_2021'] <= q3),
        (data_master['CI_2021'] > q3) & (data_master['CI_2021'] <= q4),
        (data_master['CI_2021'] > q4) 
    ]

    values = [
        'Very Low', 
        'Low', 
        'Medium', 
        'High', 
        'Very High'
    ]

    data_master['CI_2021_cat'] = np.select(conditions, values)

    q1 = data_master.CI_2000.quantile(0.05)
    q2 = data_master.CI_2000.quantile(0.30)
    q3 = data_master.CI_2000.quantile(0.70)
    q4 = data_master.CI_2000.quantile(0.95)

    conditions = [
        (data_master['CI_2000'] <= q1),
        (data_master['CI_2000'] > q1) & (data_master['CI_2000'] <= q2),
        (data_master['CI_2000'] > q2) & (data_master['CI_2000'] <= q3),
        (data_master['CI_2000'] > q3) & (data_master['CI_2000'] <= q4),
        (data_master['CI_2000'] > q4) 
    ]

    values = [
        'Very Low', 
        'Low', 
        'Medium', 
        'High', 
        'Very High'
    ]

    data_master['CI_2000_cat'] = np.select(conditions, values)

    ##############################
    # Nuclear power change categorization
    ##############################
    conditions = [
        (data_master['Nuclear_TWh2000'] == 0) & (data_master['Nuclear_TWh2021'] == 0),
        (data_master['Nuclear_ppmix2000_2021'].isnull()),
        (data_master['Nuclear_ppmix2000_2021'] <= -15),
        (data_master['Nuclear_ppmix2000_2021'] > -15) & (data_master['Nuclear_ppmix2000_2021'] <= -5),
        (data_master['Nuclear_ppmix2000_2021'] > -5) & (data_master['Nuclear_ppmix2000_2021'] <= -2),
        (data_master['Nuclear_ppmix2000_2021'] > -2) & (data_master['Nuclear_ppmix2000_2021'] < 2),
        (data_master['Nuclear_ppmix2000_2021'] >= 2) & (data_master['Nuclear_ppmix2000_2021'] < 5),
        (data_master['Nuclear_ppmix2000_2021'] >= 5) & (data_master['Nuclear_ppmix2000_2021'] < 15),
        (data_master['Nuclear_ppmix2000_2021'] >= 15)
    ]

    values = [
        'Non-Nuclear',
        'Data Unavailable',
        'Major Decrease (-15p or less)', 
        'Moderate Decrease (-5 to -15p)', 
        'Minor Decrease (-5 to -2p)', 
        'Limited Change (±2p)', 
        'Minor Increase (2 to 5p', 
        'Moderate Increase (5 to 15p)', 
        'Major Increase (15p or more)'
    ]

    data_master['nuclear_role_change'] = np.select(conditions, values)

    ##############################
    # Nuclear power current categorization
    ##############################
    conditions = [
        (data_master['Nuclear_%mix2021'] == 0) | (data_master['Nuclear_%mix2021'].isnull()),
        (data_master['Nuclear_%mix2021'] >= 30),
        (data_master['Nuclear_%mix2021'] < 30) & (data_master['Nuclear_%mix2021'] >= 10),
        (data_master['Nuclear_%mix2021'] <10) & (data_master['Nuclear_%mix2021'] > 0)
    ]

    values = [
        'Non-Nuclear',
        'Major Source (30+)',
        'Moderate Source (10-30)', 
        'Minor Source (<10)'
    ]

    data_master['nuclear_size_cat'] = np.select(conditions, values)
    
    return(data_master)