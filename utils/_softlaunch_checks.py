# -*- coding: utf-8 -*-
# +
import addons
from IPython.display import Markdown, display
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import requests
import os
import numpy as np

from ._import_data import import_data, create_excel_file, load_from_XS, export_from_API # Single dot means "same package"


from ._preprocessing import before_after




CREATIVE_PREFIXES = ["OV", "TV", "RADIO", "OD", "DISPLAY", 'SOCIAL','OOH', 'PRINT', 'BC', 'OLV', 'DOOH']

CREATIVE_SUFFIXES =  ["REC", "APP", "RATE", "EXP", "TRANSF", "INTRO", 'DIFF','INFO','LIKE', 'RELEVANT', "FIT", "ATRANSF",'BRAND','INVOLVE', 'MISAT']

CONSUMPTION_SUFFIXES = ["dagen", "dag", "days", "day", "uren", "hours", "hour", "minuten", "minutes", "minute"]

KPI_PREFIXES = ['SBA', 'ABA', "AAA", 'CONS', 'PURCH','PREF', 'BV1', 'APA', 'BEHOEFTE']

SOCIO_DEMO_PREFIX = ['geslacht', 'Regio', 'opleiding', 'Leeftijd', 'City', 'post', 'Transport', 'regio', 'beroep', 'gezin', 'inkomen']

META_DATA = ['xRESPID', 'DLNMID', 'tOPVOER', 'tFMAIL', 'tLMAIL','tCel', 'sDynata_1', 'sDynata_2', 'sDynata_3', 'sDynata_4','sDynata','Unnamed: 0', 'gebjaar2', 'xRESPID', 'ExternPanel', 'Panelcode', 'Panelcode2', 'SecurityKey', 'browser', 'mobiel', 'os', 'systeem', 'tablet', 'useragent', 'useragentt', 'xtSTART', 'xtEIND', 'tSTART', 'tEIND', 'tLanguag','tCountry','tConditi', 'tVOLGN', 'tVersie', 'Bron', 'tRespConditie']


# -
def get_info_kpi(data, kpis = None, suffix = None): 
    """
    Extracts and processes key performance indicator (KPI) columns from a given dataset.

    Args:
        data (pd.DataFrame): The input dataset containing various columns.
        kpis (list, optional): A list of KPI prefixes to filter relevant columns. 
                               Defaults to `KPI_PREFIXES` if not specified.
        suffix (str, optional): The suffix used to identify relevant columns. 
                                Defaults to "1" if not specified.

    Returns:
        None: The function attempts to process and display KPI-related information. 
              If no matching columns are found, it prints an error message.

    Behavior:
        - Identifies columns that match the given KPI prefixes.
        - Filters columns based on specific suffix conditions.
        - Includes columns that start with "BV", "bv", or "behoef".
        - Excludes columns ending in 't' unless they meet other criteria.
        - Calls `get_info_df()` with the filtered columns.
        - Prints an error message if no KPI columns are found.  
    """
    
    #setting up KPIs
    if kpis is None:
        kpis = KPI_PREFIXES
    elif type(kpis) is list:
        kpis = KPI_PREFIXES + kpis
        
    
    #setting up suffix, if not specified only the first response is taken as this is usally the client brand
    if suffix is None:
        suffix = "1"
    
    
    try: 
        cons_cols = addons.get_columns(data, starts_with=kpis)
        cols_to_check = []
    
    
        for col in cons_cols:
            
            # adding all columns ending with '_int' to the check
            if col.endswith('_' + suffix):
                cols_to_check.append(col)
        
            # adding all columns ending with 'int' to the check
            elif col.endswith(suffix):
                try:
                    int(col[-2]) # excluding columns that have two ints at the end such as 21 or 11
                except:
                    cols_to_check.append(col)
        
            # adding any columns that start with BV as we want all those to be included!
            elif col.startswith(('BV', 'bv', 'behoef')):
                cols_to_check.append(col) 
            
            # lastly adding any columns other unexpected columns that do not end with 't' --> denoting the open answer choice
            else:
                try:
                    int(col[-1])
                except:
                    if col.endswith('t'):
                        continue
                    else:
                        cols_to_check.append(col)
                        
        get_info_df(data, cols_to_check)
    
    except Exception as e:
        print(f'⚠️ Warning: No KPIs were found\n\nError: {e}')


def sort_types(data, 
               start_col = None, 
               end_col = None,
               creative_prefix = None, 
               creative_suffix = None, 
               consumption_suffix = None, 
               kpis = None,
               socio_demo_prefix = None,
               print_summary = None
            ):
    """
    Sorts columns in a DataFrame into predefined types based on prefixes and suffixes,
    and provides additional analysis for each column, including the count of non-null
    values, missing values, and occurrences of a specific value (97).

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the columns to be sorted and analyzed.
    start_col : str, optional
        The starting column name to include in the analysis. If not specified, all columns are checked.
    end_col : str, optional
        The ending column name to include in the analysis. If None, all columns 
        from the starting column onward are included.
    creative_prefix : list of str, optional
        List of prefixes identifying "Creative" type columns. Defaults to 
        `CREATIVE_PREFIXES` if not provided.
    creative_suffix : list of str, optional
        List of suffixes identifying "Creative" type columns. Defaults to 
        `CREATIVE_SUFFIXES` if not provided.
    consumption_suffix : list of str, optional
        List of suffixes identifying "Consumption" type columns. Defaults to 
        `CONSUMPTION_SUFFIXES` if not provided.
    kpis : list of str, optional
        List of prefixes identifying "KPI" type columns. Defaults to `KPI_PREFIXES`
        if not provided.
    socio_demo_prefix : list of str, optional
        List of prefixes identifying "SocioDemo" type columns. Defaults to 
        `SOCIO_DEMO_PREFIX` if not provided.
    print_summary: str, optional
        Need to supply a string of what summary you want to see. Possible options are: SocioDemo, KPI, Creative, Consumption

    Returns:
    -------
    pandas.DataFrame
        A summary DataFrame where each row corresponds to a column in the input 
        data, and includes the following information:
        - 'Column': The name of the column.
        - 'Type': The identified type (e.g., 'Creative', 'KPI', 'Consumption',
          'SocioDemo', or 'Uncategorized').
        - 'Answers': The number of non-null rows in the column.
        - 'Missings': The number of missing (null) rows in the column.
        - '97': The number of rows where the column has the value 99999997.

    Notes:
    -----
    - Prefixes and suffixes are used to classify columns based on their names.
    - The analysis adds context by highlighting columns with missing or specific
      data patterns.
    """
    
    #Setting up range of columns
    
    if start_col is None and end_col is None: #checking all column if not specified otherwise
            cols_to_check = data.loc[:,:]
            
    elif start_col is not None and end_col is None:
        try:
             cols_to_check = data.loc[:,start_col:]
        except Exception as e:
            print(e)
            
    elif start_col is None and end_col is not None:
        try:
            cols_to_check = data.loc[:,:end_col]
        except Exception as e:
            print(e)
    else: 
        try:
            cols_to_check = data.loc[:,start_col:end_col]
        except Exception as e:
             print(e)        
            
    
    
        
    #Setting up default prefixes and suffixes - if optionals are given they are appeneded to default ones
    if creative_prefix is None:
        creative_prefix = CREATIVE_PREFIXES
    elif type(creative_prefix) is list:
        creative_prefix = CREATIVE_PREFIXES + creative_prefix
       
    
    
    if creative_suffix is None:
        creative_suffix = CREATIVE_SUFFIXES
    elif type(creative_suffix) is list:
        creative_suffix = CREATIVE_SUFFIXES + creative_suffix
        
        
        
        
    if consumption_suffix is None:
        consumption_suffix = CONSUMPTION_SUFFIXES
    elif type(consumption_suffix) is list:
        consumption_suffix = CONSUMPTION_SUFFIXES + consumption_suffix
    
    if kpis is None:
        kpis = KPI_PREFIXES
    elif type(kpis) is list:
        kpis = KPI_PREFIXES + kpis
        
    if socio_demo_prefix is None:
        socio_demo_prefix = SOCIO_DEMO_PREFIX
    elif type(socio_demo_prefix) is list:
        socio_demo_prefix = SOCIO_DEMO_PREFIX + socio_demo_prefix
    
    #Checking if we want to print a specific summary if not defaulting to uncategorized
    if print_summary is None:
        print_summary = 'Uncategorized'
    
    #Grouping all prefixes and suffixes and giving them a type:
    
    prefix_dict = {
        'Creative': creative_prefix,
        'KPI': kpis,
        'SocioDemo': socio_demo_prefix,
        'Metadata': META_DATA
        }
    
    suffix_dict = {
        'Creative': creative_suffix,
        'Consumption': consumption_suffix
    }
        
        
    
    classifications = []

    for col in cols_to_check:
        col_type = None
        
        # Check prefixes
        for col_type_key, prefixes in prefix_dict.items():
            if any(col.startswith(prefix) for prefix in prefixes):
                col_type = col_type_key
                break
        
        # Check suffixes if no type identified
        if col_type is None:
            for col_type_key, suffixes in suffix_dict.items():
                if any(col.endswith(suffix) for suffix in suffixes):
                    col_type = col_type_key
                    break
         # Perform analysis
        answers = data[col].notnull().sum()
        missings = data[col].isnull().sum()
        value_97 = (data[col] == 99999997).sum()
        
        
        # Append classification
        classifications.append({
            'Column': col, 
            'Type': col_type or 'Uncategorized',
            'Answers': answers,
            'Missings': missings,
            '97': value_97})
    
    
    # Uncategorized
    data_cats = pd.DataFrame(classifications)
    summary_to_print = data_cats[data_cats['Type'] == print_summary]
    
    
    #adjust the summary you want to see below
    summary_you_want_to_see = summary_to_print  # <-- adjust here

    with pd.option_context('display.max_rows', None):
        print(summary_you_want_to_see)
    
    
    return data_cats


def empty_cols(data, start_col = None, end_col = None):
    
    """
    Identifies and prints all empty columns within a specified range in or the entire given DataFrame.
    
    This function helps detect potential issues by identifying columns that contain only missing values (NaN). 
    If an end column is not specified, it checks from the starting column to the last column of the DataFrame.
    
    Parameters
    ----------
    data : DataFrame
        The DataFrame to be analyzed.
    start_col : str, optional
        The name of the column to start checking from. If not provided all columns from the start will be checked.
    end_col : str, optional
        The name of the column to stop checking at. If not provided, all columns after the start column are checked.
    
    Returns
    -------
    None
        Prints the names of all empty columns in the specified range.
    """

    
    #Setting up range of columns
    
    if start_col is None and end_col is None: #checking all column if not specified otherwise
            cols_to_check = data.loc[:,:]
            
    elif start_col is not None and end_col is None:
        try:
             cols_to_check = data.loc[:,start_col:]
        except Exception as e:
            print(e)
            
    elif start_col is None and end_col is not None:
        try:
            cols_to_check = data.loc[:,:end_col]
        except Exception as e:
            print(e)
    else: 
        try:
            cols_to_check = data.loc[:,start_col:end_col]
        except Exception as e:
             rint(e)        
                
                
        
        
    empty_cols = cols_to_check.columns[cols_to_check.isnull().all()].tolist()
    
    empty_cols = [col for col in empty_cols if col not in META_DATA]
    
    # Print the empty columns
    display(Markdown(f'<span style="font-weight:bold"> Empty columns:\n </span>'))
    
    
    if start_col and end_col:
        display(Markdown(f'<span style="font-style:italic; color:darkgrey;"> checking columns between {start_col} and {end_col} \n </span>'))

    elif start_col:
        display(Markdown(f'<span style="font-style:italic; color:darkgrey;"> checking all columns from {start_col}  \n </span>'))
    
    elif end_col:
        display(Markdown(f'<span style="font-style:italic; color:darkgrey;"> checking all columns from start to {end_col}  \n </span>'))

    else:
        display(Markdown(f'<span style="font-style:italic; color:darkgrey;"> checking all columns  \n </span>'))
    
    if len(empty_cols) == 0:
        print("✅ No empty columns found!")
    else:
        print("\n".join(empty_cols))


def visualize_columns(data, cols = None):
    
    """
    This functions visualizes columns based on the answer labels and shows how data is distributed. 
    This is to check if we are targeting correctly.
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    col: str
        Should be the name of the age bracket column.
    """
    
    if cols is None:
        cols = ['geslacht', 'Regio', 'opleiding', 'Leeftijd', 'City', "ADblocker"]
    
    
    max_label_length = 12  # Maximum length of each label

    cons_cols = addons.get_columns(data, starts_with=cols)
    
    cons_cols = [col for col in cons_cols if 't' not in col[-1]]
    
    for col in cons_cols:
        
        try:
            labels = data.value_labels[col]
            keys = labels.keys()
            labels = labels.values()
            truncated_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label for label in labels]


            data[col] = pd.Categorical(data[col], categories= keys, ordered=True)


            plt.figure(figsize=(7, 3))
            sns.countplot(data=data, x=col, palette='viridis', legend= False, hue=col)


            # Set custom x-axis labels to use only the dictionary values (descriptive labels)
            plt.xticks(
                ticks=range(len(labels)),  # Ensure ticks cover all categories
                labels = truncated_labels,
                #labels=list(labels.values()),  # Use only the descriptive values for x-axis labels
                rotation = -30
            )
            plt.title(col)
            plt.ylabel('Count')
            
        except Exception as e:
            print("\n⚠️ Column failed to be visualized:", col, "⚠️")
            print("Error:", e)


def get_info(data, cols):
    
    """
    Displays summary information for specified columns in a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing survey or toolbox data.
        cols (list of str): A list of column names to retrieve and display information for.

    Behavior:
        - Iterates over the specified columns.
        - Displays the column name as a markdown header.
        - Calculates and displays the total number of answers (non-null values).
        - Calculates and displays the number of missing (null) values in red.
        - Handles exceptions when a column is not found in `value_counts()`.

    Notes:
        - Uses `display(Markdown())` for formatted output.
        - Ensure that the DataFrame and column names exist before calling the function.
    """
        
    for col in cols:
        display(Markdown(f'### {col}\n'))
        
        try: 
            total_answers = data[col].value_counts().sum()
        
            nulls = data[col].isnull().sum()
            
            
            display(Markdown(f'<span style="font-weight:bold">Total answers: {total_answers}\n </span>'))
            display(Markdown(f'<span style="color:red; font-weight:bold">Missings: {nulls}\n </span>'))
        except:
             print(f'{col} not in value_counts')


def get_info_df(data, cols):
    """
    Returns a DataFrame summarizing information about the specified columns 
    in the provided DataFrame.
    
    Information includes:
    - Total number of non-missing values (total_answers)
    - Number of missing values (nulls)

    Parameters
    ----------
    data : DataFrame
        Input DataFrame that contains the data.
    cols : list of str
        List of column names to get information about.
    
    Returns
    -------
    DataFrame
        A summary DataFrame with the following columns:
        - 'total_answers': Total number of non-missing values in each column.
        - 'nulls': Number of missing values in each column.
    """
    summary = []  # List to hold row-wise results
    
    for col in cols:
        try:
            total_answers = data[col].value_counts().sum()
            nulls = data[col].isnull().sum()
            summary.append({'column': col, 'total_answers': total_answers, 'nulls': nulls})
        except Exception as e:
            # Handle case where the column cannot be processed
            summary.append({'column': col, 'total_answers': None, 'nulls': None})
            print(f"Could not process column '{col}': {e}")
    
    # Convert summary to DataFrame
    result_df = pd.DataFrame(summary)
    print(result_df)
    return


def get_info_consumption(data, cols):
    
    """
    Prints out information on the Consumption columns, 
    indicating if the missings in Hours and Minutes match those in days. 
    This ensures that routing is corret.
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    cols: list of str
        List of column names to get info about.
    """
    summary = {}   
        
    days = ("dagen", "dag", "days", "day")
    hours = ("uren", "hours", "hour", 'Uren', 'Minuten', 'Hours', 'Minutes')
    minutes = ("minuten","minutes", "minute")
     
    
    # Dictionary to store matched day/hour/minute relationships
    consumption_dict = {}
    
    for col in cols:
        if col.endswith(days):
            
            # Extract the prefix by removing the suffix
            prefix = col
            for suffix in days:
                if col.endswith(suffix):
                    prefix = col[:-len(suffix)]
                    break

            if prefix not in consumption_dict:
                consumption_dict[prefix] = {"Days": None, "Hours": None, "Minutes": None}
            
            
            answers = data[col].notnull().sum()
            dont_knows = data[col].value_counts().get(99999997, 0)
            nulls = data[col].isnull().sum()
            consumption_dict[prefix]["Days"] = [col,answers, dont_knows, nulls]
            
        # HOURS

        elif col.endswith(hours):
            
            # Extract the prefix by removing the suffix
            prefix = col
            for suffix in hours:
                if col.endswith(suffix):
                    prefix = col[:-len(suffix)]
                    break

            if prefix not in consumption_dict:
                consumption_dict[prefix] = {"Days": None, "Hours": None, "Minutes": None}
            
            answers = data[col].notnull().sum()
            dont_knows = data[col].value_counts().get(99999997, 0)
            nulls = data[col].isnull().sum()
            consumption_dict[prefix]["Hours"] = [col,answers, dont_knows, nulls]
            
            
        elif col.endswith(minutes):
            
            # Extract the prefix by removing the suffix
            prefix = col
            for suffix in minutes:
                if col.endswith(suffix):
                    prefix = col[:-len(suffix)]
                    break
            if prefix not in consumption_dict:
                consumption_dict[prefix] = {"Days": None, "Hours": None, "Minutes": None}
            
            answers = data[col].notnull().sum()
            dont_knows = data[col].value_counts().get(99999997, 0)
            nulls = data[col].isnull().sum()
            consumption_dict[prefix]["Minutes"] = [col, answers, dont_knows, nulls] 
            

    # Print warnings for missing expected columns
    for key, values in consumption_dict.items():
        if values['Days'] is not None and values['Hours'] is not None:
            if values['Days'][2] == values['Hours'][3]:
                print(f'{key :<10}✅ Non-consumption in days equal to missings in hours')
            else:
                print(f"{key:<10}⚠️ Non-consumption in days NOT EQUAL to missings in hours")
                
        
        if values['Days'] is not None and values['Minutes'] is not None:
            if values['Days'][2] == values['Minutes'][3]:
                print(f'{key:<10}✅ Non-consumption in days equal to missings in minutes\n')
            else:
                print(f"{key:<10}⚠️ Non-consumption in days NOT EQUAL to missings in minutes\n")
        
        #print(f"{values[0]} {values[1]} {values[2]}")
        missing = [k for k, v in values.items() if v is None]
        if missing:
            print(f"⚠️Inconsistency found: Missing columns for '{key}': {', '.join(missing)} ⚠️\n")


def check_consumption(data, channel = None, ends_with = None):
    """
    Prints out information about consumption columns.
    Consumptions columns - columns that start with name of the channel
    and end with "days"/"hours"/"minutes".
    Consumption columns usually indicate how many days a week
    and how many hours and minutes in those days respondents
    consume certain media channel.
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    channel: str or list of str
        The name of a media channel.
        Pass a list of possible spellings if
        not sure how columns are named in the survey.
        Example: 'FB' or ['FB', 'FACEBOOK']
    ends_with: str or list of str, default ["dagen", "dag", "days", "day", "uren", "hours", "hour", "minuten", "minutes", "minute"]
        Part of the consumption column names that comes after channel name.
    """
    
    # checkin for suffixes
    cons_cols = [] # setting up consumption columns to be checked
    if ends_with == None:
        ends_with = CONSUMPTION_SUFFIXES
    elif type(ends_with) is list:
        ends_with = CONSUMPTION_SUFFIXES + ends_with
        
    # checking for consumption channel PREFIXES if no prefixes are given -> check is soley done on suffixes
    if channel == None:
        try:
            cons_cols = addons.get_columns(data, ends_with=ends_with)
        except:
            pass # checking later if there are any cons_cols
        
    # if channel is given, checking based on prefixes and suffixes
    else:
        try:
            cons_cols = addons.get_columns(data=data, starts_with=channel, ends_with=ends_with)
            
        except:
            pass # checking later if there are any cons_cols
    
    if len(cons_cols)> 0:
        get_info_consumption(data, cons_cols)
    else:
        print('⚠️ Warning: No consumption columns found')
        return


# +
def get_info_creative(data, cols):
    """
    Prints out information (variable label, value labels, value counts and number of missings)
    about every column in provided list of columns.
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    cols: list of str
        List of column names to get info about.
    """
    
    # Find the column ending with 'REC'
    columns_with_REC = [col for col in cols if col.endswith('REC')]
    
    dont_knows =0 #setting REC don't knows to zero in case first steps fails as we substract them from total_answers
    if len(columns_with_REC) == 1:
        rec_column = columns_with_REC[0]
        dont_knows = data[rec_column].value_counts().get(99999997, 0)
        print(f'99999997 in REC: {dont_knows}\n')
        
        if cols.index(rec_column)!= 0: # if rec is not in the first position, move it to the first position
            rec_index = cols.index(rec_column)
            cols.pop(rec_index)
            cols.insert(0, rec_column) #making sure rec col is always first

    elif len(columns_with_REC) > 1:
        raise ValueError("❌ More than one column ending with 'REC' found. Please check your column naming.\n")
    else:
        print("⚠️ Warning: No REC column found. Continuing check based on dont know/null matches with the first column in block!\n")

    
    ##############################################
    
    missings_match = True
    faulty_columns = []
    counter = 0
    rec_nulls = 0
    rec_check = True
    missing_values = [] # using this to save all the cols that aren't in value_counts sheet
        
    for col in cols:
        try: 
            data[col].value_counts()
            
            #saving info about first column in block to compare to
            if counter == 0:
                rec_nulls = data[col].isnull().sum() 
                total_answers = data[col].count() - dont_knows
                counter +=1 
                
            #getting info from all other columns to match to first col
            else:
                
                # Matching all other ones to REC columns missings and 97s 
                nulls = data[col].isnull().sum()
                
                if rec_nulls == nulls:
                    continue
                    
                elif nulls == (rec_nulls + dont_knows):
                    missings_match = False
                    continue
                    
                else:
                    rec_check = False
                    missings_match = False
                    faulty_columns.append(col)
        
        except:
            missing_values.append(col) # if value count failed, then the column has only Nan values we display this back later
    
    #### RESULTS: #####################################
    
    if total_answers == 0:
        display(Markdown(f'<span style="color:red">Missings: Amount of answers this block: {total_answers}\n</span>'))  
    else:
        print(f'Amount of answers this block: {total_answers}\n')
    
    # Checking Missings
    if missings_match == True:
        print(f'✅ All questions in this block have the same amount of missings:{nulls}\n')
    
    elif rec_check == True:
        print(f'🆗 People missing in REC add up to the extra missings in other columns. \n')
    
    else:
        print(f'❌ Some questions in this block have different amount of missing answers. Please check columns: {", ".join(faulty_columns)}')
    
    
    # Returning any empty columns
    if len(missing_values) > 0:
        print(f'⚠️ Warning: The following column(s) is/are empty:{" ".join(missing_values)}')
        
        

# -

def check_creative(data, creative = None, suffixes= None):
    """
    Prints out information about creative blocks.
    Creative columns - columns that start with name of the creative "OV", "TV", "RADIO", "OD", "DISPLAY", 'SOCIAL','OOH', 'PRINT', 'BC', 'OLV'
    and end with "REC", "APP", "RATE", "EXP", "TRANSF", "INTRO", "FIT", "ATRANSF".
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    creative: str or list of str
        The name of a creative.
        Pass a list of possible spellings if
        not sure how columns are named in the survey.
    suffixes: str or list of str, default = ["OV", "TV", "RADIO", "OD", "DISPLAY", 'SOCIAL','OOH', 'PRINT']
        Part of the creative column names that comes after creative type.
    """
    
    #Setting up default values for creative and suffix when none are given
    if creative is None:
        creative = CREATIVE_PREFIXES
    elif type(creative) is list:
        creative = CREATIVE_PREFIXES + creative
    
    if suffixes is None:
        suffixes = CREATIVE_SUFFIXES
    elif type(suffixes) is list:
        suffixes = CREATIVE_SUFFIXES + suffixes
    
    
    # Fetching all columns
    cons_cols = []
    
    
    try: 
        cons_cols = addons.get_columns(data, starts_with=creative, ends_with=suffixes)
    
    except:
        print('⚠️ Warning: No creatives found')
        return
        
    
    
    # Checking for different columns with atypical creative names to make the user aware of
    additional_cols = addons.get_columns(data, ends_with=suffixes) # checking based on known suffixes
    unexpected_results = [item for item in additional_cols if item not in cons_cols]
    if len(unexpected_results) > 0:
        print(f'⚠️ Unexpected columns found with unknown creative name: {", ".join(unexpected_results)}\nPlease consider adding these for a complete check.\n')    
    

    # Recreating blocks based on matching creative_names
    grouped = defaultdict(list)
    for column in cons_cols:
        # Remove known suffixes if they appear at the end of the column name
        base_name = column
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]  # Strip the suffix
                break  # Stop checking once a match is found

        # Extract the creative type and number if present
        match = re.match(r"^([A-Z]+)(\d+)?$", base_name)
        if match:
            creative_type = match.group(1)  # Base creative type
            number = match.group(2) if match.group(2) else ""  # Number (optional)
            group_key = f"{creative_type}{number}"  # Form the group key
            grouped[group_key].append(column)
    grouped = dict(grouped)
    
    for block, columns in grouped.items():
        # Fetch all columns that start with the base block name
        all_cols = addons.get_columns(data, starts_with=block)

        # Add any new columns not already included in grouped[block]
        for col in all_cols:
            if col not in columns:
                grouped[block].append(col) 

    
    # Checking for different columns with atypical suffixes that might have been missed before to make the user aware of

    final_results = np.concatenate([*grouped.values()]).tolist() #creating a list of all the creatives now in blocks to catch any last outliers
    missed_cols = addons.get_columns(data, starts_with=creative) # checking based on known creative names
    unexpected_results = [item for item in missed_cols if item not in final_results]
    if len(unexpected_results) > 0:
        print(f'⚠️ Unexpected columns found with unknown creative name: {", ".join(unexpected_results)}\n\nPlease consider adding creative names and suffixes for a complete check.\n')    

    # Printing summary per block
    count = 0

    for block, columns in grouped.items():
        count +=1 
        display(Markdown(f'<span style="font-weight:bold">\n\nCreative Block {count}: {block}\n</span>'))
        

        print(f'Total of {len(columns)} columns in {block}: {", ".join(columns)}\n')
        get_info_creative(data, columns)


def check_faulty_URL(data):
    """
    Checks for faulty URLs in the columns of a DataFrame.

    This function iterates over the columns of a given pandas DataFrame, extracts URLs 
    from the text content using a regex pattern, and validates the URLs by sending 
    HTTP GET requests. If a URL returns a non-200 status code or an error occurs 
    while accessing it, the URL is logged as faulty.

    Args:
        data (pandas.DataFrame): A DataFrame containing the data to check. Each column's 
            text content is scanned for URLs.

    Returns:
        None: The function prints a list of faulty URLs along with the column they 
        were found in. If no faulty URLs are found, it prints a message indicating this.
    """
    
    # Define regex for detecting URLs
    regex = r'\bhttps?://[^\s/$.?#].[^\s]*\b|\bwww\.[^\s/$.?#].[^\s]*\b'
    tested_links = []  # Saving faulty links

    # Iterate over all columns in the DataFrame
    for col in data.columns:
        column_content = data.variable_labels[col]
        if not isinstance(column_content, str):
            continue  # Skip if the content is not a string
        
        # Find all URLs in the column content
        urls = re.findall(regex, column_content)
        #print(f"URLs found in column '{col}': {urls}")
        
        if len(urls)>0:     
            for link in urls:
                try:
                    # Send a request to the URL
                    res = requests.get(link, timeout=5)  # 5-second timeout
                    if res.status_code != 200:  # If request fails, add to faulty links
                        tested_links.append((col, link, True))
                    else: 
                        tested_links.append((col, link, False))
                except Exception as e:
                    # Catch exceptions like connection errors or invalid URLs
                    tested_links.append((col, link, True))

            

    # Check and report faulty links
    if len(tested_links) > 0:
        for col, link, faulty in tested_links:
            if faulty is True:
                print(f"❌ Broken link: Column: {col}, Link: {link}\n")
                
            else:(f"✅ Working link: Column: {col}, Link: {link}\n")
        print('Please note: this check cannot find all links and is not a substitute for checking links in testing.')
    else:
        print("⚠️ Warning: No links found\nPlease note: this check cannot find all links and is not a substitute for checking links in testing.")


# +
# INITIAZE SCRIPT (master function to rule them all)

def script_init(**kwargs):
    """
    Initializes and executes a data processing pipeline based on provided parameters.

    This function imports data from either a file or an external source (XS), performs 
    initial cleaning (such as removing empty trailing columns), and runs a series of 
    data validation and transformation checks.

    Parameters:
    ----------
    **kwargs : dict
        Keyword arguments used to configure the script execution. The expected keys are:
        
        - filename_data (str, optional): Path to the data file to be loaded.
        - MM (str, optional): Project identifier for loading data from XS.
        - date_from (str, optional): Start date for the XS data import.
        - date_to (str, optional): End date for the XS data import.
        - start_col (int, optional): Starting column index for certain checks.
        - end_col (int, optional): Ending column index for certain checks.
        - creative_prefix (str, optional): Prefix for creative-related columns.
        - creative_suffix (str, optional): Suffix for creative-related columns.
        - consumption_suffix (str, optional): Suffix for consumption-related columns.
        - kpis (list, optional): List of KPI column names.
        - socio_demo_prefix (str, optional): Prefix for socio-demographic columns.
        - suffix (str, optional): Suffix for KPI-related processing.
        - creative (list, optional): List of creative column names.# MIGHT BE ABLE TO DELETE THIS AFTER TESTING
        # AS WE SWITCHED TO CREATIV SUFFIXES
        - suffixes (list, optional): Suffixes used in creative checks.# MIGHT BE ABLE TO DELETE THIS AFTER TESTING
        # AS WE SWITCHED TO CREATIV SUFFIXES
        - channel (str, optional): Channel column name for consumption checks.
        - ends_with (str, optional): Suffix used to filter consumption-related columns.
        - cols (list, optional): List of columns to be visualized.
        - api_export (Boolean, optional): let's you export data fetched via API to an excel file

    Returns:
    -------
    pandas.DataFrame or None
        - The processed DataFrame if data was successfully loaded and processed.
        - None if data could not be loaded.

    Side Effects:
    -------------
    - Prints status updates, warnings, and errors throughout execution.
    - Loads data from a specified file or an external XS source.
    - Drops trailing empty columns if applicable.
    - Runs multiple data validation and transformation functions.

    Processing Steps:
    -----------------
    1. Determines the data source:
       - Loads from a file (`filename_data`).
       - Loads from XS (`MM`, `date_from`, `date_to`).
    2. Removes empty trailing columns (if applicable).
    3. Runs a series of checks:
       - Detects empty columns.
       - Sorts data types.
       - Extracts KPI information.
       - Validates creative and consumption-related data.
       - Checks for faulty URLs.
       - Visualizes specified columns.

    Examples:
    ---------
    >>> script_init(filename_data="data.csv")
    data.csv got loaded
    Starting Softlaunch checks:
    ...
    
    >>> script_init(MM="project123", date_from="2024-01-01", date_to="2024-01-31")
    Project project123 got loaded
    Starting Softlaunch checks:
    ...
    
    Notes:
    ------
    - The function prints error messages if data loading or processing steps fail.
    - If `filename_data` is provided, the function assumes data is from an external source 
      that may require cleaning.
    - The function stops execution if no valid data source is provided.
    """
    data = None  # Initialize data to ensure it's available even if loading fails
    check_last_cols = False  # determines whether to try and drop the last columns, only applicable to data imported from XS

    if kwargs.get('filename_data') is not None:
        filename_data = kwargs['filename_data']
        check_last_cols = True # setting the last col check to true in order to drop all the empty columns at the end
        
        try:
            data = import_data(filename_data)
            print(f'{filename_data} got loaded\n')
            
            # only leave qualified respondents
            data = before_after(data,func=lambda data: data.loc[data['tConditi'] == 1],title=f"Dropping unqualified respondents:")
            # this will show you the amount of respondents
            print(f"{filename_data} \nRows(Amount of qualified respondents): {data.shape[0]} \nColumns: {data.shape[1]}")
            
        except Exception as e:
            print(f'Error: File {filename_data} could not be loaded. Ensure filename and path are correct. Error: {e}')
            
    elif all(key in kwargs for key in ("MM","date_from", 'date_to')):
        MM= kwargs['MM']
        date_from = kwargs['date_from']
        date_to = kwargs['date_to']
        name = f"{MM}_{date_from}_{date_to}"
        
        # only if we do an api call do we check for downloading the data
        if kwargs.get('api_export') is None: 
            api_export = False
        else:
            api_export = kwargs.get('api_export')
        
        # Fetching data
        try:
            dataset, variable_labels, value_labels = load_from_XS(MM, date_from, date_to)
            try:
                name = f"{MM}_export_{date_from}_{date_to}.xlsx"
                data = import_data(create_excel_file(dataset, variable_labels, value_labels, name))
                os.remove("data/raw/" + name)
                print(f"\nQualified respondents (rows): {data.shape[0]} \nColumns: {data.shape[1]}")
            except Exception as e:
                print(f'Error: Project {MM} failed to load as a dataframe. Error: {e}')
        except Exception as e:
            print(f'Error: Project {MM} could not be loaded. Error: {e}')
            
        if api_export is True:
            try:
                export_from_API(dataset, variable_labels, value_labels, name)
            except e:
                print(f'Error: data from API call could not be exported. Error: {e}')
    else:
        print('No valid import format inserted. Import data from XS or set up parameters for API call.')
        
    

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###  
        
    if data is None:  # If no data was successfully loaded, exit the function
        print("No data was successfully loaded. Exiting function.")
        return
    
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###  
    ### Data loaded ###

    
    # dropping empty columns at the end if the data is imported from XS instead of API call
    if check_last_cols == True:
        try:
            data = drop_last_empty_cols(data)
        except Exception as e:
            print(f'Error: Empty column check at the end of XS projects was not executed. Error: {e}')
    
    checks = [
        (empty_cols, {"start_col": kwargs.get("start_col"), 
                            'end_col': kwargs.get('end_col')}),
        
        (sort_types, {"start_col": kwargs.get("start_col"), 
                            'end_col': kwargs.get('end_col'),
                            'creative_prefix': kwargs.get('creative_prefix'), 
                            'creative_suffix': kwargs.get('creative_suffix'),
                            'consumption_suffix': kwargs.get('consumption_suffix'),
                            'kpis': kwargs.get('kpis'),
                            'socio_demo_prefix': kwargs.get('socio_demo_prefix')}),
        (get_info_kpi, {"kpis": kwargs.get("kpis"),
                             "suffix": kwargs.get("suffix")}),
        (check_creative, {'creative': kwargs.get('creative_prefix'),
                                'suffixes': kwargs.get('creative_suffix')}),
        (check_consumption, {'channel':kwargs.get('çhannel'),
                                   'ends_with':kwargs.get('ends_with')}),
        (check_faulty_URL, {}),
        (visualize_columns, {'cols':kwargs.get('cols')})
    ]
    
    print("\nStarting Softlaunch checks:")
    for check, options in checks:
        check_name = check.__name__.replace("_", " ").title()
        print(f"\n\n____________{check_name}____________\n")
        try:
            check(data, **options)  # Call the function with its options
        except Exception as e:
            print(f"\nError while running {check.__name__}: {e}")
    
    return data
# +
# INITIAZE SCRIPT (master function to rule them all) --> IN THE WEB APP

def script_init_softlaunch(**kwargs):
    """
    Initializes and executes a data processing pipeline based on provided parameters.

    This function imports data from either a file or an external source (XS), performs 
    initial cleaning (such as removing empty trailing columns), and runs a series of 
    data validation and transformation checks.

    Parameters:
    ----------
    **kwargs : dict
        Keyword arguments used to configure the script execution. The expected keys are:
        
        - filename_data (str, optional): Path to the data file to be loaded.
        - MM (str, optional): Project identifier for loading data from XS.
        - date_from (str, optional): Start date for the XS data import.
        - date_to (str, optional): End date for the XS data import.
        - start_col (int, optional): Starting column index for certain checks.
        - end_col (int, optional): Ending column index for certain checks.
        - creative_prefix (str, optional): Prefix for creative-related columns.
        - creative_suffix (str, optional): Suffix for creative-related columns.
        - consumption_suffix (str, optional): Suffix for consumption-related columns.
        - kpis (list, optional): List of KPI column names.
        - socio_demo_prefix (str, optional): Prefix for socio-demographic columns.
        - suffix (str, optional): Suffix for KPI-related processing.
        - channel (str, optional): Channel column name for consumption checks.
        - ends_with (str, optional): Suffix used to filter consumption-related columns.
        - cols (list, optional): List of columns to be visualized.
        - api_export (Boolean, optional): let's you export data fetched via API to an excel file

    Returns:
    -------
    pandas.DataFrame or None
        - The processed DataFrame if data was successfully loaded and processed.
        - None if data could not be loaded.

    Side Effects:
    -------------
    - Prints status updates, warnings, and errors throughout execution.
    - Loads data from a specified file or an external XS source.
    - Drops trailing empty columns if applicable.
    - Runs multiple data validation and transformation functions.

    Processing Steps:
    -----------------
    1. Loads data
    2. Runs a series of checks:
       - Detects empty columns.
       - Sorts data types.
       - Extracts KPI information.
       - Validates creative and consumption-related data.
       - Checks for faulty URLs.
       - Visualizes specified columns.

    Examples:
    ---------
    >>> script_init(filename_data="data.csv")
    data.csv got loaded
    Starting Softlaunch checks:
    ...
    
    >>> script_init(MM="project123", date_from="2024-01-01", date_to="2024-01-31")
    Project project123 got loaded
    Starting Softlaunch checks:
    ...
    
    Notes:
    ------
    - The function prints error messages if data loading or processing steps fail.
    - If `filename_data` is provided, the function assumes data is from an external source 
      that may require cleaning.
    - The function stops execution if no valid data source is provided.
    """
   

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###  
    
    data = kwargs.get("data")
    # If no data was successfully loaded, exit the function
    if data is None:
        print("No data was successfully loaded. Exiting function.")
        return
   
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###  

    
    checks = [
        (empty_cols, {"start_col": kwargs.get("start_col"), 
                            'end_col': kwargs.get('end_col')}),
        
        (sort_types, {"start_col": kwargs.get("start_col"), 
                            'end_col': kwargs.get('end_col'),
                            'creative_prefix': kwargs.get('creative_prefix'), 
                            'creative_suffix': kwargs.get('creative_suffix'),
                            'consumption_suffix': kwargs.get('consumption_suffix'),
                            'kpis': kwargs.get('kpis'),
                            'socio_demo_prefix': kwargs.get('socio_demo_prefix')}),
        (get_info_kpi, {"kpis": kwargs.get("kpis"),
                             "suffix": kwargs.get("suffix")}),
        (check_creative, {'creative': kwargs.get('creative_prefix'),
                                'suffixes': kwargs.get('creative_suffix')}),
        (check_consumption, {'channel':kwargs.get('çhannel'),
                                   'ends_with':kwargs.get('ends_with')}),
        (check_faulty_URL, {}),
        (visualize_columns, {'cols':kwargs.get('cols')})
    ]
    
    print("\nStarting Softlaunch checks:")
    for check, options in checks:
        check_name = check.__name__.replace("_", " ").title()
        print(f"\n\n____________{check_name}____________\n")
        try:
            check(data, **options)  # Call the function with its options
        except Exception as e:
            print(f"\nError while running {check.__name__}: {e}")
    
    return data
