# -*- coding: utf-8 -*-
# +
import toolbox

import pandas as pd

import requests

import json

from ._preprocessing import before_after

import re

import os

from datetime import datetime, timedelta

import addons


# -

def import_data(filename_data, path="data/raw"):
    
    """
    Reads data of xlsx or sav type from and returns a toolbox dataframe with value and variable labels

    Parameters
    ----------
    filename_data: str
        Name of the file. Can be excel file or sav file.
        Excel file should have 3 sheets: first one with data,
        second one with variable laberls and third one with value labels.
        Sheets with variable and value labels can be empty.

    path: str, default 'data/'
        The path to the folder where file is located.

    Returns
    -------
    data: DataFrame
        Toolbox DataFrame
    """
    path_data = f"{path}/{filename_data}"
    if path_data.endswith(".sav"):
        importer = toolbox.DataFrame.from_sav
    elif path_data.endswith(".xlsx"):
        importer = toolbox.DataFrame.from_excel
    else:
        valid_extensions = ['sav', 'xlsx']
        ext = filename_data.split(".")[-1]
        raise ValueError(
            "The data could not be imported, because the file "
            f"`{filename_data}` has the file extension `{ext}`, "
            f"which is not supported. Choose one of {valid_extensions}."
        )
        
    data = importer(path_data)

    return data


def create_excel_file(dataset: pd.DataFrame, variable_labels: pd.DataFrame, value_labels: pd.DataFrame, filename: str) -> str:
    """Creates the excel file that is similar to an XS export.
    Args:
        dataset (pd.DataFrame): first tab, data
        variable_labels (pd.DataFrame): second tab, variable labels
        value_labels (pd.DataFrame): _third tab, value labels
        filename (str): the name to be used for the file
    Returns:
        str: the path where the file is placed
        
    """
    sheets = [dataset, variable_labels, value_labels]
    for i,sheet in enumerate(sheets):
        if type(sheet) is not pd.DataFrame:
            try:
                sheets[i] = pd.DataFrame(sheet)  # Update the element in the list directly
            except Exception as e:
                print(f"The variable '{sheet}' is neither a DataFrame or variable that can be converted into one.\nPlease check your data inputs.\n\n{e}")
    dataset, variable_labels, value_labels = sheets

    path = f"data/raw/{filename}"
    
    try:
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            # Data
            dataset.to_excel(writer, sheet_name='Data')
            # Variable labes
            variable_labels.to_excel(writer, sheet_name='Variabele labes', index=False, header=False)
            # Value labels
            workbook  = writer.book
            worksheet = workbook.add_worksheet('Value labels')
            prev_var = ""
            row = 0
            for data_row in value_labels.iterrows():
                if data_row[1]['SUBJECT_LABEL'] != prev_var:
                     worksheet.write(row, 0, data_row[1]['SUBJECT_LABEL'])
                     row+=1
                if not pd.isna(data_row[1]['CLOSED_ANSWER']):
                    worksheet.write(row, 2, data_row[1]['CLOSED_ANSWER'])
                if not pd.isna(data_row[1]['ANSWER_NR']):
                    worksheet.write(row, 1, data_row[1]['ANSWER_NR'])
                row +=1
                prev_var = data_row[1]['SUBJECT_LABEL']
            return filename
    except Exception as e:
        print(e)


def load_from_XS(MM, date_from, date_to):
    """Fetches data dynamically by splitting requests until successful."""
    
    
    url = None
    headers = None
    
    full_dataset = []
    variable_labels, value_labels = None, None
    
    # insert the year test here --> 
    queue = [(date_from, date_to)]
    
    if same_year(date_from, date_to):
        while queue:
            current_from, current_to = queue.pop(0)
            res = requests.get(f"{url}?query={MM}&datefrom={current_from}&dateto={current_to}", headers=headers)
        
            if res.status_code == 200:
                data = json.loads(res.text)
                full_dataset.extend(data['data']['dataset'])
                if variable_labels is None:
                    variable_labels = data['data']['variable_labels']
                    value_labels = data['data']['value_labels']
                print(f"Data loaded successfully for {MM} from {current_from} to {current_to}")
        
            elif res.status_code == 400:
                print(f"Payload too big for {current_from} to {current_to}, splitting timeframe...")
                first_half, second_half = split_timeframe(current_from, current_to)
                queue.append(first_half)
                queue.append(second_half)
            else:
                print(f"Error fetching data for {current_from} to {current_to}. Status code: {res.status_code}")
    else:
        print('Please only enter to and from dates in the same year. Exports over years cannot be created.')
        return None, None, None
    return full_dataset, variable_labels, value_labels


def same_year(date_from: str, date_to:str)-> bool:
    """
    Checks if years are the same for API calls.
    Returns bool.
    """
    date_format = "%d-%m-%Y"
    date1 = datetime.strptime(date_from, date_format)
    date2 = datetime.strptime(date_to, date_format)

    # Compare the years
    if date1.year == date2.year:
        return True
    else:
        return False



def split_timeframe(start_date: str, end_date: str):
    date_format = "%d-%m-%Y"
    
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    # Find the midpoint
    midpoint_1 = start + (end - start) / 2 # midpoint for first half
    midpoint_2 = midpoint_1 + timedelta(days=1)# midpoint for second half
    
    return (start.strftime(date_format), midpoint_1.strftime(date_format)), (midpoint_2.strftime(date_format), end.strftime(date_format))


def export_from_API(dataset, variable_labels, value_labels, name):
    if name is not None:
        try:    
            dataset = pd.DataFrame(dataset)
            variable_labels = pd.DataFrame(variable_labels)
            value_labels = pd.DataFrame(value_labels)
            filepath = 'exports/' + name   ######## ADJUST THIS ONE LATER AS NECESSARY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            with pd.ExcelWriter(filepath) as writer:
                dataset.to_excel(writer, sheet_name='dataset', index=False)
                variable_labels.to_excel(writer, sheet_name='variable_labels', index=False)
                value_labels.to_excel(writer, sheet_name='value_labels', index=False)
            print(f"\nDataFrames were successfully saved to {filepath}")
        
        except Exception as e:
            print(f'Error: Could not export dataset. Error: {e}')
    else:
        print(f'Error: Could not export dataset. Please ensure you are only exporting data from the API call.')


def drop_last_empty_cols(data):
    """
    Removes consecutive empty (NaN) columns from the end of a DataFrame.

    This function identifies and drops all columns at the end of a DataFrame 
    where every row contains `NaN`. The process stops at the first non-empty 
    column encountered when iterating backward from the last column.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame to be checked for empty columns at the end.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the empty columns removed from the end, if any.

    Side Effects:
    -------------
    - Prints a message indicating how many columns were dropped, 
      along with the name of the last dropped column.
    - Prints a warning message advising the user to double-check 
      whether the columns should be dropped.
    - Prints a message if no empty columns are found at the end.

    Notes:
    ------
    - A column is considered empty if all its values are `NaN`.
    - The function only checks columns at the end of the DataFrame and stops 
      as soon as a non-empty column is encountered.
    - If no columns are empty at the end, the original DataFrame is returned 
      unmodified.
    """
        
    #checking if all the rows in the last col are empty to start dropping empty end cols

    if data.iloc[:, -1].isnull().sum() == data.shape[0]: 
        empty_cols_drop = [data.iloc[:, -1].name]
        for i in range(2, data.shape[1], 1): # looping through columns at the end to see where there is content again
            i =  -abs(i) # creating negative numbers to start checking all the empty columns at the end
            if data.iloc[:, i].isnull().sum() == data.shape[0]:
                empty_cols_drop.append(data.iloc[:, i].name)
                continue
            else:
                break #break out of the loop once we have no more empty columns at the end
        print(f'Last {len(empty_cols_drop)} columns get dropped until: "{empty_cols_drop[-1]}"\n⚠️ Warning: Make sure that all columns from "{empty_cols_drop[-1]}" should indeed be dropped!\n')
        
        return data.drop(empty_cols_drop, axis=1) #returning data without the empty cols at end
    
    # if the very last column isn't empty, the check will not be executed further
    else:
        print('No empty columns at the end of survey found! I guess this is not a fossil of a survey, cool beans :-)')
    
    return data # returning data not modified :)


def load_data(**kwargs): 
    """"   
    filename_data, MM, date_from, date_to, check_last_cols, drop_unqualified
    Loads data from eiter excel file or API call.
    """
    
    data = None  # Initialize data to ensure it's available even if loading fails
    
    # checking for last column check
    if kwargs.get('check_last_cols') is None:
        check_last_cols = False  # determines whether to try and drop the last columns, only applicable to data imported from XS
    elif type(kwargs.get('check_last_cols')) is bool:
        check_last_cols = kwargs.get('check_last_cols')
    else:
        print('Invalid input supplied for checking/dropping last columns.')
        
    # checking for dropping unqualified
    if kwargs.get('drop_unqualified') is None:
        drop_unqualified = False  # determines whether to try and drop the last columns, only applicable to data imported from XS
    elif type(kwargs.get('drop_unqualified')) is bool:
        drop_unqualified = kwargs.get('drop_unqualified')
    else:
        print('Invalid input supplied for dropping incompletes.')
    
    
    # getting data from excel file ##################################################################################################
    if kwargs.get('filename_data') is not None:
        filename_data = kwargs['filename_data']
        regex = r'MM\d{4}'
        try:
            MM = re.match(regex, filename_data).group()
        except:
            print(f'No MM code found in filename. Please ensure you MM codes are in the file name. Error: {e}')
        
        try:
            data = import_data(filename_data)
            print(f'{filename_data} got loaded\n')
            
        except Exception as e:
            print(f'Error: File {filename_data} could not be loaded. Ensure filename and path are correct. Error: {e}')

        # dropping empty columns at the end if the data is imported from XS instead of API call
        if check_last_cols == True:
            try:
                data = drop_last_empty_cols(data)
            except Exception as e:
                print(f'Error: Empty column check at the end of XS projects was not executed. Error: {e}')
        
        # only leave qualified respondents
        if drop_unqualified == True:
            data = before_after(data,func=lambda data: data.loc[data['tConditi'] == 1],title=f"Dropping unqualified respondents:")
            
        # lastly setting all the variables for the API export to None so it doesn't get run if the project is loaded via excel
        dataset = None
        variable_labels = None
        value_labels = None
        name = None

    # getting data from API call XS ##################################################################################################
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
                os.remove("data/raw/" + name) # getting rid of the excel file after we have imported the data
                print(f"\nQualified respondents (rows): {data.shape[0]} \nColumns: {data.shape[1]}") # move out into different function to display SUMMARY
            except Exception as e:
                print(f'Error: Project {MM} failed to load as a dataframe. Error: {e}')
        except Exception as e:
            print(f'Error: Project {MM} could not be loaded. Error: {e}')
    

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###  
    # if data was successfully loaded, return it
    if data is not None and MM is not None:
        return data, MM, dataset, variable_labels, value_labels, name
    # If no data was successfully loaded, exit the function
    else: 
        print("No data was successfully loaded. Exiting function.")
        return


# function to get a summary overview of the data that is currently loaded

def get_short_summary(data, MM):
    respondents = data.shape[0]
    columns = data.shape[1]
    
    #try:
    first_complete = addons.get_columns(data, starts_with=['xSTART', 'xtSTART'])
    #except Exception as e:
    #    first_complete = None
    #    print('Could find start or end time\n')



    print(f'Respondents: {respondents}\nColumns: {columns}\nFirst complete: {first_complete}')


    

    # get first completes and last complete from t start

