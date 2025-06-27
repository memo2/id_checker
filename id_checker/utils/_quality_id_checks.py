from io import StringIO
import pandas as pd
import time
import openai
from openai import OpenAIError
from . import _preprocessing as pp
from ._keyboard_smash import is_mashing



# -*- coding: utf-8 -*-
def mark_oversample(data, sample_requested, check_column) -> 'Data':    
    # Counting the amount of IDs we got and seeing if that matches with the total amount of rows
    total = data['Panelcode2'].notnull().sum()
    if total != data.shape[0]:
        print('⚠️ Warning: total amount of Panelcode2 (users from Dynata) does not match with the total amount of respondents in the data set.')

    # Checking for oversample but subtracting bad quality from total first
    bad_quality = data[check_column].notnull().sum()
    total = total - bad_quality

    if sample_requested > total:
        print(f'Subtracting {bad_quality} bad quality IDs.\n{sample_requested}N requested and only {total}N good quality completes are in the data. Not subtracting any oversamples.')
    elif sample_requested == total:
        print('Amount of requested N matches with received good quality IDs. No oversampling occurred.')
    else:
        oversample = total - sample_requested
        print(f'Marking {oversample} oversamples.')

        # Get the unmarked rows in order
        unmarked = data[data[check_column].isnull()]

        # Mark the first `oversample` of them as 'oversample'
        to_mark = unmarked.iloc[:oversample].index
        data.loc[to_mark, check_column] = 'oversample'

    return data

# +


# finding all the start and end columns in order to determine the survey time later 
def get_start_end_columns(data):
    start_col = pp.get_columns(data, starts_with=['tstart','start', 'xtstart'])
    end_col = pp.get_columns(data, starts_with=['teind','eind', 'xteind'])
    if len(start_col) == 1 and len(end_col) ==1:
        return start_col[0], end_col[0]
    else:
        raise Exception('Multiple start or ending columns for time were found, cannot identify start and end times.')


# Function to preprocess start and end times, removing the date in front of the time
def split_off_time(col):
    try: 
        time = col.split(' ')[1]
    except:
        print('Could not split time.')
    return time

def split_off_date(col):
    try:
        date = col.split(' ')[0]
    except:
        print('Could not split date.')
    return date



# +
from datetime import datetime 


  
def total_in_secs(start, end):
    
    # getting start and end times
    start_time = datetime.strptime(start,"%d-%m-%Y %H:%M:%S")
    end_time = datetime.strptime(end, "%d-%m-%Y %H:%M:%S") 
    
    # calculating duration  to know total time spent on survey
    duration  = end_time - start_time
    duration_sec = duration.total_seconds() 
    
    return duration_sec

def total_in_min(seconds):
    difference_min = int(seconds/ 60) # converting to minute value
    return difference_min



# +
# using quartiles to get the boundaries for the majority of the values and determine outliers
def determine_outlier_thresholds_iqr(dataframe, col_name, upper = True , th1=0.25, th3=0.75, condition_col=None, condition_value=None):
     # Filter the dataframe if a condition column and value are provided
    if condition_col is not None and condition_value is not None:
        filtered_df = dataframe[dataframe[condition_col] == condition_value]
    else:
        filtered_df = dataframe

    # usually using the difference in seconds column
    quartile1 = filtered_df[col_name].quantile(th1)
    quartile3 = filtered_df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 2.5 * iqr # more sensitive towards extreme outliers with higher multiplier 2.5
    lower_limit = quartile1 - 0.50 * iqr # more sensitive towards slight outliers with lower multiplier, our data is usually very closely distributed
    
     # Guard clause: if spread is tight, skip lower outlier detection

    if iqr < 60 and not upper:  # 30 seconds threshold, adjust as needed
        print("IQR too small (less than 60 secs) — skipping lower outlier detection as spread is too tight.")
        return None if not upper else upper_limit

    return upper_limit if upper else lower_limit


###########################################



# mark outliers that are outliers in terms of taking exceptionally long to complete the survey
def mark_outlier(time_sec, threshold, is_upper=True):
    if threshold is None or time_sec is None:
        return False
    return time_sec >= threshold if is_upper else time_sec <= threshold




def string_to_dataframe(csv_string):
    """
    Convert a CSV string to a pandas DataFrame.
    
    Args:
        csv_string (str): CSV-formatted string
        
    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    # Convert string to file-like object
    data = StringIO(csv_string)
    
    # Read CSV into DataFrame
    df = pd.read_csv(data)
    
    # Replace empty strings with NaN, then convert 'X' to True and NaN to False
    for col in df.columns:
        if col != 'RESPID':  # Skip the ID column
            df[col] = df[col].map({'X': True}).fillna('')
    
    return df

# Example usage:
"""
issues_data = '''RESPID,wrong_language,offensive_language,random_answer,repeated_answer,keyboard_smash
135638919,,,X,,X
135657925,,X,,
135943449,,X,,,
136940034,,,X,,X
136944471,,,X,,
136959602,,,X,,X'''

df = string_to_dataframe(issues_data)
print(df)
""" 



def attention_check(data: 'DataFrame', end_time, start_time)-> 'attention_column':
    attention = pp.get_columns(data, starts_with=['Test', 'attention'], ends_with=['Attention', 'att', 'test'])[0]

    #removing trailing white spaces
    data[end_time] = data[end_time].str.strip()
    data[start_time] = data[start_time].str.strip()

    data['dayend_num'] = pd.to_datetime(data[end_time], format="%d-%m-%Y %H:%M:%S").dt.strftime('%w').astype(int)
    data['daystart_num'] = pd.to_datetime(data[start_time], format="%d-%m-%Y %H:%M:%S").dt.strftime('%w').astype(int)

    data['check_failed_attention'] = (data[attention] != data['dayend_num']) | (data[attention] != data['daystart_num'])



def straight_lining(data):
    """
    Checks for straight-liners in linkert scale questions with labels:
    'BV', 'DIFF', 'APP', 'WAARD' 
    """

    try:
        #bv_statements = addons.get_columns(data, starts_with=['BV', 'DIFF']) # worked before but looking for alternatives
        question_labels = ['BV', 'DIFF', 'APP', 'WAARD']

        #linkert_scales = [col for col in data.columns if question_labels in col]
        filtered_df = data.filter(regex='BV|DIFF|APP|WAARD')        
        blocks = []
        #print(filtered_df.columns)

        # Extracting all the blocks of statements as these are usually randomized and we only check per Brand
        for column in filtered_df.columns:
            prefix = column.split('_')[0]
            if prefix not in blocks:
                blocks.append(prefix)

        # Initialize the 'straight_liner' column as False
        data['check_straight_liner'] = False

        for block in blocks:
            # Get all statement columns starting with the block prefix
            statements = pp.get_columns(data, starts_with=[block])

            # Check for straightlining per row: all values across the row in these columns are the same
            is_straight_liner = data[statements].nunique(axis=1) == 1
            

            # Update the 'straight_liner' column to True where condition is met
            data.loc[is_straight_liner, 'check_straight_liner'] = True

        return data


    except:
        print('Straigtlining test failed.')
        data['check_straight_liner'] = None
        return data
    


    import pandas as pd
import math

def batch_df_fixed_rows(
    df: pd.DataFrame,
    language: str,
    question_text: str,
    batch_size: int = 25
):
    """
    Splits the DataFrame into batches of fixed row size (default 25).
    Returns a list of prompt strings and the corresponding RESPIDs per batch.
    """
    batch_strings = []
    batch_respids = []

    num_batches = math.ceil(len(df) / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_df = df.iloc[start:end]

        batch_str = batch_df.to_string(index=False)

        #prompt version 1.1
        prompt = f"""Task:
Review the provided dataset of survey responses and classify each response as invalid or valid according to the criteria below.

Survey Context

Language of the Question: {language}

Question(s) Asked: "{question_text}"

Instructions:

For each response, evaluate it and mark its corresponding RESPID in a CSV table according to one of the following categories:

- offensive_language: Contains explicit or offensive content, including content that spans across multiple characters or cells (e.g., 'a', 's', 's' = 'ass').
- repeated_answer: Simply repeats the wording of the original question without adding meaningful or original input.
- wrong_language: text in a language than the languate of the question specified. For instance the question could be in French and the answer in Dutch 'Ik spreek geen Frans'
- valid_open_answer: A genuine attempt to answer the question, even if it includes brand name misspellings or general brand mentions as long as it fits the context of the question.

Important Rules:

Mark only one column per response with an "X". If none of the categories apply, leave that row blank.
Be strict and critical when identifying responses, it can very well be that most of them are valid.
Use the context of the question to see if the answer given makes sense. If the question asks about which shoe brands they know and the answer is 'Facebook' it is not a valid answer.

Your output should be a CSV with no additional comments or explanation.

Output Format:
RESPID,offensive_language,repeated_answer,wrong_language, valid_open_answer
139116150,X,,,  
139116295,,,,  
139116462,,,X,  


Begin reviewing the following dataset:

{batch_str}


"""
        batch_strings.append(prompt)
        batch_respids.append(batch_df["RESPID"].tolist())

    return batch_strings, batch_respids



def send_batches_with_rate_limit(batch_prompts, client, delay_seconds=1, max_retries=3):
    """
    Sends each prompt in batch_prompts to the Azure OpenAI API,
    pausing between calls to respect rate limits and retrying on errors.

    Parameters:
        batch_prompts (list of str): List of prompt strings to send.
        delay_seconds (int): Time to wait between requests (default 1 second).
        max_retries (int): Max number of retry attempts if a request fails.

    Returns:
        responses (list): GPT responses for each batch (same order).
    """
    responses = []

    for i, prompt in enumerate(batch_prompts):
        retries = 0
        success = False

        while not success and retries <= max_retries:
            try:
                print(f"Sending batch {i+1}/{len(batch_prompts)}...")

                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a data quality checker and check marketing survey data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    model='AnalyticsTeam',
                )

                responses.append(response.choices[0].message.content)
                success = True
                time.sleep(delay_seconds)  # Pause to respect rate limit

            except openai.RateLimitError:
                print("Rate limit hit. Waiting 10 seconds before retrying...")
                time.sleep(10)
                retries += 1

            except OpenAIError as e:
                print(f"API error on batch {i+1}: {e}")
                retries += 1
                time.sleep(5)

        if not success:
            print(f"Failed to process batch {i+1} after {max_retries} retries.")
            responses.append("ERROR")

    return responses

#added new
def check_gibberish(brand_name, keys_over_one):
    brand_name = str(brand_name).lower().strip()
    if brand_name not in keys_over_one:
        return is_mashing(brand_name) #checking for keyboard smashes
    return False

