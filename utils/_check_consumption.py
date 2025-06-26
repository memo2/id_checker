# -*- coding: utf-8 -*-
import addons
from IPython.display import Markdown, display
from collections import defaultdict
import re


def get_info(data, cols):
    
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
    for col in cols:
        display(Markdown(f'## {col}\n'))

        try:
            print(data.variable_labels[col], '\n')
        except:
            print(f'{col} not in variable labels')
        try:
            print(data.value_labels[col], '\n')
        except:
            print(f'{col} not in value_labels')
        
        try: 
            print(data[col].value_counts(), '\n')
        
            nulls = data[col].isnull().sum()
        
            display(Markdown(f'<span style="color:red; font-weight:bold">Missings: {nulls}\n </span>'))
        except:
             print(f'{col} not in value_counts')


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
     
    for col in cols:
        if col.endswith(days):
            display(Markdown(f'## {col}\n'))
            try:
                print(data.variable_labels[col], '\n')
            except:
                print(f'{col} not in variable labels')
            try:
                print(data.value_labels[col], '\n')
            except:
                print(f'{col} not in value_labels')
            print(data[col].value_counts(), '\n')
            
        
            dont_knows = data[col].value_counts().get(99999997, 0)
            display(Markdown(f'<span style="color:red; font-weight:bold">99999997: {dont_knows}\n </span>'))

            nulls = data[col].isnull().sum()
            display(Markdown(f'<span style="color:red; font-weight:bold">Missings: {nulls}\n </span>'))

        # HOURS

        elif col.endswith(hours):
            display(Markdown(f'### {col}\n'))
            try:
                print(data.variable_labels[col], '\n')
            except:
                print(f'{col} not in variable labels')
            try:
                print(data.value_labels[col], '\n')
            except:
                print(f'{col} not in value_labels')
            print(data[col].value_counts(), '\n')
            
            nulls = data[col].isnull().sum() # resettiung nulls to match current col
            if dont_knows == nulls:
            
                display(Markdown(f'<span style="color:green; font-weight:bold">Missings: {nulls}\n </span>'))
                print("✅ Non-consumption in days equal to missings in hours ✅\n")            
            else:
                display(Markdown(f'<span style="color:red; font-weight:bold">Missings: {nulls}\n </span>'))
                print("⚠️ Non-consumption in days NOT EQUAL to missings in hours ⚠️\n")    
      

        # MINUTES
        
        elif col.endswith(minutes):
            display(Markdown(f'### {col}\n'))
            try:
                print(data.variable_labels[col], '\n')
            except:
                print(f'{col} not in variable labels')
            try:
                print(data.value_labels[col], '\n')
            except:
                print(f'{col} not in value_labels')
            print(data[col].value_counts(), '\n')
            
            nulls = data[col].isnull().sum() # resettiung nulls to match current col
            if dont_knows == nulls:
                display(Markdown(f'<span style="color:green; font-weight:bold">Missings: {nulls}\n </span>'))
                print("✅ Non-consumption in days equal to missings in minutes ✅\n")
                display(Markdown("<hr>")) 
            else:
                display(Markdown(f'<span style="color:red; font-weight:bold">Missings: {nulls}\n </span>'))
                print("⚠️ Non-consumption in days NOT EQUAL to missings in minutes⚠️\n")
                display(Markdown("<hr>"))  
        else:
            print('🚨Something is amiss with this column:🚨\n', col)


def check_consumption(data, channel, ends_with=["dagen", "dag", "days", "day", "uren", "hours", "hour", "minuten", "minutes", "minute"]):
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
    cons_cols = addons.get_columns(data, starts_with=channel, ends_with=ends_with)
    
    get_info_consumption(data, cons_cols)
