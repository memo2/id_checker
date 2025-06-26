from datetime import datetime
import addons
import pandas as pd
import numpy as np
from collections import OrderedDict
from pandas.api.types import is_numeric_dtype, is_integer_dtype

DEFAULT_STARTS_WITH = []
DEFAULT_ENDS_WITH = []
DEFAULT_EXCLUDED_ENDINGS = ["96", "6t", "97"]
DEFAULT_IGNORE_CASE = True
DEFAULT_VERBOSE = False

def get_columns(
        data,
        starts_with=DEFAULT_STARTS_WITH,
        ends_with=DEFAULT_ENDS_WITH,
        excluded_endings=DEFAULT_EXCLUDED_ENDINGS,
        ignore_case=DEFAULT_IGNORE_CASE,
        verbose=DEFAULT_VERBOSE,
):
    """
    Find columns based on format.

    Creates and returns a list containing the columns that start and/or
    end with a certain string. When using starts_with, excluding certain
    columns is possible based on default or provided endings.

    Parameters
    ----------
    data: toolbox.DataFrame
        The DataFrame whose columns will be searched

    starts_with: str or list of str, default=[]
        Columns that start with this string will be added to the list.
        At least starts_with and/or ends_with must be set.

    ends_with: str or list of str, default=[]
        Columns that end with this string will be added to the list
        At least starts_with and/or ends_with must be set.

    excluded_endings: str or list of str, optional, default=["96", "6t", "97"]
        Columns that end with these strings will not be added to the list.

    ignore_case: bool, default=True
        Case is ignored in the starts_with, ends_with and excluded_endings
        parameters and the columns of data.

    verbose: bool, default=True
        When True, the function will print extra statements containing
        additional information on the process of the function and the output.

    Returns
    -------
    columns: list
        List containing the found columns, based on starts_with, ends_with and
        excluded_endings.

    Examples
    --------
    >>> columns = get_columns(my_data, starts_with="column_name")
    >>> columns = get_columns(
        my_data,
        starts_with="column_name",
        excluded_endings=["ending_1", "ending_2"])
    """

    if not (starts_with or ends_with):
        raise TypeError(
            "The keyword argument 'starts_with' or 'ends_with' or "
            "both must be provided with a string or list.")

    if not isinstance(starts_with, list):
        if isinstance(starts_with, str):
            starts_with = [starts_with]
        else:
            raise TypeError(
                f"{starts_with} is {type(starts_with)}; must be str or list"
                )

    if not isinstance(ends_with, list):
        if isinstance(ends_with, str):
            ends_with = [ends_with]
        else:
            raise TypeError(
                f"{ends_with} is {type(ends_with)}; must be str or list"
                )

    if not isinstance(excluded_endings, list):
        if isinstance(excluded_endings, str):
            excluded_endings = [excluded_endings]
        else:
            raise TypeError(
                f"{excluded_endings} is {type(excluded_endings)}; "
                "must be str or list of str"
                )

    for value in starts_with:
        assert isinstance(value, str), \
            f"{value} is {type(value)}; must be str"
        assert len(value) > 0, \
            f"String '{value}' is of length {len(value)}; must be > 0"

    for value in ends_with:
        assert isinstance(value, str), \
            f"{value} is {type(value)}; must be str"
        assert len(value) > 0, \
            f"String '{value}' is of length {len(value)}; must be > 0"

    for value in excluded_endings:
        assert isinstance(value, str), \
            f"{value} is {type(value)}; must be str"
        assert len(value) > 0, \
            f"String '{value}' is of length {len(value)}; must be > 0"

    if not isinstance(ignore_case, bool):
        raise TypeError(f"{ignore_case} is {type(ignore_case)}; must be bool")

    if not isinstance(verbose, bool):
        raise TypeError(f"{verbose} is {type(verbose)}; must be bool")

    if verbose:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verboseprint = lambda *a, **k: None  # do-nothing function

    if ignore_case:
        col_names = list(zip(data.columns, [x.lower() for x in data.columns]))
        verboseprint(
            "Search terms: "
            f"starts_with={starts_with}; "
            f"ends_with={ends_with}; "
            f"excluded_endings={excluded_endings}; "
            f"ignore_case={ignore_case}"
        )
    else:
        col_names = list(zip(data.columns, data.columns))
        verboseprint(
            "Search terms: "
            f"starts_with={starts_with}; "
            f"ends_with={ends_with}; "
            f"excluded_endings={excluded_endings}; "
            f"ignore_case={ignore_case}"
        )

    if starts_with:
        starts_with = [x.lower() for x in starts_with]\
            if ignore_case else starts_with

        start_cols = []
        for start in starts_with:
            for col, check_col in col_names:
                if check_col.startswith(start):
                    start_cols.append(col)
    else:
        start_cols = data.columns

    if ends_with:
        ends_with = [x.lower() for x in ends_with]\
            if ignore_case else ends_with

        end_cols = []
        for end in ends_with:
            for col, check_col in col_names:
                if check_col.endswith(end):
                    end_cols.append(col)
    else:
        end_cols = data.columns

    if excluded_endings:
        excluded_endings = [x.lower() for x in excluded_endings]\
            if ignore_case else excluded_endings

        excluded_cols = []
        for excl in excluded_endings:

            for col, check_col in col_names:
                if check_col.endswith(excl):
                    excluded_cols.append(col)
    else:
        excluded_cols = []

    columns = [col for col in start_cols
               if col in end_cols and
               col not in excluded_cols]

    verboseprint('Columns found:', columns, "\n")

    if not columns:
        raise KeyError(
            "No columns were found using these search terms: "
            f"starts_with={starts_with}; "
            f"ends_with={ends_with}; "
            f"excluded_endings={excluded_endings}; "
            f"ignore_case={ignore_case}"
            )

    return columns

def _read_categories(categories):
    """
    Read user-friendly string input and formats it into an ordered dict.

    Parameters
    ----------
    categories : dict
        contains:
            Range: string where ranges are defined with : and single values
            by comma
            Category: Numerical value, either string or int

    Returns
    -------
    cat_dict : OrderedDict
        contains:
            Key: Range: tuple(start range, end range)
            Value: Category: int
    """
    cat_dict = OrderedDict()

    for interval, category in categories.items():
        if isinstance(category, str):
            category = eval(category)

        if interval is None:
            cat_dict[(np.nan, np.nan)] = category
            continue

        interval_str = str(interval)
        interval_list = interval_str.split(',')

        # Loop over the ranges belonging to a category and add to dictionary
        for interval in interval_list:
            interval_range = interval.split(':')
            if interval_range[0] == 'nan':
                cat_dict[(np.nan, np.nan)] = category
                continue

            start_range = eval(interval_range[0])
            if len(interval_range) > 1:
                end_range = eval(interval_range[1]) + 1
            else:
                end_range = start_range

            if not isinstance(start_range, int) or \
                not isinstance(end_range, int):
                raise TypeError(
                    'Interval ranges are of type {type(start_range)} and'
                    ' {type(end_range)}, should both be type int')
            cat_dict[(start_range, end_range)] = category

    is_inttype = all([isinstance(val, int) for val in cat_dict.values()])
    return cat_dict, is_inttype


def _categorize(column, categories, is_inttype=False):
    """
    Categorizes the values in the input column into the new category values.

    Parameters
    ----------
    column : pd.Series
    categories : OrderedDict
        containing:
            range: begin and end+1 in form of tuple
            category: int

    Returns
    -------
    cat_col : pd.Series
    """
    def get_category_value(x):
        for cat_range, cat_value in categories.items():
            start = cat_range[0]
            end = cat_range[1]

            if pd.isna(x) or np.isnan(x):
                if np.isnan(start) or pd.isna(start):
                    return cat_value
            elif (x >= start and x < end) or (x == start and start == end):
                return cat_value

        if is_inttype:
            return pd.NA
        else:
            return np.nan

    cat_col = column.apply(get_category_value)
    if is_inttype:
        cat_col = pd.Series(cat_col, dtype='Int64')
    else:
        cat_col = pd.Series(cat_col)

    return cat_col

def numerical_encode(column, categories):
    """
    Recodes numerical variable into categories.

    Takes a pandas Series as ``column`` and recodes the values into the
    categories defined in the categories dictionary.
    Function needs positive values.

    Parameters
    ----------
    column: pd.Series
        column to be recoded, containing numerical values
    categories: dict
        dictionary containing:
        - 'intervals': ranges that belong to category
            Note: Interval range are given in str format:
                '<start range> : <end range (INCLUDING)>, <single value>'
        - 'category' : the category values

    Returns
    -------
    column: pd.Series
        recoded column containing new category values

    Example
    -------
    Demonstration of parameter usage. Series is created of numerical values,
    called 'age'.

    >>> col = pd.Series([
    ...     10, 10.1, 19, 21, 65, 80, 45, 21, 67, 33, 20, 59, 2, 15, 99
    ...     ], name='age')

    Create interval dictionary as follows:

    >>> intervals = {
    ...     "0:19": 1, "20:29": 2, "30:121": 3}

    Call function to create new Series

    >>> numerical_encode(col, categories)
    """
    if not is_integer_dtype(column):
        try:
            column = column.astype(float)
        except:
            raise Exception('column contains non-numerical values')

    if not isinstance(column, pd.Series):
        raise TypeError('input column is not of type pd.Series')

    if not isinstance(categories, dict):
        raise TypeError('input categories is not of type dictionary')

    cat_dict, is_inttype = _read_categories(categories)

    # Use the range dictionary to assign the categories to input column
    return _categorize(column, cat_dict, is_inttype)


def before_after(data, func, title=None):
    """
    Prints out N before and after calling the func.
    
    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame
    func: function
        Function to apply to data.
        Example: lambda data: data.loc[data['age'] >= min_age]
    title: str
        Title to display when applying the function.
    
    Returns
    -------
    data: DataFrame
        Toolbox DataFrame
    """
    
    if title is not None:
        print(title)

    before = data.shape[0]
    print(f" - N (before):", before)
    
    data = func(data)
    
    after = data.shape[0]
    print(" - N (after):", after)
    print(" - N (diff):", before-after)
    print()

    return data


def preprocessing(data, min_age, max_age, sociodemos, drop_resps_not_from_country, numerical_encoding={},
                  merge_columns=[], open_answers_encoding=[]):
    
    """
    Cleans and prepares data for analysis.

    Parameters
    ----------
    data: DataFrame
        Toolbox DataFrame

    min_age: int
        Minimun age. Respondents of age below minimum will be dropped.
    max_age: int
        Maximum age. respondents fo age above maximum will be dropped.
    sociodemos: lis of str
        List of column names with sociodemographic variables to drop missings.
        Respondents with missings in sociodemographic variables will be dropped.
    drop_resps_not_from_country: bool
        If True, respondents not living in the country (99999997 in Regio)
        will be dropped.
    numerical_encoding: list of dict, default {}
        List of dictionaries with information about numericalvariables that
        need to be recoded into categories.
        Example: [
                    {
                        "src_col": "age",
                        "dest_col": "age_cat",
                        "categories": {
                            "18:34": 1,
                            "35:49": 2,
                            "50:99": 3
                        }
                    },
                    {
                        "src_col": "opleiding",
                        "dest_col": "opleiding_cat",
                        "categories": {
                            "1:3": 1,
                            "4:5": 2,
                            "6:7": 3
                        }
                    }
                ]
    merge_columns: list of list, default []
        List of lists of columns to be merged. 
        Missings in first column in a list will be filled with
        values from the second column.
        Example: [["geslacht", "MM000.geslacht"],
                    ["beroep", "MM000.beroep"]]
    open_answers_encoding: list of dict, default []
        List of dictionaries with information to encode open answers.
        Example: [{
                    "cols": addons.get_columns(data, starts_with="SBA1"),
                    "brands": BRANDS["generic"]
                }]
        
    Returns
    -------
    data: DataFrame
        Toolbox DataFrame
    """
    
    # Print out initial sample size before deleting any respondents.
    print(f"# Initial sample size of dataset: {len(data)}.\n")

    # Merge columns with each other.
    for dest_col, src_col in merge_columns:
        data[dest_col] = data[dest_col].combine_first(data[src_col])

    # Calculate `age` variable containing continous ages (e.g. 0-99).
    src_col = "gebjaar2"
    dest_col = "age"
    data[dest_col] = datetime.now().year - data[src_col]
    data.variable_labels[dest_col] = f"{dest_col} - created from {src_col}"

    # Recode numerical vars into categoricals.
    for encoding in numerical_encoding:
        src_col = encoding["src_col"]
        dest_col = encoding["dest_col"]
        categories = encoding["categories"]

        data[dest_col] = numerical_encode(data[src_col], categories).astype("Int64")
        data.variable_labels[dest_col] = f"{dest_col} - created from {src_col}"
        data.value_labels[dest_col] = {v: k for k, v in categories.items()}

    # Remove respondents falling outside of the age range.
    if min_age is not None:
        data = before_after(
            data,
            func=lambda data: data.loc[data['age'] >= min_age],
            title=f"# Dropping respondents whose age is < {min_age}:"
        )
    if max_age is not None:
        data = before_after(
            data,
            func=lambda data: data.loc[data['age'] <= max_age],
            title=f"# Dropping respondents whose age is > {max_age}:"
        )    

    # Remove respondents who do not live in the country.
    if drop_resps_not_from_country:
        data = before_after(
            data,
            func=lambda data: data.loc[data['Regio'] != 99999997],
            title="# Dropping respondents not living in the country:"
        )

    # Removing respondents with missings on sociodemos.
    for sociodemo in sociodemos:
        print(f"# Dropping respondents with missings in {sociodemo}:")
        data = before_after(
            data,
            func=lambda data: data.dropna(subset=[sociodemo])
        )

    print(f"# Final sample size of dataset: {len(data)}.\n")

    # Encode open answers into brands (i.e. SBA).
    for encoding in open_answers_encoding:
        cols = encoding["cols"]
        brands = encoding["brands"]

        data = addons.encode_open_answers(
            data, brands=brands, columns=cols
        )
        
    return data

