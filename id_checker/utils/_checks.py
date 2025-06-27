from . import _preprocessing as pp
import difflib
from . import _quality_id_checks as qic
import pandas as pd
import importlib.resources
import os

def speeding_check(data):
    try:
    # identifying columns of start and end times 
        start_time, end_time = qic.get_start_end_columns(data)

        data['duration_sec'] = data.apply(lambda x: qic.total_in_secs(x[start_time].strip(), x[end_time].strip()), axis=1)

        try:
            data['duration_min'] = data['duration_sec'].apply(qic.total_in_min)
        except:
            print('Could not calculate timeframes')


        # getting upper outliers and cleaning them 
        upper_limit = qic.determine_outlier_thresholds_iqr(data, 'duration_sec',upper = True)

            # create a new column type bool to show all the outliers not to be considered in the analysis
        data['upper_outlier'] = data.apply(lambda x: qic.mark_outlier(x['duration_sec'], upper_limit, is_upper = True), axis = 1)


        # Mark lower outliers (speeders) only if spread is sufficient
        lower_limit = qic.determine_outlier_thresholds_iqr(data, 'duration_sec', upper=False, condition_col='upper_outlier', condition_value=False)
        data['lower_outlier_raw'] = data['duration_sec'].apply(lambda x: qic.mark_outlier(x, lower_limit, is_upper=False)) if lower_limit else False

        # --- Cap Lower Outliers to 3% ---
        if lower_limit:
            speeders = data[data['lower_outlier_raw']]
            max_speeders = int(0.03 * len(data))  # 3% cap

            if len(speeders) > max_speeders:
                percentage_speeders = len(speeders) / len(data) *100

                print(f'Percentage of speeders is {percentage_speeders:.2f}%, which is over the limit of 3%. Marking only most extreme speeders.')
                speeders_sorted = speeders.sort_values('duration_sec')
                allowed_speeders = speeders_sorted.head(max_speeders)
                allowed_ids = set(allowed_speeders.index)
                data['lower_outlier'] = data.index.isin(allowed_ids)
            else:
                data['lower_outlier'] = data['lower_outlier_raw']
        else:
            data['lower_outlier'] = False  # No flagging if IQR is too small

        # Drop temporary column
        data.drop(columns=['lower_outlier_raw'], inplace=True)
        return data

    except Exception as e:
        print(f"Couldn't complete speeding-check.\n\nError: {e}")


def keyboard_smash(data, open_answer_cols = ['SBA1','CEP1']):
    try:
        # getting SBA or CEP columns
        SBA_COLS = pp.get_columns(data, starts_with = open_answer_cols)
        
        # counting how many mentions each brand name has
        sba_counter = dict()
        for sba_question in SBA_COLS[0:2]: #getting the first two columns identified to look for brand names
            for brand_name in data[sba_question]:
                brand_name = str(brand_name).lower().strip()
                if sba_counter.get(brand_name):
                    sba_counter[brand_name] +=1
                else:
                    sba_counter[brand_name]=1

        keys_over_one = [k for k, v in sba_counter.items() if v > 2] #if in sba 1 or sba 2 at least 3 mentions
        #print(f'Total unique entries: {len(sba_counter)}\nMultiple mention entries: {len(keys_over_one)}')

        for brand_name in sba_counter:
            if brand_name not in keys_over_one:
                matches = difflib.get_close_matches(brand_name, keys_over_one, n=1, cutoff=0.8)
                if matches:
                    print(f"Received: {brand_name}, did you mean: {matches[0]}?")
                    keys_over_one.append(brand_name) # add to the list of accepted entries


        data['valid_open_answer'] = (
        data[SBA_COLS[0]].fillna('').astype(str).str.lower().str.strip().isin(keys_over_one) |
        data[SBA_COLS[1]].fillna('').astype(str).str.lower().str.strip().isin(keys_over_one) |
        data[SBA_COLS[0]].fillna('').astype(str).str.strip() == '99999997'
        )

        data['cond_1'] = data[SBA_COLS[0]].fillna('').astype(str).str.lower().str.strip().isin(keys_over_one)
        data['cond_2'] = data[SBA_COLS[1]].fillna('').astype(str).str.lower().str.strip().isin(keys_over_one)
        #data['cond_3'] = data[SBA_COLS[0]].fillna('').astype(str).str.strip() == '99999997'


        data['valid_open_answer'] = data['cond_1'] | data['cond_2'] #| data['cond_3']
        data.drop(columns=['cond_1','cond_2'])# getting rid of condition columns

        # # if either their first or second entry is in th
        # data['valid_open_answer'] = (
        #     data[SBA_COLS[0]].fillna('').str.lower().str.strip().isin(keys_over_one) |
        #     data[SBA_COLS[1]].fillna('').str.lower().str.strip().isin(keys_over_one)
        #     )
        # #print(f"Data in valid open answer: {data[data['valid_open_answer'] == True].shape}")


        # data['valid_open_answer'] = (data[SBA_COLS[0]].astype(str).str.strip() == '99999997')
        # print(f"Data in valid open answer: {data[data['valid_open_answer'] == True].shape}")

        # filtering out every response that is seen as valid
        unchecked_data = data[data['valid_open_answer'] == False]

        # Apply row-wise
        data['gibberish'] = unchecked_data[SBA_COLS[0]].apply(lambda x: qic.check_gibberish(x, keys_over_one))
        return data

    except Exception as e:
        print(f"Couldn't complete gibberish/keyboard-smash-check.\n\nError: {e}")


def attention_check(data, end_time, start_time):
    try:
        qic.attention_check(data, end_time, start_time)

        attention = pp.get_columns(data, starts_with=['Test', 'attention'], ends_with=['Attention', 'att', 'test'])[0]

        return data
        # for viewing results
        #print(data['check_failed_attention'].value_counts())
        #data[['dayend_num','daystart_num',attention, 'check_failed_attention']].sort_values(by='check_failed_attention', axis=0, ascending=False).head()
    except Exception as e:
        print(f'Could not complete attention-check. For this check to work, please include the "TestAttention" question in your project.\nError: {e}')


def straightlining_check(data):
    try:
        data = qic.straight_lining(data)

        #to view results
        #print(data['check_straight_liner'].value_counts())
        return data

    except Exception as e:
        print(f'Could not complete straight-lining-check.\nError: {e}')


def badwords_check(data, file, open_answer_cols):
    # Step 1: Load bad words
    open_answer_q = pp.get_columns(data, starts_with=open_answer_cols)

    if not os.path.isabs(file):
        with importlib.resources.open_text('id_checker.data', file) as f:
            badwords = set(word.strip().lower() for word in f if word.strip())
    else:
        # absolute path: open normally
        with open(file, 'r') as f:
            badwords = set(word.strip().lower() for word in f if word.strip())

    # Step 2: Function to check if any word in an answer is a bad word
    def contains_badword(answer):
        if pd.isna(answer):
            return False
        words = [word.strip().lower() for word in str(answer).split()]
        return any(word in badwords for word in words)

    # Step 3: Loop through open-ended question columns and check for bad words
    data['badwords'] = data[open_answer_q].apply(
        lambda row: any(contains_badword(row[col]) for col in open_answer_q),
        axis=1
    )
    return data


def lazyanswer_check(data, file, open_answer_cols):
    # Step 1: Load lazy words
    open_answer_q = pp.get_columns(data, starts_with=open_answer_cols)

    if not os.path.isabs(file):
        with importlib.resources.open_text('id_checker.data', file) as f:
            lazy_answer = set(word.strip().lower() for word in f if word.strip())
    else:
        # absolute path: open normally
        with open(file, 'r') as f:
            lazy_answer = set(word.strip().lower() for word in f if word.strip())

    # Step 2: Function to check if any word in an answer is a lazy answer
    def contains_lazy_answer(answer):
        if pd.isna(answer):
            return False
        words = [word.strip().lower() for word in str(answer).split()]
        return any(word in lazy_answer for word in words)

    # Step 3: Loop through open-ended question columns and check for bad words
    data['lazy_answer'] = data[open_answer_q].apply(
        lambda row: any(contains_lazy_answer(row[col]) for col in open_answer_q),
        axis=1
    )


def apply_scoring_rules(df, scoring_dict):
    # Initialize the 'score' column with 0s
    df['score'] = 0

    for column, weight in scoring_dict.items():
        if column in df.columns:
            df['score'] += df[column].fillna(False).astype(bool) * weight

    return df


def id_check(
    data,
    file_bw='badwords.txt',
    file_la='lazy_answer.txt',
    open_answer_cols=['SBA', 'CEP'],
    scoring_rules=None,
    checks_to_run=None,
):
    """
    Perform a series of data quality checks on survey or response data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to run the checks on. This DataFrame will be modified in-place with new columns and scores.
    
    file_bw : str, optional
        Path to the file containing bad words (default is 'data/badwords.txt'). Used in the badwords_check.
    
    file_la : str, optional
        Path to the file containing lazy answer indicators (default is 'data/lazy_answer.txt'). Used in lazyanswer_check.
    
    open_answer_cols : list of str, optional
        List of column prefixes or names that represent open answer columns to be checked (default is ['SBA', 'CEP']).
    
    scoring_rules : dict, optional
        A dictionary mapping check names to their scoring weights. If None, defaults will be used.
    
    checks_to_run : dict, optional
        Dictionary specifying which checks to run. Keys are check names (str), values are booleans.
        By default, all checks are run. Example:
            {
                'speeding_check': True,
                'keyboard_smash': True,
                'attention_check': True,
                'straightlining_check': True,
                'badwords_check': True,
                'lazyanswer_check': True,
                'apply_scoring_rules': True,
            }

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional scoring columns and checks applied.

    Notes
    -----
    - The function modifies the input DataFrame in-place.
    - The checks include speeding, keyboard smash detection, attention check, straightlining, bad words, and lazy answers.
    - Scoring is applied based on the provided or default scoring rules.
    """
    if scoring_rules is None:
        scoring_rules = {
            'lower_outlier': 2,
            #'offensive_language': 2,
            'badwords': 2,
            #'wrong_language': 1.5,
            'gibberish': 1.5,
            'lazy_answer': 1,
            #'repeated_answer': 1,
            'check_straight_liner': 1,
            'check_failed_attention': 1,
            'valid_open_answer': -1,
        }

    # Default to run all checks
    if checks_to_run is None:
        checks_to_run = {
            'speeding_check': True,
            'keyboard_smash': True,
            'attention_check': True,
            'straightlining_check': True,
            'badwords_check': True,
            'lazyanswer_check': True,
            'apply_scoring_rules': True,
        }

    start_time, end_time = None, None
    if checks_to_run.get('attention_check') or checks_to_run.get('apply_scoring_rules'):
        start_time, end_time = qic.get_start_end_columns(data)

    if checks_to_run.get('speeding_check'):
        speeding_check(data)

    if checks_to_run.get('keyboard_smash'):
        keyboard_smash(data, open_answer_cols)

    if checks_to_run.get('attention_check'):
        attention_check(data, end_time, start_time)

    if checks_to_run.get('straightlining_check'):
        straightlining_check(data)

    if checks_to_run.get('badwords_check'):
        badwords_check(data, file_bw, open_answer_cols)

    if checks_to_run.get('lazyanswer_check'):
        lazyanswer_check(data, file_la, open_answer_cols)

    if checks_to_run.get('apply_scoring_rules'):
        apply_scoring_rules(data, scoring_rules)
        data['score'] = data['score'].round(2)  # rounding decimals to max two digits

    return data
