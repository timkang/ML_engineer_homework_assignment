import collections
import numpy as np
import pandas as pd
import copy as cp
import sys


ADOS_v1_vs_v2_mapping_df = pd.DataFrame([
    {'v1_key': 'ados1_a1', 'v2_key': 'ados1_a1', 'details_of_question_or_answer_changes': 'mapping of answer, plus minor clarifications, probably not a big deal',
        'v1_to_v2_answer_mapping': {'8': '4'}, 'v2_to_v1_answer_mapping': {'4': '8'}},
    {'v1_key': 'ados1_a2', 'v2_key': 'ados1_a2', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_a3', 'v2_key': 'ados1_a3', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados1_a4', 'v2_key': 'ados1_a4', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados1_a5', 'v2_key': 'ados1_a5', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados1_a6', 'v2_key': 'ados1_a6', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_a7', 'v2_key': 'ados1_a7', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_a8', 'v2_key': 'ados1_a8', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_b1', 'v2_key': 'ados1_b1', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    {'v1_key': 'ados1_b2', 'v2_key': 'ados1_b2', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    {'v1_key': 'ados1_b3', 'v2_key': 'ados1_b3', 'details_of_question_or_answer_changes': 'Answer 2 minor change in meaning'},
    {'v1_key': 'ados1_b4', 'v2_key': 'ados1_b4', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_b5', 'v2_key': 'ados1_b5', 'details_of_question_or_answer_changes': 'Splitting of answer plus minor clarifications',
        'v1_to_v2_answer_mapping': {'2': ['2','3']}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados1_b6', 'v2_key': 'ados1_b6', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_b7', 'v2_key': 'ados1_b7', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_b8', 'v2_key': 'ados1_b8', 'details_of_question_or_answer_changes': 'Minor change in meaning of response 1, minor clarifications'},
    {'v1_key': 'ados1_b9', 'v2_key': 'ados1_b9', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados1_b10', 'v2_key': 'ados1_b10', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados1_b11', 'v2_key': 'ados1_b11', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_b12', 'v2_key': 'ados1_b12', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    {'v1_key': None, 'v2_key': 'ados1_b13a'},
    {'v1_key': None, 'v2_key': 'ados1_b13b'},
    {'v1_key': None, 'v2_key': 'ados1_b14'},
    {'v1_key': None, 'v2_key': 'ados1_b15'},
    {'v1_key': None, 'v2_key': 'ados1_b16'},
    {'v1_key': 'ados1_c1', 'v2_key': 'ados1_c1', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    {'v1_key': 'ados1_c2', 'v2_key': 'ados1_c2', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    #### d1 <=> d4 same topics in v1 versus v2 except that in v2 weird reactions to sights/smells would go into d4, while in v1 it would
    ######## go into d1
    {'v1_key': 'ados1_d1', 'v2_key': 'ados1_d1', 'details_of_question_or_answer_changes': 'New answer code and significant clarifications',
        'v1_to_v2_answer_mapping': {'2': ['2','3']}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados1_d2', 'v2_key': 'ados1_d2', 'details_of_question_or_answer_changes': 'New answer code and significant clarifications',
        'v1_to_v2_answer_mapping': {'2': ['2','3']}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados1_d3', 'v2_key': 'ados1_d3', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados1_d4', 'v2_key': 'ados1_d4', 'details_of_question_or_answer_changes': 'Major clarification, taking material from what used to be d1'},
    {'v1_key': 'ados1_e1', 'v2_key': 'ados1_e1', 'details_of_question_or_answer_changes': 'Answer remapping and minor clarifications',
        'v1_to_v2_answer_mapping': {'1': ['1','2'], '2': '3'}, 'v2_to_v1_answer_mapping': {'2': '1', '3': '2'}},
    {'v1_key': 'ados1_e2', 'v2_key': 'ados1_e2', 'details_of_question_or_answer_changes': 'Answer remapping and minor clarifications',
        'v1_to_v2_answer_mapping': {'1': ['1','2'], '2': ['2','3']}, 'v2_to_v1_answer_mapping': {'2': ['1', '2'], '3': '2'} },
    {'v1_key': 'ados1_e3', 'v2_key': 'ados1_e3', 'details_of_question_or_answer_changes': 'Minor clarifications'},


    {'v1_key': 'ados2_a1', 'v2_key': 'ados2_a1', 'details_of_question_or_answer_changes': 'mapping of answer, plus minor clarifications, probably not a big deal',
        'v1_to_v2_answer_mapping': {'7': '3'}, 'v2_to_v1_answer_mapping': {'3': ['3', '7']}},
    {'v1_key': 'ados2_a2', 'v2_key': None},
    {'v1_key': 'ados2_a3', 'v2_key': 'ados2_a2', 'details_of_question_or_answer_changes': 'Trivial clarifications'},
    {'v1_key': 'ados2_a4', 'v2_key': 'ados2_a3', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados2_a5', 'v2_key': 'ados2_a4', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_a6', 'v2_key': 'ados2_a5', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_a7', 'v2_key': 'ados2_a6', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados2_a8', 'v2_key': 'ados2_a7', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_b1', 'v2_key': 'ados2_b1', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_b2', 'v2_key': 'ados2_b2', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_b3', 'v2_key': 'ados2_b3', 'details_of_question_or_answer_changes': 'Significant rephrasing of responses',
        'v1_to_v2_answer_mapping': {'2': ['2','3'], '8': None}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados2_b4', 'v2_key': 'ados2_b4', 'details_of_question_or_answer_changes': 'Significant clarifications, and minor tightening of criteria'},
    {'v1_key': 'ados2_b5', 'v2_key': 'ados2_b5', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados2_b6', 'v2_key': 'ados2_b6', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_b7', 'v2_key': 'ados2_b7', 'details_of_question_or_answer_changes': 'Medium rephrasing'},
    {'v1_key': 'ados2_b8', 'v2_key': 'ados2_b8', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': None, 'v2_key': 'ados2_b9a', },
    {'v1_key': None, 'v2_key': 'ados2_b9b', },
    {'v1_key': 'ados2_b9', 'v2_key': 'ados2_b10', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados2_b10', 'v2_key': 'ados2_b11', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_b11', 'v2_key': 'ados2_b12', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados2_c1', 'v2_key': 'ados2_c1', 'details_of_question_or_answer_changes': 'Minor clarifications'},
    {'v1_key': 'ados2_c2', 'v2_key': 'ados2_c2', 'details_of_question_or_answer_changes': ''},
    {'v1_key': 'ados2_d1', 'v2_key': 'ados2_d1', 'details_of_question_or_answer_changes': 'New answer code and significant clarifications',
        'v1_to_v2_answer_mapping': {'2': ['2','3']}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados2_d2', 'v2_key': 'ados2_d2', 'details_of_question_or_answer_changes': 'New answer code and significant clarifications',
        'v1_to_v2_answer_mapping': {'2': ['2','3']}, 'v2_to_v1_answer_mapping': {'3': '2'}},
    {'v1_key': 'ados2_d3', 'v2_key': 'ados2_d3', 'details_of_question_or_answer_changes': 'Medium clarifications'},
    {'v1_key': 'ados2_d4', 'v2_key': 'ados2_d4', 'details_of_question_or_answer_changes': 'Major clarification, taking material from what used to be d1'},
    {'v1_key': 'ados2_e1', 'v2_key': 'ados2_e1', 'details_of_question_or_answer_changes': 'Answer remapping and minor clarifications',
        'v1_to_v2_answer_mapping': {'1': ['1','2'], '2': '3'}, 'v2_to_v1_answer_mapping': {'2': '1', '3': '2'}},
    {'v1_key': 'ados2_e2', 'v2_key': 'ados2_e2', 'details_of_question_or_answer_changes': 'Answer remapping and minor clarifications',
        'v1_to_v2_answer_mapping': {'1': ['1','2'], '2': ['2','3']}, 'v2_to_v1_answer_mapping': {'2': ['1', '2'], '3': '2'} },
    {'v1_key': 'ados2_e3', 'v2_key': 'ados2_e3', 'details_of_question_or_answer_changes': 'Minor clarifications'},
])


def apply_response_transforms(row, response_transform_rules, transform_key, id_key):
    ''' Runs on a single row, intended to be part of a pandas apply function.
    Transforms responses where needed, but not question itself. response_transform_rules defines
    the transformation that should be applied for each value. If no  transform is defined for a given
    value then returns original value unchanged.

    response_transform_rules should be a dictionary of mappings like this: {'1': '2', '2': ['2', '3']}
    If the mapping is to a list rather than a single value then a deterministic choice is made
    based upon a modulus of the device id, specified by id_key '''

    #print 'convert ', transform_key
    #print 'initial value is ', row[transform_key]
    if row[transform_key] in response_transform_rules.keys():
        transform_instructions = response_transform_rules[row[transform_key]]
        #print 'transform_instructions: ', transform_instructions
        if isinstance(transform_instructions, (list, np.ndarray)):
            ### Get index of output to map to based on last digit in ID if ID is string, or based on % len(list) if int
            number_of_choices = len(transform_instructions)
#            print 'row: ', row
#            print 'id_key: ', id_key
#            print 'id: ', row[id_key]
#            sys.stdout.flush()
            try:   ### if ID is string type
                if row[id_key] == '':
                    last_digit_of_id = 'a'  #### just pik something
                else:
                    last_digit_of_id = row[id_key][-1]
                #print 'last_digit_of_id: ', last_digit_of_id
                assert isinstance(last_digit_of_id, (str, unicode))
                assert len(last_digit_of_id) == 1
                idx_chosen = ord(last_digit_of_id) % number_of_choices
            except:
                idx_chosen = row[id_key] % number_of_choices
#            print 'idx_chosen: ', idx_chosen, ' from ', transform_instructions
#            print row[transform_key], ' transforms to ', transform_instructions[idx_chosen]
            return transform_instructions[idx_chosen]
        else:
            return response_transform_rules[row[transform_key]]
    else:   ### No instructions for a transformation of this key. Return original.
        return row[transform_key]

def map_df_between_ADOS_versions(in_df, id_col, orig_version='v2', new_version='v1', key_suffix=''):
    ''' Convert between ADOS formats. Can go either v1 => v2, or v2 => v1.
    ... ID column gives user IDs of each row, required for deterministic mapping if multiple plausible answers exist
    ... Returns dataframe with converted columns
    ... Any extra/non ADOS rows are passed through without touching them
    ...... based on 'responses_differ' key in ADOS_v1_vs_v2_mapping_df

    Note: if you are not using standard naming and have a common suffix (such as '_thompson'),
    specify that in the key_suffix variable
    '''

    assert orig_version in ['v1', 'v2']
    assert new_version in ['v1', 'v2']
    conversion_df = cp.deepcopy(ADOS_v1_vs_v2_mapping_df)
    if key_suffix != '':
        conversion_df['v1_key'] = [None if ele is None else ele+key_suffix for ele in conversion_df['v1_key'].values]
        conversion_df['v2_key'] = [None if ele is None else ele+key_suffix for ele in conversion_df['v2_key'].values]

    orig_key = orig_version + '_key'
    new_key = new_version + '_key'
    out_df = cp.deepcopy(in_df)
    in_cols = cp.deepcopy(list(out_df.columns))
    out_cols = []
    cols_to_drop = []
    for in_col in in_cols:
        out_col = in_col   ### unless we replace it below
        responses_differ = False
        matching_row_idxs = conversion_df[conversion_df[orig_key]==in_col].index.tolist()
        if len(matching_row_idxs) == 0:
            print 'no match found for question ', in_col, '. Do not convert.'
        elif len(matching_row_idxs) == 1:
            idx = matching_row_idxs[0]
            conversion_dict = conversion_df.ix[idx].to_dict()
            matching_col = conversion_dict[new_key]
            if matching_col is None or pd.isnull(matching_col):
                print 'No corresponding column for ', in_col, ' exists. Drop column.'
                cols_to_drop.append(out_col)
            else:   ### ok to replace
                out_col = matching_col
                if in_col != out_col:
                    print 'Convert ', in_col, ' to ', out_col
            response_transform_key = None
            if orig_version=='v1' and new_version == 'v2': response_transform_key = 'v1_to_v2_answer_mapping'
            if orig_version=='v2' and new_version == 'v1': response_transform_key = 'v2_to_v1_answer_mapping'
            response_transform_rules = conversion_dict.get(response_transform_key, None)
            if response_transform_rules is not None and isinstance(response_transform_rules, dict):
                out_df[in_col] = out_df[[in_col, id_col]].apply(apply_response_transforms, args=(response_transform_rules,in_col, id_col), axis=1)
        else:
            print 'in_col: ', in_col, ', matching_row_idxs: ', matching_row_idxs
            print 'df: ', out_df
            print 'Error, could not understand matching for in_col: ', in_col
            raise ValueError
        out_cols.append(out_col)
    assert len(out_cols) == len(in_cols)
    out_df.columns = out_cols
    if cols_to_drop != []:
        out_df = out_df.drop(cols_to_drop, axis=1)
    return out_df



def map_between_ADOS_versions_old(in_df, use_strict=False, orig_version='v2', new_version='v1', key_suffix=''):
    ''' Convert between ADOS formats. Can go either v1 => v2, or v2 => v1.
    ... Returns dataframe with converted columns
    ... Any extra/non ADOS rows are passed through without touching them
    ... If use_strict is True then will drop any columns that are flagged as suspicious
    ...... based on 'responses_differ' key in ADOS_v1_vs_v2_mapping_df

    Note: if you are not using standard naming and have a common suffix (such as '_thompson'),
    specify that in the key_suffix variable
    '''

    assert orig_version in ['v1', 'v2']
    assert new_version in ['v1', 'v2']
    conversion_df = cp.deepcopy(ADOS_v1_vs_v2_mapping_df)
    if key_suffix != '':
        print 'v1 values: ', list(conversion_df['v1_key'].values)
        print 'v2 values: ', list(conversion_df['v2_key'].values)
        print 'key_suffix: ', key_suffix
        conversion_df['v1_key'] = [None if ele is None else ele+key_suffix for ele in conversion_df['v1_key'].values]
        conversion_df['v2_key'] = [None if ele is None else ele+key_suffix for ele in conversion_df['v2_key'].values]

    orig_key = orig_version + '_key'
    new_key = new_version + '_key'
    out_df = cp.deepcopy(in_df)
    in_cols = cp.deepcopy(list(out_df.columns))
    out_cols = []
    cols_to_drop = []
    for in_col in in_cols:
        out_col = in_col   ### unless we replace it below
        responses_differ = False
        matching_row_idxs = conversion_df[conversion_df[orig_key]==in_col].index.tolist()
        if len(matching_row_idxs) == 0:
            print 'no match found for question ', in_col, '. Do not convert.'
        elif len(matching_row_idxs) == 1:
            idx = matching_row_idxs[0]
            matching_col = conversion_df.iloc[idx][new_key]
            if matching_col is None or pd.isnull(matching_col):
                print 'No corresponding column for ', in_col, ' exists. Drop column.'
                cols_to_drop.append(out_col)
            else:   ### ok to replace
                out_col = matching_col
                responses_differ = conversion_df.iloc[idx]['responses_differ']
                if use_strict and responses_differ:
                    print 'running in mode use_strict and matching between ', in_col,\
                             ' and ', out_col, ' is suspicious. Drop column.'
                    cols_to_drop.append(out_col)
                else:
                    if in_col != out_col:
                        print 'Convert ', in_col, ' to ', out_col
        else:
            print 'in_col: ', in_col, ', matching_row_idxs: ', matching_row_idxs
            print 'df: ', out_df
            print 'Error, could not understand matching for in_col: ', in_col
            raise ValueError
        out_cols.append(out_col)
    assert len(out_cols) == len(in_cols)
    out_df.columns = out_cols
    if cols_to_drop != []:
        out_df = out_df.drop(cols_to_drop, axis=1)
    return out_df





valid_qs_and_as = {
    'ADOS1': collections.OrderedDict([
        ('ados1_a1', [0, 1, 2, 3, 8]),
        ('ados1_a2', [0, 1, 2, 3]),
        ('ados1_a3', [0, 1, 2, 8]),
        ('ados1_a4', [0, 1, 2, 3, 8]),
        ('ados1_a5', [0, 1, 2, 3, 8]),
        ('ados1_a6', [0, 1, 2, 8]),
        ('ados1_a7', [0, 1, 2, 3]),
        ('ados1_a8', [0, 1, 2, 8]),

        ('ados1_b1', [0, 2]),
        ('ados1_b2', [0, 1, 2, 3]),
        ('ados1_b3', [0, 1, 2]),
        ('ados1_b4', [0, 1, 2, 3]),
        ('ados1_b5', [0, 1, 2]),
        ('ados1_b6', [0, 1, 2, 3]),
        ('ados1_b7', [0, 1, 2, 3]),
        ('ados1_b8', [0, 1, 2]),
        ('ados1_b9', [0, 1, 2]),
        ('ados1_b10', [0, 1, 2]),
        ('ados1_b11', [0, 1, 2, 3]),
        ('ados1_b12', [0, 1, 2, 3]),

        ('ados1_c1', [0, 1, 2, 3]),
        ('ados1_c2', [0, 1, 2, 3]),

        ('ados1_d1', [0, 1, 2]),
        ('ados1_d2', [0, 1, 2]),
        ('ados1_d3', [0, 1, 2]),
        ('ados1_d4', [0, 1, 2, 3]),

        ('ados1_e1', [0, 1, 2, 7]),
        ('ados1_e2', [0, 1, 2]),
        ('ados1_e3', [0, 1, 2]),
        ]

    ),
    'ADOS2': collections.OrderedDict([
        ('ados2_a1', [0, 1, 2, 3, 7]),
        ('ados2_a2', [0, 1, 2, 7]),
        ('ados2_a3', [0, 1, 2, 7, 8]),
        ('ados2_a4', [0, 1, 2, 3]),
        ('ados2_a5', [0, 1, 2, 3]),
        ('ados2_a6', [0, 1, 2, 3]),
        ('ados2_a7', [0, 1, 2, 3]),
        ('ados2_a8', [0, 1, 2, 3, 8]),

        ('ados2_b1', [0, 2 ]),
        ('ados2_b2', [0, 1, 2 ]),
        ('ados2_b3', [0, 1, 2, 8 ]),
        ('ados2_b4', [0, 1, 2, 3 ]),
        ('ados2_b5', [0, 1, 2 ]),
        ('ados2_b6', [0, 1, 2 ]),
        ('ados2_b7', [0, 1, 2, 3 ]),
        ('ados2_b8', [0, 1, 2, 3 ]),
        ('ados2_b9', [0, 1, 2, 3 ]),
        ('ados2_b10', [0, 1, 2, 3 ]),
        ('ados2_b11', [0, 1, 2, 3 ]),

        ('ados2_c1', [0, 1, 2, 3 ]),
        ('ados2_c2', [0, 1, 2, 3 ]),

        ('ados2_d1', [0, 1, 2 ]),
        ('ados2_d2', [0, 1, 2 ]),
        ('ados2_d3', [0, 1, 2 ]),
        ('ados2_d4', [0, 1, 2, 3 ]),

        ('ados2_e1', [0, 1, 2, 7 ]),
        ('ados2_e2', [0, 1, 2 ]),
        ('ados2_e3', [0, 1, 2 ]),

        ]
    )
}

def sanity_check_responses(in_df, evaluator):
    ''' Inputs:
            in_df: dataframe of the data you want to sanity check
            evaluator: 'ADOS1' or 'ADOS2' (add in adirs later)
        methodology:
            for each key in in_df that is part of the evaluator, checks for
            presence or absence of expected responses
        returns:
            missing_responses_dict: an ordered dictionary of questions and valid answers
               that do not appear in the data
               .... Note that this may not be a problem if statistics are low and
               some responses simply were not given
            invalid_responses_dict: an ordered dictionary of questions and responses
               which were never defined (such as an evaluator giving a 3 when only 0, 1, and 2
               have a meaning
            questions_not_asked: a list of questions from the evaluator that are not in the data
            questions_that_may_be_typos: a list of keys in in_df that look like they may have been
               intended to be valid questions but which do not match
    '''

    def convert_response(response):
        ''' If response is in bad format, convert it '''
        try:
            out_response = int(response)
        except:
            out_response = response
        return out_response

    missing_responses_dict = collections.OrderedDict()
    invalid_responses_dict = collections.OrderedDict()
    expected_qs_and_as = valid_qs_and_as[evaluator]
    questions_that_may_be_typos = []
    for column in in_df.columns:
        if column not in expected_qs_and_as.keys():
            continue
        observed_responses = np.unique(in_df[column].values)
        observed_responses_converted = [convert_response(ele) for ele in observed_responses]
        invalid_responses_dict[column] = [ele for ele in observed_responses_converted if ele not in expected_qs_and_as[column]]
        missing_responses_dict[column] = [ele for ele in expected_qs_and_as[column] if ele not in observed_responses_converted]
        #print 'for column: ', column, ', check for responses'
        #print 'expected: ', expected_qs_and_as[column]
        #print 'observed: ', observed_responses
        #print 'missing: ', missing_responses_dict[column]
        #print 'invalid: ', invalid_responses_dict[column]


    questions_not_asked = [question for question in expected_qs_and_as.keys() if question not in in_df.columns]
    questions_that_may_be_typos = [col for col in in_df.columns if evaluator.lower() in col and col not in expected_qs_and_as.keys()]
    #print 'For our data found following results when checking for inconsistencies for evaluator, ', evaluator
    #print 'questions_not_asked: ', questions_not_asked
    #print 'valid responses that were not given: ', missing_responses_dict
    #print 'invalid responses that were given: ', invalid_responses_dict
    #print 'questions that may be typos: ', questions_that_may_be_typos

    return missing_responses_dict, invalid_responses_dict, questions_not_asked, questions_that_may_be_typos


def cross_check_ADOS_consistent_with_new_version(in_df):
    ''' Quick and dirty check to see whether in_df is more consistent with
    newer or older version

    Newer version has a number of responses that are not in the earlier version,
    and ados2_b12 is a new question. Check to see if these responses and this question are
    present

    Returns boolean true if consistent with new version
    '''

    seems_like_new_version = False
    #print 'columns: ', list(in_df.columns)
    if 'ados2_b12' in in_df.columns:
        print 'ados2_b12 is present in data columns. This is consistent with the newer version.'
        seems_like_new_version = True
    def convert_response(response):
        try:
            out_response = int(response)
        except:
            out_response = response
        return out_response

    question_responses_only_in_new_version = collections.OrderedDict([
            ('ados1_a1', 4),
            ('ados1_b5', 3),
            ('ados1_d1', 3),
            ('ados1_d2', 3),
            ('ados1_e1', 3),
            ('ados1_e2', 3),
            ('ados2_d1', 3),
            ('ados2_d2', 3),
            ('ados2_e3', 3),
        ])
    for question, response in question_responses_only_in_new_version.iteritems():
        value_counts = in_df[question].value_counts()
        if response in [convert_response(ele) for ele in list(value_counts.index)]:
            seems_like_new_version = True
    return seems_like_new_version
