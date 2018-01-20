### This code is optionally used when running training optimization
### To get information about which features should be added to an
### existing list of features
###
### It handles both running of the tally method of feature selection as well as
### X-validation comparisons of performances when a single feature from
### The tally list has been added


# Many of these imports are unused and should be removed.
# Also, it is generally bad practice to do wildcard imports; only import what you need
import math, random
import copy as cp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_helper_functions import *
from sklearn.ensemble import RandomForestClassifier

##### Functions for studying new features when adding to triton dataset
##### When you add the new feature this will re-compute the aggregate features
def max_severity(data_row):
    return_value = 'missing'
    for data_element in data_row:
        try:
            value = int(data_element)
            if value<0 or value>4:
                continue
            if return_value == 'missing':
                return_value = value
            elif value>return_value:
                return_value = value
        except ValueError:
            continue
    return str(return_value)
def min_severity(data_row):
    return_value = 'missing'
    for data_element in data_row:
        try:
            value = int(data_element)
            if value<0 or value>4:
                continue
            if return_value == 'missing':
                return_value = value
            elif value<return_value:
                return_value = value
        except ValueError:
            continue
    return str(return_value)

def count_severity_level(data_row, severity_level):
    return_value = 0
    for data_element in data_row:
        if data_element==severity_level:
            return_value += 1
    return return_value


def incorporate_new_adir_features_to_aggregations(in_df, old_features, new_features):
    ''' new_features_list is a list of new adir features you might want to add to an existing
	set of features (old_features)

	This function updates the adir/triton aggregations appropriately '''

    out_df = cp.deepcopy(in_df)
    if new_features == ['']:
        return out_df
    all_base_features = old_features + new_features
    for feature in ['adir_max_severity', 'adir_min_severity', 'adir_severity_0_count', 'adir_severity_1_count',
                   'adir_severity_2_count', 'adir_severity_3_count', 'adir_severity_4_count']:
        if feature in all_base_features:
            all_base_features.remove(feature)
    out_df['adir_max_severity'] = out_df[all_base_features].apply(max_severity, axis=1)
    out_df['adir_min_severity'] = out_df[all_base_features].apply(max_severity, axis=1)
    out_df['adir_severity_0_count'] = out_df[all_base_features].apply(lambda row: count_severity_level(row, "0"), axis=1)
    out_df['adir_severity_1_count'] = out_df[all_base_features].apply(lambda row: count_severity_level(row, "1"), axis=1)
    out_df['adir_severity_2_count'] = out_df[all_base_features].apply(lambda row: count_severity_level(row, "2"), axis=1)
    out_df['adir_severity_3_count'] = out_df[all_base_features].apply(lambda row: count_severity_level(row, "3"), axis=1)
    out_df['adir_severity_4_count'] = out_df[all_base_features].apply(lambda row: count_severity_level(row, "4"), axis=1)
    return out_df


def run_new_feature_importance_tally(in_df, for_sure_features, new_candidate_features, number_of_tries,
    enforce_group_weight_instructions, number_of_features_to_keep, relative_weight_cutoff, model_parameters_dict):
    ''' run number_of_tries bootstrapping experiments to tally using both the original and the new candidate features,
	figure out which *of the new* features are the most important
    ... for_sure_features represents features that are used, but which you will ignore when tallying.
	... new_candidate_features represents features that might potentially be added. Tally-based feature
	selection will focus on figuring out which of these are best to add.
    ... model_parameters_dict should contain all the details of the model that you care about. For example, max_depth,
	....... n_trees, dunno_range, feature_encoding_map, outcome_column, ...
    Returns a dataframe with ranked features by importance, including the tally and importance metrics '''

    criterion = model_parameters_dict['criterion']
    max_features = model_parameters_dict['max_features']
    max_depth = model_parameters_dict['max_depth']
    n_estimators = model_parameters_dict['n_estimators']
    feature_encoding_map = model_parameters_dict['feature_encoding_map']
    outcome_column = model_parameters_dict['outcome_column']
    dunno_range = model_parameters_dict['dunno_range']
    class_weight = model_parameters_dict['class_weight']
    balancing_dimensions = model_parameters_dict['balancing_dimensions']

    feature_tally = {}
    all_features =  for_sure_features + new_candidate_features
    test_dataset = incorporate_new_adir_features_to_aggregations(in_df, for_sure_features, new_candidate_features)
    ### Make sure encoding is discrete and sanity check these are not already present:
    for feature in new_candidate_features:
        if feature not in feature_encoding_map.keys():
            feature_encoding_map[feature] = 'discrete'
    for i in range(0,number_of_tries):
        print 'try number ', i, ' out of ', number_of_tries

        dataset_for_this_try = subsample_per_class(test_dataset, outcome_column, {'autism':0.9, 'not':0.9} )

        sample_weights_for_this_try = balance_dataset_on_dimensions(dataset_for_this_try, ['age_years','outcome'],
                                        enforce_group_weight_instructions=enforce_group_weight_instructions, verbose=False)
        if model_parameters_dict.get('young_sample', False):
            for index in range(len(dataset_for_this_try)):
                if dataset_for_this_try['age_years'][index]==3:
                    sample_weights_for_this_try[index] *= 2
        model, encoded_features_output, y_predicted_without_dunno, y_predicted_with_dunno,\
            y_predicted_probs = all_data_model(dataset_for_this_try, all_features,
            feature_encoding_map, outcome_column, sample_weights_for_this_try, dunno_range, RandomForestClassifier,
            n_estimators = n_estimators, criterion = criterion, max_features = max_features, max_depth=max_depth,
            class_weight = class_weight)

        important_features = get_important_features(model, encoded_features_output, relative_weight_cutoff=relative_weight_cutoff)
        top_feature_columns = get_best_features(important_features, number_of_features_to_keep, '=', ['gender','age_months','age_years'])

        for feature in top_feature_columns:
            if feature not in new_candidate_features:
                continue    ### Already have this one
            if feature in feature_tally:
                feature_tally[feature]+=1
            else:
                feature_tally[feature]=1
    sorted_tally = sorted(feature_tally.items(), key=lambda pair: pair[1], reverse=True)
    print 'sorted_tally: ', sorted_tally

    ### Now do a final run will the best features to figure out which should be kept
    passing_tally_new_candidate_features = [ele[0] for ele in sorted_tally]
    filtered_passing_tally_new_candidate_features = [ele for ele in sorted_tally if ele[1]>0.2*number_of_tries]

    passing_tally_all_features = passing_tally_new_candidate_features[:number_of_features_to_keep] + for_sure_features
    passing_tally_dataset = incorporate_new_adir_features_to_aggregations(in_df, for_sure_features, passing_tally_new_candidate_features)
    sample_weights = balance_dataset_on_dimensions(passing_tally_dataset, balancing_dimensions,
                                    enforce_group_weight_instructions=enforce_group_weight_instructions, verbose=False)

    # If these are young kids then appy appropriate weighting
    if model_parameters_dict.get('young_sample', False):
        for index in range(len(in_df)):
            if in_df['age_years'][index]==3:
                sample_weights[index] *= 3
            if in_df['age_years'][index]==2:
                sample_weights[index] *= 2
    model, encoded_features_output, y_predicted_without_dunno, y_predicted_with_dunno,\
                y_predicted_probs = all_data_model(passing_tally_dataset, passing_tally_all_features,
                feature_encoding_map, outcome_column, sample_weights, dunno_range, RandomForestClassifier,
                n_estimators = n_estimators, criterion = criterion, max_features = max_features, max_depth=max_depth,
                class_weight = class_weight)

    passing_tally_important_features = get_important_features(model, encoded_features_output, relative_weight_cutoff=relative_weight_cutoff)
    new_passing_tally_features = get_best_features(passing_tally_important_features, number_of_features_to_keep, '=', for_sure_features)

    final_tally_results_dict = {'feature': [], 'tally': [], 'sum_encoded_importance': []}
    for feature in new_passing_tally_features:
        print 'Get final dict for feature: ', feature
        final_tally_results_dict['feature'].append(feature)
        final_tally_results_dict['tally'].append(feature_tally[feature])
        print 'final_tally_results_dict so far: ', final_tally_results_dict
        associated_encoded_features_and_importances = [ele for ele in passing_tally_important_features if feature in ele[0]]
        print 'associated_encoded_features_and_importances: ', associated_encoded_features_and_importances
        sum_importance = sum([ele[1] for ele in associated_encoded_features_and_importances])
        final_tally_results_dict['sum_encoded_importance'].append(sum_importance)
    print 'Final final_tally_results_dict: ', final_tally_results_dict
    final_tally_df = (pd.DataFrame(final_tally_results_dict)).sort_values('sum_encoded_importance', ascending=False)
    return final_tally_df


def bootstrap_cross_validate_new_features(data_df, new_features_to_try, for_sure_features, model_parameters_dict,
        enforce_group_weight_instructions, n_folds ):
    ''' This function would be run in a chain after the run_new_feature_importance_tally function has been run
    if it is desired to get an estimate of the out-of-sample error on the new features. In contrast to the run_new_feature_importance_tally
    function, it is set up to only consider a single new candidate feature to try at a time (and even so runs pretty slow)

    Inputs:
        data_df: full dataframe to be analyzed
        new_features_to_try: new features that have been selected as important by the run_new_feature_importance_tally function
        for_sure_features: features that are guaranteed to be present so will not be considered to add to the list
        model_parameters_dict should contain all the details of the model that you care about
    Output:
        dataframe of out of sample performances, where each row represents results when exactly one of the new features has been included.  '''


    def sampling_function_per_try(dataset):
        sample = subsample_per_class(dataset, outcome_column, {'autism':bootstrapping_sample_percent, 'not':bootstrapping_sample_percent})
        return sample

    criterion = model_parameters_dict['criterion']
    max_features = model_parameters_dict['max_features']
    max_depth = model_parameters_dict['max_depth']
    n_estimators = model_parameters_dict['n_estimators']
    feature_encoding_map = model_parameters_dict['feature_encoding_map']
    outcome_column = model_parameters_dict['outcome_column']
    dunno_range = model_parameters_dict['dunno_range']
    class_weight = model_parameters_dict['class_weight']
    balancing_dimensions = model_parameters_dict['balancing_dimensions']
    bootstrapping_number_of_tries = model_parameters_dict['bootstrapping_number_of_tries']
    bootstrapping_sample_percent = model_parameters_dict['bootstrapping_sample_percent']
    outcome_classes = model_parameters_dict['outcome_classes']
    outcome_class_priors = model_parameters_dict['outcome_class_priors']

    dict_of_feature_lists = {'feature': [], 'XValid_AUC': [], 'autism_recall': [], 'not_recall': []}
    for feature in ['']+list(new_features_to_try):
        if feature in for_sure_features:
			raise ValueError('Cannot add feature '+feature+' if it is already used')
        new_features = [feature]
        if feature == '':
            all_feature_columns_for_try = for_sure_features
        else:
            all_feature_columns_for_try = for_sure_features + new_features
        test_dataset = incorporate_new_adir_features_to_aggregations(data_df, for_sure_features, new_features)

        def ml_function_per_try(dataset_per_try):
            sample_weights_per_try = balance_dataset_on_dimensions(dataset_per_try, balancing_dimensions,
                enforce_group_weight_instructions=enforce_group_weight_instructions, )
            #AS A HACK, here we bump 3 years olds sample weights by a factor of 3 relative to the 4 year olds and 2 year olds
            if model_parameters_dict.get('young_sample', False):
                for index in range(len(dataset_per_try)):
                    if dataset_per_try['age_years'][index]==3:
                        sample_weights_per_try[index] *= 3
                    if dataset_per_try['age_years'][index]==2:
                        sample_weights_per_try[index] *= 2

            if model_parameters_dict.get('young_sample', False):
                metrics = cross_validate_model_with_addon(dataset_per_try, sample_weights_per_try, all_feature_columns_for_try,
                        feature_encoding_map, outcome_column, dunno_range, n_folds, outcome_classes, outcome_class_priors,
                        RandomForestClassifier, criterion = criterion, max_features = max_features,
                        max_depth=max_depth, n_estimators = n_estimators)
            else:
				metrics = cross_validate_model(dataset_per_try, sample_weights_per_try, all_feature_columns_for_try,
                        feature_encoding_map, outcome_column, dunno_range, n_folds, outcome_classes, outcome_class_priors,
                        RandomForestClassifier, criterion = criterion, max_features = max_features,
                        max_depth=max_depth, n_estimators = n_estimators)
            return metrics['overall_metrics']
        averaged_metrics, averaged_metrics_err =  bootstrap(test_dataset, bootstrapping_number_of_tries,
                            sampling_function_per_try, ml_function_per_try, return_errs=True, verbose=False)
        print ''
        print '~~~~~~~~~~~~~~'
        print 'Got metrics when adding this feature (blank means no new feature): ', feature, ':'
        print 'AUC: ', averaged_metrics['without_dunno']['auc'], ', autism recall: ',\
            averaged_metrics['without_dunno']['dataset_recall_per_class']['autism'], ', not recall: ',\
            averaged_metrics['without_dunno']['dataset_recall_per_class']['not']
        print '~~~~~~~~~~~~~~'
        print ''

        dict_of_feature_lists['feature'].append(feature)
        dict_of_feature_lists['XValid_AUC'].append(averaged_metrics['without_dunno']['auc'])
        dict_of_feature_lists['autism_recall'].append(averaged_metrics['without_dunno']['dataset_recall_per_class']['autism'])
        dict_of_feature_lists['not_recall'].append(averaged_metrics['without_dunno']['dataset_recall_per_class']['not'])
    df_of_feature_lists = (pd.DataFrame(dict_of_feature_lists)).sort_values('XValid_AUC', ascending=False)
    print 'df_of_feature_lists: ', df_of_feature_lists
    return df_of_feature_lists
