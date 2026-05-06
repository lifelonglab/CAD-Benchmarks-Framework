import pathlib

import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cadbench.paths import TABULAR_DATASETS_PATH, OUTPUT_PATH


def preprocess_nsl_kdd(output_path: pathlib.Path = OUTPUT_PATH / 'datasets'):
    output_path.mkdir(parents=True, exist_ok=True)

    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
               "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
               "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
               "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
               "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
               "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate",
               "dst_host_srv_rerror_rate", "class", "difficulty"]
    categorical_columns = ['protocol_type', 'service', 'flag']
    numerical_columns = [l for l in columns if l not in [*categorical_columns, 'class', 'difficulty']]

    train_raw_df = pd.read_csv(TABULAR_DATASETS_PATH / 'nsl-kdd' / 'KDDTrain+.txt', header=None)
    test_raw_df = pd.read_csv(TABULAR_DATASETS_PATH / 'nsl-kdd' / 'KDDTest+.txt', header=None)

    train_raw_df.columns = columns
    test_raw_df.columns = columns

    # Transform labels
    train_label = train_raw_df['class'].apply(lambda v: 0 if v == 'normal' else 1)
    test_label = test_raw_df['class'].apply(lambda v: 0 if v == 'normal' else 1)
    train_label = train_label.rename('label')
    test_label = test_label.rename('label')

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="error", sparse_output=False)

    train_encoded_cat_df = pd.DataFrame(
        encoder.fit_transform(train_raw_df[categorical_columns]),
        columns=encoder.get_feature_names_out().tolist(),
        index=train_raw_df.index
    )
    test_encoded_cat_df = pd.DataFrame(
        encoder.transform(test_raw_df[categorical_columns]),
        columns=encoder.get_feature_names_out().tolist(),
        index=test_raw_df.index
    )

    # Save unscaled version of the dataset (with encoded categorical features)
    train_unscaled_df = pd.concat([train_raw_df[numerical_columns], train_encoded_cat_df, train_label], axis=1)
    train_unscaled_df['concept_part'] = 'train'
    test_unscaled_df = pd.concat([test_raw_df[numerical_columns], test_encoded_cat_df, test_label], axis=1)
    test_unscaled_df['concept_part'] = 'test'
    full_unscaled_df = pd.concat([train_unscaled_df, test_unscaled_df], axis=0)
    full_unscaled_df.to_csv(output_path / 'nsl-kdd.csv', index=False)
    logging.info(
        f"Saved unscaled version of the dataset with encoded categorical features to {output_path / 'nsl-kdd.csv'}")

    # Scale numerical features
    scaler = StandardScaler()

    train_scaled_df = pd.DataFrame(
        scaler.fit_transform(train_raw_df[numerical_columns]),
        columns=scaler.get_feature_names_out().tolist(),
        index=train_raw_df.index
    )
    test_scaled_df = pd.DataFrame(
        scaler.transform(test_raw_df[numerical_columns]),
        columns=scaler.get_feature_names_out().tolist(),
        index=test_raw_df.index
    )

    # Save scaled version of the dataset
    train_df = pd.concat([train_scaled_df, train_encoded_cat_df, train_label], axis=1)
    train_df['concept_part'] = 'train'
    test_df = pd.concat([test_scaled_df, test_encoded_cat_df, test_label], axis=1)
    test_df['concept_part'] = 'test'

    full_df = pd.concat([train_df, test_df], axis=0)
    full_df.to_csv(output_path / 'nsl-kdd_scaled.csv', index=False)
    logging.info(f"Saved scaled version of the dataset to {output_path / 'nsl-kdd_scaled.csv'}")

    # Save the dataset as single-concept dataset
    full_df['concept_id'] = 0
    full_df['concept_name'] = 'nsl-kdd'
    full_df.to_csv(output_path / 'nsl-kdd_single_concept.csv', index=False)
    logging.info(f"Saved single-concept version of the dataset to {output_path / 'nsl-kdd_single_concept.csv'}")
