from typing import Any
import os 
import sys
import numpy as np
import pandas as pd
import random
import pickle
import torch
from torch.utils.data import Dataset
import re
from gimmemotifs.scanner import command_scan
import genomepy
genome = genomepy.Genome('hg38')

file_dir = os.path.dirname(__file__)

N_CPU_PER_PROCESS = 8
# raw_data_dir = os.path.join(file_dir, '..', '..','re_design', '10x_data')
# genome_file = os.path.join(raw_data_dir, 'refdata-gex-GRCh38-2020-A','fasta','genome.fa')
training_data_path =  os.path.join(file_dir, 'train_data')
scratch_motif_call_path = '/scratch/welchjd_root/welchjd1/dorzhey/motif_call'

# random.seed(10)
# np.random.seed(10)
# if dna_diff_path not in sys.path:
#     sys.path.append(dna_diff_path)

def load_TF_data_bidir(
    data_path: str,
    seqlen: int = 200,
    saved_file_name: str = "encode_data.pkl",
    limit_total_sequences: int = 0,
    to_save_file_name = "encode_data",
    load_saved_data: bool = False,
    train_cond: bool = True,
    run_name: str = '',
):
    """Load and preprocess transcription factor (TF) data for bidirectional analysis.

    This function loads TF data either from a saved pickle file or by processing a raw data file using
    `preprocess_TF_data_bidir`. It then constructs one-hot encoded DNA sequences and extracts motif information,
    organizing the data into training, testing, and shuffled sets along with associated cell type labels.

    Args:
        data_path (str): Path to the raw data CSV file.
        seqlen (int, optional): Desired sequence length. Defaults to 200.
        saved_file_name (str, optional): Name of the pickle file to load preprocessed data from. Defaults to "encode_data.pkl".
        limit_total_sequences (int, optional): Limit on the total number of sequences to process. If 0, all sequences are used.
        to_save_file_name (str, optional): Base name used when saving preprocessed data. Defaults to "encode_data".
        load_saved_data (bool, optional): If True, load preprocessed data from the pickle file; otherwise, preprocess the raw data. Defaults to False.
        train_cond (bool, optional): If True, filter training data to include only sequences with non-zero `cre_pos` or `cre_neg` values. Defaults to True.
        run_name (str, optional): String identifier for the current run, used to tag output files. Defaults to ''.

    Returns:
        dict: A dictionary containing:
            - "train_motifs": Motif information for training data.
            - "train_motifs_cell_specific": Cell-type-specific motif information for training data.
            - "test_motifs": Motif information for test data.
            - "test_motifs_cell_specific": Cell-type-specific motif information for test data.
            - "shuffle_motifs": Motif information for shuffled data.
            - "shuffle_motifs_cell_specific": Cell-type-specific motif information for shuffled data.
            - "train_sequences_cell_specific": A dictionary mapping TF clusters to lists of DNA sequences.
            - "bidir_cre_dict": Dictionary mapping TF clusters to combined `cre_pos` and `cre_neg` values.
            - "tf_cluster_state_dict": Dictionary mapping TF clusters to their state arrays.
            - "X_train": One-hot encoded training sequences.
            - "x_train_cell_type": Array of TF cluster labels for training data if `train_cond` is True; otherwise, None.
    """

    if load_saved_data:
        path = f"{training_data_path}/{saved_file_name}"
        with open(path, "rb") as file:
            encode_data = pickle.load(file)

    else:
        encode_data = preprocess_TF_data_bidir(
            data_file=data_path,
            seqlen=seqlen,
            subset=limit_total_sequences,
            save_name=to_save_file_name,
            save_output = True,
            run_name = run_name,
        )
    
    # Creating sequence dataset
    train_df = encode_data['train_df']
    
    if train_cond:
        train_df = train_df[(train_df['cre_pos']>0) | (train_df['cre_neg'] >0)]
        X_train = np.stack(train_df['cre_sequence'].apply(one_hot_encode_dna_sequence, length=seqlen).values)
    else:
        X_train = np.stack([one_hot_encode_dna_sequence(seq, length=seqlen) for seq in train_df['cre_sequence'].unique()])
    X_train[X_train == 0] = -1
    
    # Creating labels
    x_train_cell_type = None
    if train_cond:
        # tf_states = np.stack(train_df['tf_state'].to_numpy())
        # expr_pos = train_df['cre_pos'].to_numpy().astype(np.int8).reshape(-1, 1)
        # expr_neg = train_df['cre_neg'].to_numpy().astype(np.int8).reshape(-1, 1)
        # x_train_cell_type = np.hstack((tf_states, expr_pos, expr_neg))
        x_train_cell_type = train_df['tf_cluster'].to_numpy(dtype=np.int16)
    
    # Collecting variables into a dict
    return {
        "train_motifs": encode_data["train"]["motifs"],
        "train_motifs_cell_specific": encode_data["train"]["motifs_by_cell_type"],
        "test_motifs": encode_data["test"]["motifs"],
        "test_motifs_cell_specific": encode_data["test"]["motifs_by_cell_type"],
        "shuffle_motifs": encode_data["shuffled"]["motifs"],
        "shuffle_motifs_cell_specific": encode_data["shuffled"]["motifs_by_cell_type"],
        # "train_sequences": train_df['cre_sequence'].values.tolist(),
        "train_sequences_cell_specific": train_df.groupby('tf_cluster')['cre_sequence'].apply(list).to_dict(),
        'bidir_cre_dict' : encode_data['bidir_cre_dict'],
        'tf_cluster_state_dict':encode_data['tf_cluster_state_dict'],
        # 'cre_expression_bins' : encode_data['cre_expression_bins'],
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

def preprocess_TF_data_bidir(
        data_file,
        seqlen=200,
        subset=None,
        save_name = "encode_data",
        save_output = True,
        run_name = '',
    ):
    """Preprocess transcription factor (TF) data for bidirectional analysis.

    This function reads raw TF data from a CSV file, cleans and filters the data (e.g., removing sequences containing 'N'),
    applies padding and cutting of DNA sequences based on a specified sequence length, and constructs motif information for
    training, testing, and shuffled datasets. It also prepares auxiliary dictionaries for bidirectional CRE information and
    TF cluster states.

    Args:
        data_file (str): Path to the raw data CSV file.
        seqlen (int, optional): Desired sequence length for DNA sequences. Defaults to 200.
        subset (int, optional): If provided, a subset of the data is sampled with this many rows. Defaults to None.
        save_name (str, optional): Base name for saving the preprocessed data to a pickle file. Defaults to "encode_data".
        save_output (bool, optional): If True, saves the preprocessed data as a pickle file. Defaults to True.
        run_name (str, optional): Identifier string for the current run, appended to output file names. Defaults to ''.

    Returns:
        dict: A dictionary containing:
            - "train": Motif information for training data.
            - "test": Motif information for test data.
            - "shuffled": Motif information for shuffled data.
            - "train_df": DataFrame with selected columns from the training data.
            - "tf_cluster_state_dict": Dictionary mapping TF clusters to their state arrays.
            - "bidir_cre_dict": Dictionary mapping TF clusters to a 2-column array of `cre_pos` and `cre_neg` values.
    """

    data = pd.read_csv(data_file, index_col=0)

    # drop sequences with N
    data = data.drop(data[data['cre_sequence'].str.contains("N")].index).reset_index(drop=True)

    if subset is not None:
        # take subset rows of each cell type 
        data = data.sample(subset).reset_index(drop=True)

    data = data[data['tf_cluster'].isin([180, 348, 117, 208, 249, 121, 44, 424, 61, 245, 13, 106])]
    tf_cluster_state_dict = {x:string_to_array(y) for x,y in data[['tf_cluster','tf_state']].value_counts().index.to_list()}
    # yes in this order, coz otherwise raises TypeError: unhashable type: 'numpy.ndarray'
    data['tf_state'] = data['tf_state'].apply(string_to_array)
    # using SCAFE midpoint to cut the sequences:
    if data['cre_sequence'].str.len().min() <= seqlen:
        # CUTTING SEQUENCES FOR LAST seqlen BP
        data = pad_sequences_midpoint(data, seqlen)
    data = cut_sequences_midpoint(data, seqlen)
    # cut expression values of 10 chunks where 0 expression is a seperate bin, and take medians of those chunks
    # data['cre_expression_discrete'], cre_expression_bins = get_bin_medians(data)

    bidir_cre_dict = {
        tf_cluster: np.column_stack((group['cre_pos'].values, group['cre_neg'].values))
        for tf_cluster, group in data[(data['cre_pos']>0) | (data['cre_neg'] >0)].groupby('tf_cluster')
    }

    test_data = data[data['cre_chrom'] == "chr1"].reset_index(drop=True)

    shuffled_data = data[data['cre_chrom']=='chr2'].reset_index(drop=True)
    shuffled_data["cre_sequence"] = shuffled_data["cre_sequence"].apply(
        lambda x: "".join(random.sample(x, len(x)))
    )
    
    train_data = data[(data["cre_chrom"]!= "chr1") & (data["cre_chrom"] != "chr2")].reset_index(drop=True)

    # Getting motif information from the sequences
    train = construct_motifs(train_data, run_name+"train", bidir=True)
    test = construct_motifs(test_data, run_name+"test", bidir=True)
    shuffled = construct_motifs(shuffled_data, run_name+"shuffled", bidir=True)
    # label_ratio = (data['tf_cluster'].value_counts(normalize=True)).to_dict()
    columns_to_save = ['cre_sequence','tf_state','tf_cluster','cre_pos','cre_neg']
    combined_dict = {"train": train, "test": test, "shuffled": shuffled, 'train_df': train_data[columns_to_save],
                     'tf_cluster_state_dict':tf_cluster_state_dict,
                     'bidir_cre_dict' : bidir_cre_dict}

    # Writing to pickle
    if save_output:
        # Saving all train, test, shuffled dictionaries to pickle
        with open(f"{training_data_path}/{save_name}.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict


def load_TF_data(
    data_path: str,
    seqlen: int = 200,
    saved_file_name: str = "encode_data.pkl",
    limit_total_sequences: int = 0,
    to_save_file_name = "encode_data",
    load_saved_data: bool = False,
    run_name: str = '',
):
    """Load and preprocess transcription factor (TF) data.

    This function loads TF data either from a preprocessed pickle file or by processing a raw CSV file using
    `preprocess_TF_data`. It generates one-hot encoded DNA sequences and constructs motif information for training,
    testing, and shuffled datasets. Additionally, it prepares labels by combining TF state information with discretized
    CRE expression values.

    Args:
        data_path (str): Path to the raw data CSV file.
        seqlen (int, optional): Desired sequence length for DNA sequences. Defaults to 200.
        saved_file_name (str, optional): Name of the pickle file to load preprocessed data from. Defaults to "encode_data.pkl".
        limit_total_sequences (int, optional): Limit on the total number of sequences to process. If 0, all sequences are used.
        to_save_file_name (str, optional): Base name used when saving preprocessed data. Defaults to "encode_data".
        load_saved_data (bool, optional): If True, loads preprocessed data from the pickle file; otherwise, processes the raw data. Defaults to False.
        run_name (str, optional): Identifier string for the current run, used to tag output files. Defaults to ''.

    Returns:
        dict: A dictionary containing:
            - "train_motifs": Motif information for training data.
            - "train_motifs_cell_specific": Cell-type-specific motif information for training data.
            - "test_motifs": Motif information for test data.
            - "test_motifs_cell_specific": Cell-type-specific motif information for test data.
            - "shuffle_motifs": Motif information for shuffled data.
            - "shuffle_motifs_cell_specific": Cell-type-specific motif information for shuffled data.
            - "train_sequences": List of training DNA sequences.
            - "label_ratio": Dictionary with the ratio of labels.
            - "tf_cluster_state_dict": Dictionary mapping TF clusters to their state arrays.
            - "X_train": One-hot encoded training sequences.
            - "x_train_cell_type": Array combining TF state and discretized CRE expression values for training data.
    """

    if load_saved_data:
        path = f"{training_data_path}/{saved_file_name}"
        with open(path, "rb") as file:
            encode_data = pickle.load(file)

    else:
        encode_data = preprocess_TF_data(
            data_file=data_path,
            seqlen=seqlen,
            subset=limit_total_sequences,
            save_name=to_save_file_name,
            save_output = True,
            run_name = run_name,
        )
        
    # Splitting enocde data into train/test/shuffle
    # train_motifs = encode_data["train"]["motifs"]
    # train_motifs_cell_specific = encode_data["train"]["motifs_by_cell_type"]

    # test_motifs = encode_data["test"]["motifs"]
    # test_motifs_cell_specific = encode_data["test"]["motifs_by_cell_type"]

    # shuffle_motifs = encode_data["shuffled"]["motifs"]
    # shuffle_motifs_cell_specific = encode_data["shuffled"]["motifs_by_cell_type"]

    # Creating sequence dataset
    train_df = encode_data['train_df']
    X_train = np.stack(train_df['cre_sequence'].apply(one_hot_encode_dna_sequence, length=seqlen).values)
    X_train[X_train == 0] = -1
    

    
    # Creating labels
    # tf_states = np.stack(df['tf_state'].to_numpy())
    # exprs = df['cre_expression'].to_numpy().reshape(-1,1)
    # x_train_cell_type = np.hstack((tf_states, exprs))

    # Collecting variables into a dict
    return {
        "train_motifs": encode_data["train"]["motifs"],
        "train_motifs_cell_specific": encode_data["train"]["motifs_by_cell_type"],
        "test_motifs": encode_data["test"]["motifs"],
        "test_motifs_cell_specific": encode_data["test"]["motifs_by_cell_type"],
        "shuffle_motifs": encode_data["shuffled"]["motifs"],
        "shuffle_motifs_cell_specific": encode_data["shuffled"]["motifs_by_cell_type"],
        "train_sequences": train_df['cre_sequence'].values.tolist(),
        # "train_sequences_cell_specific": encode_data["train"]["df"].groupby(['tf_cluster', 'cre_expression_discrete'])['cre_sequence'].apply(list).to_dict(),
        'label_ratio' : encode_data['label_ratio'], 
        'tf_cluster_state_dict' : encode_data['tf_cluster_state_dict'], 
        # 'cre_expression_bins' : encode_data['cre_expression_bins'],
        "X_train": X_train,
        "x_train_cell_type": np.hstack((np.stack(train_df['tf_state'].to_numpy()), train_df['cre_expression_discrete'].to_numpy().astype(np.int8).reshape(-1, 1))),
    }


def preprocess_TF_data(
        data_file,
        seqlen=200,
        subset=None,
        save_name = "encode_data",
        save_output = True,
        run_name = '',
    ):
    """Preprocess transcription factor (TF) data.

    This function reads raw TF data from a CSV file, cleans and filters the data by removing sequences containing 'N',
    applies padding and midpoint cutting based on a specified sequence length, and creates discretized CRE expression labels.
    It partitions the data into training, test, and shuffled datasets based on chromosome information, and extracts motif
    information from the sequences.

    Args:
        data_file (str): Path to the raw data CSV file.
        seqlen (int, optional): Desired sequence length for DNA sequences. Defaults to 200.
        subset (int, optional): If provided, a subset of the data is sampled with this many rows. Defaults to None.
        save_name (str, optional): Base name for saving the preprocessed data to a pickle file. Defaults to "encode_data".
        save_output (bool, optional): If True, saves the preprocessed data as a pickle file. Defaults to True.
        run_name (str, optional): Identifier string for the current run, appended to output file names. Defaults to ''.

    Returns:
        dict: A dictionary containing:
            - "train": Motif information for training data.
            - "test": Motif information for test data.
            - "shuffled": Motif information for shuffled data.
            - "train_df": DataFrame with selected columns from the training data.
            - "label_ratio": Dictionary representing the ratio of label occurrences.
            - "tf_cluster_state_dict": Dictionary mapping TF clusters to their state arrays.
    """

    data = pd.read_csv(data_file, index_col=0)

    # drop sequences with N
    data = data.drop(data[data['cre_sequence'].str.contains("N")].index).reset_index(drop=True)

    # using SCAFE midpoint to cut the sequences:

    if subset is not None:
        # take subset rows of each cell type 
        data = data.sample(subset).reset_index(drop=True)


    # dict_cell_type_centroids = get_cell_type_centroids(data)
    # data['cell_type_centroids'] = data['cell_type_leiden'].map(dict_cell_type_centroids)
    
    # transform string of array into array
    
    tf_cluster_state_dict = {x:string_to_array(y) for x,y in data[['tf_cluster','tf_state']].value_counts().index.to_list()}
    # yes in this order, coz otherwise raises TypeError: unhashable type: 'numpy.ndarray'
    data['tf_state'] = data['tf_state'].apply(string_to_array)
    
    if data['cre_sequence'].str.len().min() <= seqlen:
        # CUTTING SEQUENCES FOR LAST seqlen BP
        data = pad_sequences_midpoint(data, seqlen)
    data = cut_sequences_midpoint(data, seqlen)
    # cut expression values of 10 chunks where 0 expression is a seperate bin, and take medians of those chunks
    # data['cre_expression_discrete'], cre_expression_bins = get_bin_medians(data)
    data['cre_expression_discrete'] = data['cre_expression'].apply(lambda x: 1 if x>0 else 0)

    test_data = data[data['cre_chrom'] == "chr1"].reset_index(drop=True)

    shuffled_data = data[data['cre_chrom']=='chr2'].reset_index(drop=True)
    shuffled_data["cre_sequence"] = shuffled_data["cre_sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )
    train_data = data[(data["cre_chrom"]!= "chr1") & (data["cre_chrom"] != "chr2")].reset_index(drop=True)

    # Getting motif information from the sequences
    # train = generate_motifs_and_fastas(train_data, "train", number_of_sequences_to_motif_creation)
    # test = generate_motifs_and_fastas(test_data, "test", number_of_sequences_to_motif_creation)
    # shuffled = generate_motifs_and_fastas(shuffled_data,"shuffled",number_of_sequences_to_motif_creation)

    train = construct_motifs(train_data, run_name+"train")
    test = construct_motifs(test_data, run_name+"test")
    shuffled = construct_motifs(shuffled_data, run_name+"shuffled")

    # pandas Series
    label_ratio = (data[['tf_cluster', 'cre_expression_discrete']].value_counts() / data.shape[0]).to_dict()
    columns_to_save = ['cre_sequence','tf_state','cre_expression_discrete','tf_cluster']
    combined_dict = {"train": train, "test": test, "shuffled": shuffled, 'train_df': train_data[columns_to_save],
                     'label_ratio' : label_ratio, 'tf_cluster_state_dict' : tf_cluster_state_dict} #, 'cre_expression_bins' : cre_expression_bins}

    # Writing to pickle
    if save_output:
        # Saving all train, test, shuffled dictionaries to pickle
        with open(f"{training_data_path}/{save_name}.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict


def construct_motifs(df, name, bidir=False):
    """Construct motif information from a given DataFrame of genomic sequences.

    This function generates a FASTA file from the provided DataFrame, runs a motif scanning tool on the FASTA file, and
    processes the resulting motif data. The motifs are aggregated both globally and by cell type or expression status,
    depending on the `bidir` flag.

    Args:
        df (pandas.DataFrame): DataFrame containing genomic sequences and associated metadata. Must include 'locus_id',
            'tf_cluster', and relevant expression columns (either 'cre_expression' or both 'cre_pos' and 'cre_neg').
        name (str): Base name used for the FASTA file and motif scan identifiers.
        bidir (bool, optional): If True, groups motifs by TF cluster using bidirectional CRE information; otherwise, groups
            by TF cluster and discretized CRE expression. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - "motifs": A DataFrame of aggregated motif counts across all sequences.
            - "motifs_by_cell_type": A dictionary where keys are group identifiers (e.g., TF clusters or TF cluster with
            expression status) and values are DataFrames of aggregated motif counts for that group.

    Raises:
        Exception: If the input DataFrame does not contain expected expression columns.
    """

    # Step 1: Prepare Data for FASTA and Motifs Extraction
    data = df.drop_duplicates('locus_id')
    fasta_path = f"{training_data_path}/{name}.fasta"
    with open(fasta_path, "w") as save_fasta_file:
        if "cre_expression" in df.columns:
            write_fasta_component = "\n".join(
                data[["locus_id", "tf_cluster", "cre_expression", "cre_sequence"]]
                .apply(lambda x: f">{x.locus_id}_leiden_{x.tf_cluster}_cre_expr_{str(x.cre_expression)}_\n{x.cre_sequence}", axis=1)
                .values.tolist()
            )
        elif "cre_pos" in df.columns and "cre_neg" in df.columns:
            write_fasta_component = "\n".join(
                data[["locus_id", "tf_cluster", "cre_pos", "cre_neg", "cre_sequence"]]
                .apply(lambda x: f">{x.locus_id}_leiden_{x.tf_cluster}_cre_expr_{str(x.cre_pos)}_{str(x.cre_neg)}_\n{x.cre_sequence}", axis=1)
                .values.tolist()
            )
        else:
            raise Exception("no expression columns in the dataframe")
        
        save_fasta_file.write(write_fasta_component)
    
    # Step 2: Run motif scan
    df_motifs = call_motif_scan(fasta_path, get_table=True, progress_bar=True, is_endogen=True)

    # Step 3: Restructure motifs
    motifs_by_cell_type = {}
    if bidir:
        df_groupby = df[(df['cre_pos']>0) | (df['cre_neg'] >0)].groupby('tf_cluster')
    else:
        df_groupby = df.groupby(['tf_cluster', 'cre_expression_discrete'])  
    for group_idx, group_df in df_groupby:
        subset_motif = df_motifs[df_motifs.index.isin(group_df['locus_id'].unique())]
        subset_motif = pd.DataFrame(subset_motif.T.sum(axis=1))
        subset_motif = subset_motif[subset_motif[0] != 0]
        motifs_by_cell_type[group_idx] = subset_motif
    
    motifs = pd.DataFrame(df_motifs.T.sum(axis=1))

    return {
        "motifs": motifs,
        "motifs_by_cell_type": motifs_by_cell_type
    }


def call_motif_scan(fasta, get_table=False, progress_bar=False, is_endogen = False):
    """Run a motif scanning tool on a given FASTA file and return the results.

    This function calls an external motif scanning command (via `command_scan`) on the provided FASTA file using specified
    parameters. It processes the raw output into a pandas DataFrame. Optionally, if `get_table` is True, it formats the DataFrame
    with appropriate headers and indices. If `is_endogen` is True, it truncates the locus identifier for endogenously processed sequences.

    Args:
        fasta (str): Path to the FASTA file containing genomic sequences.
        get_table (bool, optional): If True, formats the output as a table with headers. Defaults to False.
        progress_bar (bool, optional): If True, displays a progress bar during motif scanning. Defaults to False.
        is_endogen (bool, optional): If True, processes locus identifiers to include only the first four underscore-separated components.
            Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing motif scanning results, either as raw output or formatted as a table.
    """

    df_results = pd.DataFrame(line.split('\t') for line in command_scan(
        fasta,
        "JASPAR2020_vertebrates",
        nreport=1,
        fpr=0.01,
        cutoff=None,
        bed=False,
        scan_rc=True,
        table=get_table,
        score_table=False,
        bgfile=None,
        genome="hg38",
        ncpus=N_CPU_PER_PROCESS, # 12
        zscore=False,
        gcnorm=False,
        random_state=None,
        progress=progress_bar
    ))
    if is_endogen:
        # take only locus_id cut off the rest (locus_id is chrZ_start_end)
        df_results[0] = df_results[0].apply(lambda x: '_'.join(x.split('_')[:4]))
    if get_table:
        # function returns index and columns as part of the values, need to make it back
        df_results = df_results.set_index(0)
        df_results.index.names = ['locus_id']
        header = df_results.iloc[0]
        df_results = df_results[1:]
        df_results.columns = header
        return df_results.astype(np.float32)
    
    return df_results
    

def string_to_array(s):
    """Convert a string representation of a numerical array to a NumPy array.

    This function cleans a string by removing extra whitespace and newline characters, then parses the string as a list
    of floating-point numbers. The resulting array is returned as a NumPy array with dtype float16.

    Args:
        s (str): A string representing a numerical array, e.g., "[1.0 2.0 3.0]".

    Returns:
        numpy.ndarray: A NumPy array of floats (dtype=float16) parsed from the input string.
    """

    # Remove newline characters and replace multiple spaces with a single space
    clean_str = re.sub(r'\s+', ' ', s.replace('\n', ' '))
    # Remove the brackets
    clean_str = clean_str.strip('[]')
    # Split the string into a list of floats
    array_list = [float(item) for item in clean_str.split()]
    # Convert the list to a NumPy array
    return np.array(array_list, dtype=np.float16)

def cut_sequences_midpoint(data: pd.DataFrame, seqlen: int):
    """Cut DNA sequences to a specified length centered around the midpoint.

    This function adjusts each sequence in the input DataFrame to have a fixed length (`seqlen`) by centering the cut
    around the midpoint defined by `summit_center`. Depending on the distances from the start and end of the CRE region,
    the sequence is either taken from the beginning, the end, or centered around the summit.

    Args:
        data (pandas.DataFrame): DataFrame containing columns 'cre_sequence', 'summit_center', 'cre_start', and 'cre_end'.
        seqlen (int): Desired sequence length after cutting.

    Returns:
        pandas.DataFrame: The updated DataFrame with the 'cre_sequence' column modified to the new length.
    """

    half = round(seqlen / 2)
    
    # Calculate mid-start and end-mid upfront
    ms = data['summit_center'] - data['cre_start']
    em = data['cre_end'] - data['summit_center']
    
    # Initialize new sequence column
    new_sequences = pd.Series(index=data.index, dtype=str)
    
    # Conditions
    condition1 = ms < half
    condition2 = em < half
    condition3 = ~(condition1 | condition2)
    
    # Apply conditions
    new_sequences[condition1] = data.loc[condition1, 'cre_sequence'].str[:seqlen]
    new_sequences[condition2] = data.loc[condition2, 'cre_sequence'].str[-seqlen:]
    new_sequences[condition3] = data.loc[condition3].apply(
        lambda row: row['cre_sequence'][int(row['summit_center'] - row['cre_start'] - half):
                                     int(row['summit_center'] - row['cre_start'] + half)], axis=1)
    
    # Update the DataFrame
    data['cre_sequence'] = new_sequences
    return data



def fetch_bases_from_genome(chromosome, start, end):
    # fetch the sequence from the genome using genomepy
    sequence = genome.get_seq(chromosome, start + 1, end)
    return sequence.seq


def pad_sequences_midpoint(data: pd.DataFrame, seqlen: int):
    """Pad DNA sequences to a specified length by fetching additional bases from the reference genome.

    For each sequence in the DataFrame, this function calculates the required left and right padding lengths based on
    the distance from the sequence's summit to its start and end positions. It then retrieves the missing bases from
    the reference genome and prepends/appends them to the existing sequence.

    Args:
        data (pandas.DataFrame): DataFrame containing columns 'cre_sequence', 'summit_center', 'cre_start', 'cre_end',
            and 'cre_chrom'.
        seqlen (int): Desired total sequence length after padding.

    Returns:
        pandas.DataFrame: The updated DataFrame with the 'cre_sequence' column padded to the desired length.
    """

    half = round(seqlen / 2)

    # Function to fetch and pad the sequence from the genome
    def pad_sequence(row):
        left_pad_len = max(0, half - (row['summit_center'] - row['cre_start']))  # How much padding is needed on the left
        right_pad_len = max(0, half - (row['cre_end'] - row['summit_center']))  # How much padding is needed on the right
        
        # Fetch left and right padding from genome
        if left_pad_len > 0:
            left_bases = fetch_bases_from_genome(row['cre_chrom'], row['cre_start'] - left_pad_len, row['cre_start'])
        else:
            left_bases = ''
        
        if right_pad_len > 0:
            right_bases = fetch_bases_from_genome(row['cre_chrom'], row['cre_end'], row['cre_end'] + right_pad_len)
        else:
            right_bases = ''
        
        # Return the padded sequence
        return left_bases + row['cre_sequence'] + right_bases
    
    # Apply the padding function to each row
    data['cre_sequence'] = data.apply(lambda row: pad_sequence(row), axis=1)
    
    return data


def onehot_to_1channel_image(array):
    return np.expand_dims(array, axis=1)

class SequenceDataset(Dataset):
    """Custom PyTorch Dataset for DNA sequences and associated labels.

    This dataset wraps one-hot encoded DNA sequences and their corresponding labels (if provided) for easy integration
    with PyTorch data loaders. An optional transformation can be applied to the DNA sequences upon initialization.
    """

    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor = None,
        transform_dna=onehot_to_1channel_image,
        transform_ddsm=False,
        include_labels=True  # flag to control whether to include labels
    ):
        """Initialize the SequenceDataset.

        Args:
            seqs (numpy.ndarray): Array of DNA sequences (already one-hot encoded).
            c (torch.Tensor, optional): Tensor of labels or additional context corresponding to each sequence.
                Defaults to None.
            transform_dna (callable, optional): Function to transform the DNA sequences. Defaults to `onehot_to_1channel_image`.
            transform_ddsm (bool, optional): Placeholder for an additional transformation flag (not used in this implementation).
                Defaults to False.
            include_labels (bool, optional): Flag indicating whether to include labels in the dataset.
                If False, only sequences will be returned. Defaults to True.
        """

        if transform_dna:
            seqs = transform_dna(seqs)
        self.seqs = seqs
        self.include_labels = include_labels

        if include_labels:
            self.c = c

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, index):
        if self.include_labels:
            return self.seqs[index], self.c[index]
        else:
            return self.seqs[index]

def one_hot_encode_dna_sequence(seq, length, transpose=True):

        if len(seq) > length:
            # trim_start = (len(seq) - length) // 2
            # trim_end = trim_start + length
            # seq = seq[trim_start:trim_end]
            seq = seq[:length]

        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        one_hot_seq = np.zeros((length, 4), dtype=np.int8)
        
        # Fill in the one-hot encoded sequence
        for i, nucleotide in enumerate(seq):
            if nucleotide in nucleotide_map:
                one_hot_seq[i, nucleotide_map[nucleotide]] = 1
        
        return one_hot_seq.T if transpose else one_hot_seq

def one_hot_encode(seq, max_seq_len, alphabet):
    """One-hot encode a sequence."""
    alphabet = ["A", "C", "G", "T"]
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array


# def generate_motifs_and_fastas(
#     df: pd.DataFrame, 
#     name: str, 
#     num_sequences: int | None = None
#     ) -> dict[str, Any]:
#     print("Generating Motifs and Fastas...", name)
#     print("---" * 10)

#     # Saving fasta
#     fasta_path = save_fasta(df, name, num_sequences, seq_to_subset_comp=False)

#     # Computing motifs
#     motifs = motifs_from_fasta(fasta_path)

#     # nominally by cell type, actually by tf state of a cell and bin number of cre expression
#     motifs_by_cell_type = {}

#     for group_idx, groud_df in df.groupby(['tf_cluster', 'cre_expression_discrete']):
#         group_name = f"{name}_cell_type_{group_idx[0]}_exprs_{group_idx[1]}"
#         print(group_name)
#         fasta_path = save_fasta(groud_df, group_name, num_sequences, seq_to_subset_comp=True)
#         # groud_idx is tuple (tf_cluster, cre_expression_discrete)
#         motifs_by_cell_type[group_idx] = motifs_from_fasta(fasta_path)

#     return {
#         "motifs": motifs,
#         "motifs_by_cell_type": motifs_by_cell_type,
#         "df": df,
#     }

# def get_cell_type_centroids(df):
#     cell_type_centroids = {}

#     for cell_type, sub_df in df.groupby('cell_type_leiden'):
#         centroid = sub_df['tf_state'].mean()
#         cell_type_centroids[cell_type] = centroid
    
#     return cell_type_centroids

# def get_bin_medians(data: pd.DataFrame):
#     # Calculate bin edges using qcut
#     _, bins = pd.qcut(data[data['cre_expression'] > 0]['cre_expression'], n_expression_bins-1, retbins=True)

#     # Assign each row to a bin using cut
#     data['cre_expression_discrete'] = pd.cut(data['cre_expression'], bins=bins, include_lowest=True)

#     # Calculate the median for each bin
#     bin_medians = data[data['cre_expression'] > 0].groupby(by = 'cre_expression_discrete', observed=False)['cre_expression'].median()

#     # Replace bin labels with the median value of the bin
#     data['cre_expression_discrete'] = data['cre_expression_discrete'].map(bin_medians)

#     # Convert the column to a numeric type before filling NaN values
#     data['cre_expression_discrete'] = pd.to_numeric(data['cre_expression_discrete'], errors='coerce')

#     # Replace NaN values (for zero expressions) with 0
#     data['cre_expression_discrete'] = data['cre_expression_discrete'].fillna(0)

#     return data['cre_expression_discrete'], bins

