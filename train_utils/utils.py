import os 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
import editdistance
import torch
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import pairwise_distances
# from sklearn.feature_extraction.text import CountVectorizer
# from collections import Counter
# from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import re
# from concurrent.futures import ProcessPoolExecutor
# from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor
# from itertools import product

import time
from datetime import timedelta
import random

file_dir = os.path.dirname(__file__)


training_data_path =  os.path.join(file_dir, 'train_data')

from utils_data import one_hot_encode, one_hot_encode_dna_sequence, call_motif_scan

def calculate_start_end_indices(total_length, num_processes, process_index):
    """Calculate start and end indices for data partitioning.

    Calculates the start and end indices for a subset of data based on the total number of samples,
    the number of processes, and the index of the current process. The load is balanced by distributing any
    remainder samples to the initial processes.

    Args:
        total_length (int): Total number of data samples.
        num_processes (int): Total number of processes to divide the data among.
        process_index (int): Index of the current process (zero-based).

    Returns:
        tuple: A tuple (start_index, end_index) representing the starting index (inclusive) and ending index (exclusive)
            of the data portion assigned to the specified process.
    """

    num_per_process = total_length // num_processes
    remainder = total_length % num_processes
    
    start_index = process_index * num_per_process + min(process_index, remainder)
    end_index = start_index + num_per_process + (1 if process_index < remainder else 0)
    
    return start_index, end_index


def calculate_validation_metrics(
        motif_data, 
        train_sequences, 
        generated_motif, 
        generated_sequences, 
        get_kmer_metrics=False, # takes a long time
        train_kmer_emb=None, 
        kmer_length=5):
    
    """Calculate validation metrics comparing generated data against training data.

    This function computes multiple validation metrics comparing generated motifs and sequences against training data.
    Metrics include Jensen-Shannon divergence for training, testing, and shuffled motifs, GC content ratio, minimum edit
    distance between sets, and optionally, k-mer based metrics computed via PCA and k-nearest neighbors.

    Args:
        motif_data (dict): Dictionary containing motif information with keys "train_motifs", "test_motifs", and "shuffle_motifs".
        train_sequences (list): List of training DNA sequences.
        generated_motif (pandas.DataFrame or None): DataFrame containing generated motif information. If None, motif metrics default to NaN.
        generated_sequences (list): List of generated DNA sequences.
        get_kmer_metrics (bool, optional): Flag to compute additional k-mer based metrics. Defaults to False.
        train_kmer_emb (numpy.ndarray, optional): Precomputed k-mer embeddings for training sequences. Defaults to None.
        kmer_length (int, optional): Length of k-mers to use for k-mer based metrics. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - train_js (float): Jensen-Shannon divergence between generated motifs and training motifs.
            - test_js (float): Jensen-Shannon divergence between generated motifs and test motifs.
            - shuffle_js (float): Jensen-Shannon divergence between generated motifs and shuffled motifs.
            - gc_ratio (float): Ratio of GC content between generated and training sequences.
            - min_edit_distance (float): Mean minimum edit distance between generated and training sequences.
            - knn_distance (float): Mean k-nearest neighbor distance (if k-mer metrics are computed; otherwise 0).
            - distance_from_closest (float): Average distance from the closest neighbor (if k-mer metrics are computed; otherwise 0).
    """

    if generated_motif is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    train_js = compare_motif_list(generated_motif, motif_data["train_motifs"]) if motif_data.get("train_motifs") is not None else np.nan
    test_js = compare_motif_list(generated_motif, motif_data["test_motifs"]) if motif_data.get("test_motifs") is not None else np.nan
    shuffle_js = compare_motif_list(generated_motif, motif_data["shuffle_motifs"]) if motif_data.get("shuffle_motifs") is not None else np.nan

    if train_sequences is None: # happens when there is no label in train subset that is in label_ratio from full dataset
        return train_js, test_js, shuffle_js, np.nan, np.nan, np.nan, np.nan
    
    gc_ratio = gc_content_ratio(generated_sequences, train_sequences)
    min_edit_distance = min_edit_distance_between_sets(train_sequences, generated_sequences)
    knn_distance = 0
    distance_from_closest = 0
    if get_kmer_metrics:
        generated_kmer_emb = compute_kmer_embeddings(generated_sequences, k=kmer_length)

        train_emb_pca, generated_emb_pca = fit_transform_ipca(train_kmer_emb, generated_kmer_emb, use_gpu=True)
        # Add a batch dimension of size 1 to both tensors to make them compatible with cdist
        distances = torch.cdist(train_emb_pca.unsqueeze(0), generated_emb_pca.unsqueeze(0)).squeeze(0)
        knn_distance = knn_distance_torch(distances, k=15)
        distance_from_closest = distance_from_closest_in_second_set(distances)

    return train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest

def calculate_validation_metrics_parallel(
        motif_data,
        generated_motif,
        train_sequences,
        generated_sequences, 
        accelerator,
        get_seq_metrics=False,
        get_kmer_metrics=False, # takes a long time
        train_kmer_emb=None,
        kmer_length=5
    ):
    """Calculate validation metrics in parallel using an accelerator.

    This function computes validation metrics for generated motifs and sequences in a parallelized manner using an accelerator
    object. It calculates Jensen-Shannon divergences for motifs on the main process and, if requested, computes sequence-based metrics
    such as GC content ratio and minimum edit distance. Optionally, it also computes k-mer based metrics using GPU acceleration.

    Args:
        motif_data (dict): Dictionary containing motif information with keys "train_motifs", "test_motifs", and "shuffle_motifs".
        generated_motif (pandas.DataFrame or None): DataFrame containing generated motif information. If None, motif metrics default to NaN.
        train_sequences (list): List of training DNA sequences.
        generated_sequences (list): List of generated DNA sequences.
        accelerator: An accelerator object providing device information and print methods (e.g., from a distributed training framework).
        get_seq_metrics (bool, optional): Flag to compute sequence-based metrics (GC content and edit distance). Defaults to False.
        get_kmer_metrics (bool, optional): Flag to compute k-mer based metrics. Defaults to False.
        train_kmer_emb (numpy.ndarray, optional): Precomputed k-mer embeddings for training sequences. Defaults to None.
        kmer_length (int, optional): Length of k-mers to use for k-mer based metrics. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - train_js (float or None): Jensen-Shannon divergence between generated motifs and training motifs (computed on the main process).
            - test_js (float or None): Jensen-Shannon divergence between generated motifs and test motifs.
            - shuffle_js (float or None): Jensen-Shannon divergence between generated motifs and shuffled motifs.
            - gc_ratio (float): Ratio of GC content between generated and training sequences.
            - min_edit_distance (float): Mean minimum edit distance between generated and training sequences.
            - knn_distance (float): Mean k-nearest neighbor distance (if k-mer metrics are computed; otherwise 0).
            - distance_from_closest (float): Average distance from the closest neighbor (if k-mer metrics are computed; otherwise 0).
    """

    if accelerator.is_main_process:
        start_time = time.time()
        train_js = compare_motif_list(generated_motif, motif_data["train_motifs"]) if motif_data.get("train_motifs") is not None else np.nan
        test_js = compare_motif_list(generated_motif, motif_data["test_motifs"]) if motif_data.get("test_motifs") is not None else np.nan
        shuffle_js = compare_motif_list(generated_motif, motif_data["shuffle_motifs"]) if motif_data.get("shuffle_motifs") is not None else np.nan
        accelerator.print(f"JSDs run {str(timedelta(seconds=time.time() - start_time))} seconds")
    else:
        train_js, test_js, shuffle_js = None, None, None 
    if not get_seq_metrics: # happens when there is no label in train subset that is in label_ratio from full dataset
        return train_js, test_js, shuffle_js, np.nan, np.nan, np.nan, np.nan
    device = accelerator.device
    # Perform parallelized calculations on each process
    gc_ratio = gc_content_ratio(generated_sequences, train_sequences, device=device)
    min_edit_distance = min_edit_distance_between_sets(generated_sequences, train_sequences, device=device)
    
    knn_distance = torch.tensor(0.0, device=device)
    distance_from_closest = torch.tensor(0.0, device=device)
    
    if get_kmer_metrics:
        generated_kmer_emb = compute_kmer_embeddings(generated_sequences, k=kmer_length)
        train_emb_pca, generated_emb_pca = fit_transform_ipca(train_kmer_emb, generated_kmer_emb, use_gpu=True)
        distances = torch.cdist(train_emb_pca.unsqueeze(0), generated_emb_pca.unsqueeze(0)).squeeze(0)

        knn_distance = knn_distance_torch(distances, k=15)
        distance_from_closest = distance_from_closest_in_second_set(distances)
    return train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest

def gc_content_ratio(train_sequences, generated_sequences, device='cuda', batch_size=500000):
    def process_batches(encoded_sequences, batch_size, device):
        gc_contents = []
        for i in range(0, len(encoded_sequences), batch_size):
            batch = encoded_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            gc_contents.append(torch.mean(torch.sum(batch_tensor, dim=1).float() / batch_tensor.size(1)).item())
            
        return np.mean(gc_contents)

    # Encode sequences
    train_encoded = np.array([[1 if char in "GC" else 0 for char in seq] for seq in train_sequences])
    generated_encoded = np.array([[1 if char in "GC" else 0 for char in seq] for seq in generated_sequences])

    # Process batches
    train_gc = process_batches(train_encoded, batch_size, device)
    generated_gc = process_batches(generated_encoded, batch_size, device)
    
    return generated_gc / train_gc if train_gc > 0 else np.nan

def encode_dna_sequences(sequences):
    """
    Vectorized encoding of a batch of DNA sequences into int8 format.

    Args:
        sequences (list of str): DNA sequences.

    Returns:
        torch.Tensor: Encoded DNA sequences (batch_size, seq_len).
    """
    # Use a mapping table for ASCII-based encoding
    encoding_map = torch.full((256,), -1, dtype=torch.int8)  # Initialize all values as -1
    encoding_map[ord('A')] = 0
    encoding_map[ord('C')] = 1
    encoding_map[ord('G')] = 2
    encoding_map[ord('T')] = 3

    # Convert sequences to ASCII and encode using the mapping
    ascii_tensor = torch.tensor([[ord(base) for base in seq] for seq in sequences], dtype=torch.long)
    encoded_sequences = encoding_map[ascii_tensor]  # Map characters to encoding
    return encoded_sequences

def compute_edit_distance_batches(train_seqs, gen_seqs, max_train_batch=100, max_gen_batch=100):
    """Compute minimum edit distances between batches of training and generated sequences.

    This function calculates the edit distance between each generated sequence and all training sequences using dynamic programming.
    It processes the sequences in batches to handle memory constraints and returns a tensor containing the minimum edit distance
    for each generated sequence across all training batches.

    Args:
        train_seqs (torch.Tensor): Encoded training sequences as a tensor of shape (train_size, seq_len).
        gen_seqs (torch.Tensor): Encoded generated sequences as a tensor of shape (gen_size, seq_len).
        max_train_batch (int, optional): Maximum number of training sequences to process per batch. Defaults to 100.
        max_gen_batch (int, optional): Maximum number of generated sequences to process per batch. Defaults to 100.

    Returns:
        torch.Tensor: A tensor containing the minimum edit distance for each generated sequence.
    """

    train_size, seq_len = train_seqs.size()
    gen_size = gen_seqs.size(0)
    all_min_distances = []

    # Batching over generated sequences
    for i in range(0, gen_size, max_gen_batch):
        gen_batch_end = min(i + max_gen_batch, gen_size)
        gen_batch = gen_seqs[i:gen_batch_end]

        batch_min_distances = []

        # Batching over training sequences
        for j in range(0, train_size, max_train_batch):
            train_batch_end = min(j + max_train_batch, train_size)
            train_batch = train_seqs[j:train_batch_end]

            # Initialize DP matrix for current batches
            dp = torch.zeros((gen_batch_end - i, train_batch_end - j, seq_len + 1, seq_len + 1), dtype=torch.int8, device=train_seqs.device)
            dp[:, :, :, 0] = torch.arange(seq_len + 1, dtype=torch.int8, device=train_seqs.device).view(1, 1, -1).expand(gen_batch_end - i, train_batch_end - j, -1)
            dp[:, :, 0, :] = torch.arange(seq_len + 1, dtype=torch.int8, device=train_seqs.device).view(1, 1, -1).expand(gen_batch_end - i, train_batch_end - j, -1)

            # Compute the DP table
            for k in range(1, seq_len + 1):
                for l in range(1, seq_len + 1):
                    train_col = train_batch[:, k - 1].unsqueeze(0).unsqueeze(2).expand(gen_batch_end - i, train_batch_end - j, seq_len)
                    gen_col = gen_batch[:, l - 1].unsqueeze(1).unsqueeze(2).expand(gen_batch_end - i, train_batch_end - j, seq_len)
                    cost = (train_col[:, :, 0] != gen_col[:, :, 0]).to(torch.int8)
                    dp[:, :, k, l] = torch.min(
                        dp[:, :, k - 1, l] + 1,
                        torch.min(
                            dp[:, :, k, l - 1] + 1,
                            dp[:, :, k - 1, l - 1] + cost
                        )
                    )

            # Collect minimum distances for the current batch of generated sequences against this batch of training sequences
            batch_min_distances.append(dp[:, :, seq_len, seq_len].min(dim=1)[0])

        # Aggregate minimum distances across all training batches for each batch of generated sequences
        all_min_distances.append(torch.stack(batch_min_distances).min(dim=0)[0])

    # Aggregate minimum distances across all batches of generated sequences
    return torch.cat(all_min_distances)  # Return the minimum edit distance for each generated sequence


def min_edit_distance_between_sets(train_sequences, generated_sequences, device='cpu'):
    """
    Compute the mean minimum edit distance between each generated sequence and
    all training sequences.

    Args:
        train_sequences (list of str): DNA sequences in the training set.
        generated_sequences (list of str): DNA sequences in the generated set.
        device (str): 'cpu' or 'cuda' to specify computation device.

    Returns:
        float: Mean of minimum edit distances.
    """
    # Encode and transfer sequences to the specified device
    encoded_train_sequences = encode_dna_sequences(train_sequences).to(device)
    encoded_generated_sequences = encode_dna_sequences(generated_sequences).to(device)

    # Compute batch-wise edit distances
    min_distances = compute_edit_distance_batches(encoded_train_sequences, encoded_generated_sequences)

    # Compute the mean of minimum distances
    return min_distances.float().mean().item()


def sequence_to_kmer_indices_vectorized(sequences, k=5):
    """Convert a list of DNA sequences into vectorized k-mer indices.

    This function maps each DNA sequence to a series of k-mer indices by treating each nucleotide as a digit in a base-4 number,
    using the mapping: {'A': 0, 'C': 1, 'G': 2, 'T': 3}. It returns an array where each row corresponds to a sequence and each column
    contains the integer representation of a k-mer.

    Args:
        sequences (list of str): List of DNA sequences.
        k (int, optional): Length of k-mers to extract. Defaults to 5.

    Returns:
        numpy.ndarray: A 2D array of shape (num_sequences, sequence_length - k + 1) containing the k-mer indices.
    """

    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.array([[nucleotide_map[nuc] for nuc in seq] for seq in sequences])

    seq_len = seq_matrix.shape[1]
    kmer_indices = np.zeros((seq_matrix.shape[0], seq_len - k + 1), dtype=int)
    
    for i in range(k):
        # treat sequences as 4-based number
        kmer_indices += seq_matrix[:, i:seq_len - k + 1 + i] * (4 ** (k - i - 1))

    return kmer_indices

def compute_kmer_embeddings(sequences, k=5):
    """Compute k-mer embeddings for a list of DNA sequences.

    This function generates a k-mer count embedding for each sequence by first converting sequences to k-mer indices and then counting
    the frequency of each possible k-mer. The resulting counts are normalized by the total number of k-mers in each sequence.

    Args:
        sequences (list of str): List of DNA sequences.
        k (int, optional): Length of the k-mers to use. Defaults to 5.

    Returns:
        numpy.ndarray: An array of shape (num_sequences, 4**k) containing normalized k-mer count embeddings.
    """

    kmer_indices = sequence_to_kmer_indices_vectorized(sequences, k=k)
    num_kmers = 4 ** k
    num_sequences = len(sequences)
    embeddings = np.zeros((num_sequences, num_kmers), dtype=np.float32)

    np.add.at(embeddings, (np.arange(num_sequences)[:, None], kmer_indices), 1)
    # Normalize
    embeddings /= (len(sequences[0]) - k + 1)
    return embeddings

def fit_transform_ipca(endogenous_embeddings, generated_embeddings, batch_size=1000000, n_components=50, use_gpu=True):
    """
    Perform Incremental PCA on combined endogenous and generated k-mer embeddings and transform them separately.
    """
    n_components = min([n_components, endogenous_embeddings.shape[0], generated_embeddings.shape[0]])
    
    ipca = IncrementalPCA(n_components=n_components)

    total_size = len(endogenous_embeddings) + len(generated_embeddings)
        
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        
        if start < len(endogenous_embeddings):
            if end <= len(endogenous_embeddings):
                batch = endogenous_embeddings[start:end]
            else:
                batch = np.vstack([endogenous_embeddings[start:], generated_embeddings[:end - len(endogenous_embeddings)]])
        else:
            batch = generated_embeddings[start - len(endogenous_embeddings):end - len(endogenous_embeddings)]
        
        ipca.partial_fit(batch)

    endogenous_embeddings_pca = np.empty((len(endogenous_embeddings), n_components))
    for i in range(0, len(endogenous_embeddings), batch_size):
        batch = endogenous_embeddings[i:i+batch_size]
        endogenous_embeddings_pca[i:i+batch_size] = ipca.transform(batch)

    generated_embeddings_pca = np.empty((len(generated_embeddings), n_components))
    for i in range(0, len(generated_embeddings), batch_size):
        batch = generated_embeddings[i:i+batch_size]
        generated_embeddings_pca[i:i+batch_size] = ipca.transform(batch)
    
    if use_gpu:
        endogenous_embeddings_pca = torch.tensor(endogenous_embeddings_pca).cuda()
        generated_embeddings_pca = torch.tensor(generated_embeddings_pca).cuda()
    
    return endogenous_embeddings_pca, generated_embeddings_pca

def knn_distance_torch(distances, k=15):
    # Find the k-nearest neighbors for each point in set1
    knn_distances, _ = torch.topk(distances, k, dim=-1, largest=False)

    # Return the mean of the k-nearest neighbor distances
    return knn_distances.mean().cpu().item()

def distance_from_closest_in_second_set(distances):
    # Find the minimum distance to any sequence in set2 for each sequence in set1
    min_distances, _ = torch.min(distances, dim=1)

    # Calculate the average of these minimum distances
    return torch.mean(min_distances).cpu().item()


def calculate_similarity_metric(local_train_sequences, local_generated_sequences, seq_length, device, batch_size=500000):
    """Calculate a similarity metric between training and generated DNA sequences.

    This function one-hot encodes the input sequences and computes a similarity matrix using a tensor dot product.
    For each generated sequence, it finds the maximum similarity with any training sequence, and the final metric is the mean
    of these maximum similarity scores.

    Args:
        local_train_sequences (list of str): List of training DNA sequences.
        local_generated_sequences (list of str): List of generated DNA sequences.
        seq_length (int): Length to which sequences should be one-hot encoded.
        device (str): Device to use for tensor computations (e.g., 'cuda' or 'cpu').
        batch_size (int, optional): Batch size for processing sequences. Defaults to 500000.

    Returns:
        float: The mean maximum similarity score between generated and training sequences.
    """

    # One-hot encode the sequences
    train_encoded = torch.tensor(np.array([one_hot_encode_dna_sequence(x, seq_length) for x in local_train_sequences]), dtype=torch.float16)
    generated_encoded = torch.tensor(np.array([one_hot_encode_dna_sequence(x, seq_length) for x in local_generated_sequences]), dtype=torch.float16)

    max_similarities_local = []

    for i in range(0, generated_encoded.shape[0], batch_size):
        generated_batch = generated_encoded[i:i + batch_size].to(device)

        for j in range(0, train_encoded.shape[0], batch_size):
            train_batch = train_encoded[j:j + batch_size].to(device)

            similarity_matrix = torch.tensordot(generated_batch, train_batch, dims=([1, 2], [1, 2]))
            max_sim_local_batch = torch.max(similarity_matrix, dim=1).values
            
            max_similarities_local.append(max_sim_local_batch)

    if len(max_similarities_local) > 0:
        local_max_similarity = torch.cat(max_similarities_local).mean().item()
    else:
        local_max_similarity = 0  # or handle as needed

    return local_max_similarity

def extract_motifs(sequence_list: list, run_name='', save_bed_file=False, get_table=False):
    """Extract motif information from a list of sequences.

    This function writes the provided list of sequences to a FASTA file, runs an external motif scanning tool on the file,
    and processes the output to aggregate motif counts. Optionally, the results can be saved to a BED file or returned
    as a detailed table.

    Args:
        sequence_list (list): List of DNA sequences.
        run_name (str, optional): Identifier to prefix output file names. Defaults to ''.
        save_bed_file (bool, optional): If True, saves the motif scanning results to a BED file. Defaults to False.
        get_table (bool, optional): If True, returns the full table of motif scanning results. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing aggregated motif counts by motif name. If get_table is True, returns the raw motif table.
    """

    fasta_name = f"{training_data_path}/{run_name}synthetic_motifs.fasta"
    motifs = open(fasta_name, "w")
    motifs.write("\n".join(sequence_list))
    motifs.close()
    # os.system(f"gimme scan {fasta_name} -p JASPAR2020_vertebrates -g hg38 --nthreads 8 > {training_data_path}/{run_name}syn_results_motifs.bed")    
    df_results_syn = call_motif_scan(fasta_name, get_table, progress_bar=False)
    if df_results_syn.empty:
        return df_results_syn
    df_results_syn_ = df_results_syn.copy()
    if get_table:
        df_results_syn_ = pd.DataFrame(df_results_syn.T.sum(axis=1))
    if save_bed_file:
        bed_file = f"{training_data_path}/{run_name}syn_results_motifs.bed"
        df_results_syn_.to_csv(bed_file, sep='\t', index=True)
    if get_table:
        return df_results_syn
    
    # df_results_syn = read_csv_with_skiprows(f"{training_data_path}/{run_name}syn_results_motifs.bed")
    
    # extract motif id
    # example: {motif_name "MA0761.2_ETV1" ; motif_instance "AACTCTTCCTGTTT"} ==> {MA0761.2_ETV1}
    try:
        df_results_syn["motifs"] = df_results_syn[8].apply(extract_text_between_quotes)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(df_results_syn)
        raise e
    # extract sequence id from sequence list by taking all before last "_"
    # example seq_test_0_0 ==> seq_test_0, which does not nothing for the count
    #df_results_syn[0] = df_results_syn[0].apply(lambda x: x.rpartition("_")[0])
    df_motifs_count_syn = df_results_syn[df_results_syn['motifs'].notna()]
    df_motifs_count_syn = df_motifs_count_syn[[0, "motifs"]].groupby("motifs").count()
    return df_motifs_count_syn


def compare_motif_list(df_motifs_a: pd.DataFrame, df_motifs_b: pd.DataFrame):
    """Compare two motif count distributions using Jensen-Shannon divergence.

    This function takes two DataFrames of motif counts, reindexes them to share the same set of motifs (filling missing values with 1),
    and computes the probability distributions. It then calculates and returns the Jensen-Shannon divergence between these distributions.
    If one or both DataFrames are empty, a divergence of 1.0 is returned.

    Args:
        df_motifs_a (pandas.DataFrame): DataFrame of motif counts for the first dataset, indexed by motif names.
        df_motifs_b (pandas.DataFrame): DataFrame of motif counts for the second dataset, indexed by motif names.

    Returns:
        float: The Jensen-Shannon divergence between the two motif distributions.
    """

    
    if df_motifs_a.empty or df_motifs_b.empty:
        print("One or both DataFrames are empty. Returning JS divergence of 1.")
        return 1.0
    
    # Get a combined set of motif names
    motifs_union = df_motifs_a.index.union(df_motifs_b.index)
    
    # Reindex both dataframes to this combined set, filling missing values with 1
    df_motifs_a = df_motifs_a.reindex(motifs_union, fill_value=1)
    df_motifs_b = df_motifs_b.reindex(motifs_union, fill_value=1)
    
    # Calculate probabilities
    total_a = df_motifs_a[0].sum()
    total_b = df_motifs_b[0].sum()
    df_motifs_a['prob_a'] = df_motifs_a[0] / total_a
    df_motifs_b['prob_b'] = df_motifs_b[0] / total_b
    
    # Compute the Jensen-Shannon divergence
    js_divergence = jensenshannon(df_motifs_a['prob_a'], df_motifs_b['prob_b'])
    
    return js_divergence

def extract_text_between_quotes(text):
    if type(text) != str:
        text = str(text)
        print("-"*24, "\n error:")
        print(text)
    match = re.search(r'"(.*?)"', text)
    if match:
        return match.group(1)
    return None

def js_heatmap(
    cell_dict1:dict,
    cell_dict2:dict,
    label_list,   
    ):
    """Compute a matrix of Jensen-Shannon divergences for heatmap visualization.

    This function iterates over a list of cell labels and compares motif distributions between two cell dictionaries.
    For each pair of labels, it computes the Jensen-Shannon divergence using the motif comparison function.
    If motif data is missing for a given cell label, a divergence value of -1 is used.

    Args:
        cell_dict1 (dict): Dictionary mapping cell labels to motif count DataFrames (first dataset).
        cell_dict2 (dict): Dictionary mapping cell labels to motif count DataFrames (second dataset).
        label_list (list): List of cell labels to compare.

    Returns:
        list: A 2D list (matrix) containing Jensen-Shannon divergence values for each pair of cell labels.
    """

    
    final_comp_js = []
    for cell_num1 in label_list:
        comparison_array = []
        motifs1 = cell_dict1.get(cell_num1)
        for cell_num2 in label_list:
            motifs2 = cell_dict2.get(cell_num2)
            if motifs1 is not None and motifs2 is not None:
                js_out = compare_motif_list(motifs1, motifs2)
            else:
                js_out = -1
            comparison_array.append(js_out)
        final_comp_js.append(comparison_array)
    return final_comp_js


def generate_heatmap(df_heat: pd.DataFrame, x_label: str, y_label: str, label_list: list, save_dir: str = training_data_path):
    """Generate and save a heatmap of Jensen-Shannon divergence values.

    This function creates a heatmap from a DataFrame of divergence values. It labels the axes using the provided label list,
    applies visual customizations such as annotation size and color scaling, and saves the resulting heatmap image to the specified directory.

    Args:
        df_heat (pandas.DataFrame): DataFrame containing divergence values.
        x_label (str): Label for the x-axis (e.g., name of the first dataset).
        y_label (str): Label for the y-axis (e.g., name of the second dataset).
        label_list (list): List of tuples or identifiers used to label the heatmap axes.
        save_dir (str, optional): Directory where the heatmap image will be saved. Defaults to the training data path.

    Returns:
        None
    """

    df_plot = pd.DataFrame(df_heat)
    df_plot.columns = [f"{x[0]} : {x[1]}" for x in label_list]
    df_plot.index = df_plot.columns

    # remove rows and columns containing only -1 values
    df_plot = df_plot.loc[~(df_plot == -1).all(axis=1), :]
    df_plot = df_plot.loc[:, ~(df_plot == -1).all(axis=0)]

    num_rows, num_cols = df_plot.shape
    fig_width = 10 + num_cols * 2  
    fig_height = 10 + num_rows * 2  

    plt.clf()
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    font_size = 40
    plt.rcParams.update({'font.size': font_size})

    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.5, vmax=1, vmin=0, annot_kws={"size": font_size})

    plt.title(f"JS divergence \n {x_label} sequences x {y_label} sequences \n MOTIFS probabilities")
    plt.xlabel(f"{x_label} Sequences \n(motifs dist)")
    plt.ylabel(f"{y_label} \n (motifs dist)")
    plt.grid(False)
    plt.savefig(f"{save_dir}/{x_label}_{y_label}_js_heatmap.png")


def plot_training_loss(values, save_dir):
    """Plot and save the training loss curve.

    This function generates a line plot of the training loss values over epochs and saves the plot image to the specified directory.

    Args:
        values (list or array-like): Sequence of loss values recorded during training.
        save_dir (str): Directory where the loss plot image will be saved.

    Returns:
        None
    """

    plt.figure()
    plt.plot(values)
    plt.title(f"Training process \n Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"{save_dir}/loss_training.png")


def plot_training_validation(values_list, y_labels, per_epoch, save_dir):
    """Plot and save validation metrics recorded during training.

    This function creates a plot for one or more validation metrics, where each metric is plotted against epochs.
    The x-axis is scaled based on the frequency (per_epoch) at which metrics were recorded, and the plot is saved to the specified directory.

    Args:
        values_list (list of list or array-like): A list containing sequences of validation metric values.
        y_labels (list of str): Labels corresponding to each validation metric.
        per_epoch (int): Frequency (in epochs) at which validation metrics were recorded.
        save_dir (str): Directory where the validation metrics plot will be saved.

    Returns:
        None
    """

    plt.figure()
    for idx, values in enumerate(values_list):
        X = list(range(0,len(values)*per_epoch,per_epoch))
        plt.plot(X, values, label=y_labels[idx])
    
    plt.title(f"Training process \n Validation stats every {per_epoch} epoch")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(f"{save_dir}/validation_training.png")

# def calculate_similarity_metric(train_sequences, generated_sequences, seq_length, batch_size=1000000):
#     start_time = time.time()

#     # One-hot encode the sequences
#     train_encoded = torch.tensor([one_hot_encode_dna_sequence(x, seq_length) for x in train_sequences], dtype=torch.float16)
#     generated_encoded = torch.tensor([one_hot_encode_dna_sequence(x, seq_length) for x in generated_sequences], dtype=torch.float16)

#     max_similarities = []

#     for i in range(0, generated_encoded.shape[0], batch_size):
#         generated_batch = generated_encoded[i:i + batch_size].cuda(non_blocking=True)

#         batch_max_similarities = []
#         for j in range(0, train_encoded.shape[0], batch_size):
#             # Corrected to use train_encoded instead of generated_encoded
#             train_batch = train_encoded[j:j + batch_size].cuda(non_blocking=True)

#             # Check for empty batches before performing operations
#             if train_batch.size(0) > 0 and generated_batch.size(0) > 0:
#                 # Calculate similarity using dot product
#                 similarity_matrix = torch.tensordot(generated_batch, train_batch, dims=([1, 2], [1, 2]))

#                 # Check for non-empty similarity_matrix
#                 if similarity_matrix.size(1) > 0:
#                     batch_max_similarities.append(torch.max(similarity_matrix, dim=1).values.cpu())

#         if batch_max_similarities:
#             max_similarities.append(torch.max(torch.stack(batch_max_similarities), dim=0).values)

#     # Ensure max_similarities is not empty before concatenation
#     if max_similarities:
#         mean_similarity = torch.cat(max_similarities).mean().item() / seq_length
#     else:
#         raise ValueError("No valid similarities found; check your input data.")

#     print(f"calculate_similarity_metric run {str(timedelta(seconds=time.time() - start_time))} seconds")
#     return mean_similarity

# def gc_content_ratio(train_sequences, generated_sequences):
#     def gc(seq):
#         return (seq.count("G") + seq.count("C")) / len(seq) if len(seq) > 0 else 0

#     with ThreadPoolExecutor(max_workers=28) as executor:
#         train_gc_futures = executor.map(gc, train_sequences)
#         generated_gc_futures = executor.map(gc, generated_sequences)

#     train_gc = np.mean(list(train_gc_futures))
#     generated_gc = np.mean(list(generated_gc_futures))

#     return generated_gc / train_gc if train_gc > 0 else np.nan

# def min_edit_distance_between_sets(train_sequences, generated_sequences, batch_size=1000000):
#     def min_distance_from_one(seq, set2):
#         return min(editdistance.eval(seq, seq2) for seq2 in set2)

#     def batch_min_distance(batch, set2):
#         return [min_distance_from_one(seq, set2) for seq in batch]

#     min_distances = []
#     with ThreadPoolExecutor(max_workers=28) as executor:
#         for i in range(0, len(train_sequences), batch_size):
#             train_batch = train_sequences[i:i + batch_size]
#             future = executor.submit(batch_min_distance, train_batch, generated_sequences)
#             min_distances.extend(future.result())

#     return np.mean(min_distances)



# def generate_kmer_frequencies(set1, set2, k=5, normalize=True, use_pca=True, n_components=50, print_variance=True):
#     # Combine both sets to fit the vectorizer
#     combined_sequences = set1 + set2
#     # Create all possible k-mers from 'A', 'C', 'G', 'T'
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    
#     # Fit the vectorizer on the combined set
#     vectorizer.fit(combined_sequences)
    
#     # Transform each set separately
#     X1 = vectorizer.transform(set1).toarray()
#     X2 = vectorizer.transform(set2).toarray()
    
#     if normalize:
#         set1_length = len(set1[0]) - k + 1
#         set2_length = len(set2[0]) - k + 1
#         X1 = X1 / set1_length
#         X2 = X2 / set2_length
    
#     if use_pca:
#         # Apply PCA transformation
#         pca = PCA(n_components=n_components, svd_solver = "arpack")
#         # Fit PCA on the combined data to ensure the same transformation is applied to both sets
#         pca.fit(np.vstack((X1, X2)))
#         if print_variance:
#             print("Fraction of total variance explained by PCA components: ", np.sum(pca.explained_variance_ratio_))
#         X1 = pca.transform(X1)
#         X2 = pca.transform(X2)
    
#     return X1, X2

def calculate_similarity(train_sequences, generated_sequences, seq_length, batch_size=100000):
    """
    Calculate similarity metrics between one-hot encoded train DNA sequences and generated DNA sequences.
    This function uses batching to reduce memory usage.
    """
    # One-hot encode the DNA sequences
    train_encoded = np.array([one_hot_encode_dna_sequence(x, seq_length) for x in train_sequences], dtype=np.float32)
    generated_encoded = np.array([one_hot_encode_dna_sequence(x, seq_length) for x in generated_sequences], dtype=np.float32)

    num_train = train_encoded.shape[0]
    num_generated = generated_encoded.shape[0]

    # Initialize the max similarities array
    max_similarities = np.zeros(num_generated, dtype=np.float32)
    
    # Process in batches to reduce memory usage
    for i in range(0, num_generated, batch_size):
        generated_batch = generated_encoded[i:i + batch_size]
        batch_max_similarities = np.zeros(generated_batch.shape[0], dtype=np.float32)

        for j in range(0, num_train, batch_size):
            train_batch = train_encoded[j:j + batch_size]
            # Calculate similarity for the batch using dot product
            similarity_matrix = np.tensordot(generated_batch, train_batch, axes=([1, 2], [1, 2]))
            # Update the max similarities for each generated sequence
            batch_max_similarities = np.maximum(batch_max_similarities, np.max(similarity_matrix, axis=1))

        max_similarities[i:i + batch_max_similarities.shape[0]] = batch_max_similarities

    # Compute the mean similarity normalized by the sequence length
    mean_similarity = np.mean(max_similarities) / seq_length  # Normalize by sequence length

    return mean_similarity

