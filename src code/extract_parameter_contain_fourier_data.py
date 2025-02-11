import os
import csv
from datetime import datetime
import random
import gc
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from threading import Lock
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
from itertools import product
from collections import Counter

dna_map = {
    "JC+G": 0,
    "K2P+G": 1,
    "F81+G": 2,
    "HKY+G": 3,
    "TN93+G": 4,
    "GTR+G": 5
}


data_path = r"/data/vinhbio/duongpd/new_net/dataset/val"
output_path = r"/data/vinhbio/duongpd/new_net/dataset/extracted_dataset/val"
random_ = 200

csv_lock = Lock()


def count_kmers(sequences, k):
    
    kmer_counts = Counter()
    for sequence in sequences:
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            kmer_counts[kmer] += 1
    return kmer_counts

def generate_kmer_labels(k):
    return [''.join(kmer) for kmer in product("ACGT", repeat=k)]

def create_dataframe_with_labels(kmer_counts, k):
    labels = generate_kmer_labels(k)
    data = {label: kmer_counts.get(label, 0) for label in labels}
    return pd.DataFrame(data.items(), columns=['k-mer', 'Count'])


def list_folders_in_directory(directory):
    results = []
    try:

        first_level_folders = [
            d for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))
        ]
        

        for folder in first_level_folders:
            folder_path = os.path.join(directory, folder)
            second_level_folders = [
                subf for subf in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, subf))
            ]
  
            for subf in second_level_folders:
                results.append(os.path.join(folder_path, subf))

    except Exception as e:
        print(f"An error occurred: {e}")
    return results

def list_folders_in_directory_for_test(directory):

    try:
        return [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def calculate_propensity(substitution_out_matrix, nu_count, transition_count, transversion_count):

    sum_substitution = np.sum(substitution_out_matrix)
    if sum_substitution != 0:
        part_parameters = (substitution_out_matrix / sum_substitution).flatten().tolist()
    else:
        part_parameters = [0] * 16

    sum_nu_count = np.sum(nu_count)
    if sum_nu_count != 0:
        part_parameters.extend((nu_count / sum_nu_count).tolist())
    else:
        part_parameters.extend([0] * 8)

    sum_transition_transversion = transition_count + transversion_count
    if sum_transition_transversion != 0:
        part_parameters.append(transition_count / sum_transition_transversion)
        part_parameters.append(transversion_count / sum_transition_transversion)
    else:
        part_parameters.extend([0, 0])

    return part_parameters


def process_aln(lines):
    result = []
    for line in lines:
        if line:
            result.append(list(filter(None, line.strip().split()))[1])
    return result

def calculate_similarity(seq1, seq2):
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) * 100
def encode_sequence(sequence):
    nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 0} 
    return np.array([nucleotide_map.get(nuc, 0) for nuc in sequence])


def compute_fourier_features(sequence):
    encoded_seq = encode_sequence(sequence)
    fft_result = np.fft.fft(encoded_seq)
    fft_magnitude = np.abs(fft_result)


    dominant_frequency = np.argmax(fft_magnitude)
    max_magnitude = np.max(fft_magnitude)
    mean_magnitude = np.mean(fft_magnitude)
    total_power = np.sum(fft_magnitude ** 2)
    low_frequency_power = np.sum(fft_magnitude[:len(fft_magnitude)//10] ** 2)  
    high_frequency_power = np.sum(fft_magnitude[len(fft_magnitude)//10:] ** 2)
    low_high_ratio = low_frequency_power / (high_frequency_power + 1e-10)


    magnitude_std = np.std(fft_magnitude)
    magnitude_skew = skew(fft_magnitude)
    magnitude_kurtosis = kurtosis(fft_magnitude)

    peak_indices = np.where(fft_magnitude > np.mean(fft_magnitude))[0]
    num_peaks = len(peak_indices)

    return {
        "dominant_frequency": dominant_frequency,
        "max_magnitude": max_magnitude,
        "mean_magnitude": mean_magnitude,
        "total_power": total_power,
        "low_frequency_power": low_frequency_power,
        "high_frequency_power": high_frequency_power,
        "low_high_ratio": low_high_ratio,
        "magnitude_std": magnitude_std,
        "magnitude_skew": magnitude_skew,
        "magnitude_kurtosis": magnitude_kurtosis,
        "num_peaks": num_peaks
    }
def compute_embedding_vector(sequence, k=3):
    nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 0}
    k_mer_vectors = []

    for i in range(len(sequence) - k + 1):
        k_mer = sequence[i:i + k]
        k_mer_vector = [nucleotide_map.get(nuc, 0) for nuc in k_mer]
        k_mer_vectors.append(k_mer_vector)

    embedding_vector = np.mean(k_mer_vectors, axis=0) if k_mer_vectors else np.zeros(k)
    return embedding_vector

def compute_fourier_distances(fourier_matrices):
    num_sequences = len(fourier_matrices)
    distance_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            distance = euclidean(fourier_matrices[i], fourier_matrices[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def extract_fourier_features_from_phy(taxa_sequences):

    features_list = []
    fourier_matrices = []
    embedding_vectors = []

    for sequence in taxa_sequences:
        fourier_features = compute_fourier_features(sequence)
        features_list.append(fourier_features)

        encoded_seq = encode_sequence(sequence)
        fft_result = np.fft.fft(encoded_seq)
        fourier_matrices.append(np.abs(fft_result))

        embedding_vector = compute_embedding_vector(sequence)
        embedding_vectors.append(embedding_vector)

    distance_matrix = compute_fourier_distances(fourier_matrices)

    features_df = pd.DataFrame(features_list)
    embedding_df = pd.DataFrame(embedding_vectors, columns=[f"embedding_dim_{i+1}" for i in range(len(embedding_vectors[0]))])
    features_df = pd.concat([features_df, embedding_df], axis=1)

    features_df = features_df.drop(columns=["peak_indices"], errors="ignore")
    distance_matrix = pd.DataFrame(distance_matrix)
    return features_df.describe().values.flatten().tolist(), distance_matrix.describe().values.flatten().tolist()

def extract_26_parameters(file_path):
    nucleotide_map_local = {
        'A': np.uint16(1),
        'C': np.uint16(3),
        'G': np.uint16(5),
        'T': np.uint16(7)
    }

    substitution_out_matrix = np.zeros((4, 4), dtype=int)
    nu_count = np.zeros(8, dtype=int)
    transition_count = 0
    transversion_count = 0

    with open(file_path, 'r') as file:

        taxa_count, site_count = map(int, file.readline().strip().split())
        lines = file.readlines()
        lines = process_aln(lines)
        pairwise_similarities = []
        fourier_features, fourier_distances = extract_fourier_features_from_phy(lines.copy())

        dataframes_by_k = {}
        for k in range(2, 6):
            kmer_counts = count_kmers(lines.copy(), k)
            dataframes_by_k[k] = create_dataframe_with_labels(kmer_counts, k)
        arrays_by_k = {k: dataframes_by_k[k]['Count'] for k in dataframes_by_k}

        sequence_data = np.array(
            [[nucleotide_map_local.get(nuc, 0) for nuc in seq] for seq in lines],
            dtype=np.uint16
        )


        for i in range(taxa_count):
            for j in range(i + 1, taxa_count):
                seq1 = sequence_data[i]
                seq2 = sequence_data[j]
                pairwise_similarities.append(calculate_similarity(seq1, seq2))

                valid_mask = (seq1 != 0) & (seq2 != 0)
                seq1_valid = seq1[valid_mask]
                seq2_valid = seq2[valid_mask]

                combined = (seq1_valid << 4) | seq2_valid

                substitution_matrix_map = {
                    (1 << 4 | 1): (0, 0),  (1 << 4 | 3): (0, 1),
                    (1 << 4 | 5): (0, 2),  (1 << 4 | 7): (0, 3),
                    (3 << 4 | 1): (1, 0),  (3 << 4 | 3): (1, 1),
                    (3 << 4 | 5): (1, 2),  (3 << 4 | 7): (1, 3),
                    (5 << 4 | 1): (2, 0),  (5 << 4 | 3): (2, 1),
                    (5 << 4 | 5): (2, 2),  (5 << 4 | 7): (2, 3),
                    (7 << 4 | 1): (3, 0),  (7 << 4 | 3): (3, 1),
                    (7 << 4 | 5): (3, 2),  (7 << 4 | 7): (3, 3)
                }
                
                for combo, (row_i, col_j) in substitution_matrix_map.items():
                    count = np.sum(combined == combo)
                    substitution_out_matrix[row_i, col_j] += count

 
                transition_mask = (
                    ((seq1_valid == 1) & (seq2_valid == 5)) |
                    ((seq1_valid == 5) & (seq2_valid == 1)) |
                    ((seq1_valid == 3) & (seq2_valid == 7)) |
                    ((seq1_valid == 7) & (seq2_valid == 3))
                )
                transversion_mask = (seq1_valid != seq2_valid) & ~transition_mask

                transition_count += np.sum(transition_mask)
                transversion_count += np.sum(transversion_mask)

                for idx, nuc in enumerate([1, 3, 5, 7]):
                    nu_count[idx]     += np.sum(seq1_valid == nuc)
                    nu_count[idx + 4] += np.sum(seq2_valid == nuc)


    parameters = calculate_propensity(
        substitution_out_matrix,
        nu_count,
        transition_count,
        transversion_count
    )
    parameters.extend(pd.DataFrame(pairwise_similarities, columns=["Similarity"]).describe().values.flatten().tolist())
    parameters.extend(fourier_features)


    for k, array in arrays_by_k.items():

        parameters.extend(array)
    return parameters



def write_results_to_file(results, filename):

    header = ['label'] + [f'prop{i}' for i in range(1, len(results[0]))]
    output_file_path = os.path.join(output_path, filename)
    file_exists = os.path.isfile(output_file_path)

    with csv_lock:
        with open(output_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(header)
            for result in results:
                csvwriter.writerow(result)


def process_file(file_path, label, filename3):

    try:
        parameters = extract_26_parameters(file_path)
        print(len(parameters))
        result = [label] + parameters
        write_results_to_file([result], "x200_" + filename3)

    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {e}")


def process_folder(folder, filename3):
    label = dna_map.get(os.path.basename(folder))
    if label is None:
        return


    files = os.listdir(folder)

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(process_file, os.path.join(folder, file), label, filename3)
            for file in files
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


def main():

    test = False
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if test:
        folders = list_folders_in_directory_for_test(data_path)
    else:
        folders = list_folders_in_directory(data_path)
    filename3 = f'output_extraction_{current_datetime}.csv'

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_folder, folder, filename3) for folder in folders]
        for future in as_completed(futures):
            try:
                future.result()
            except concurrent.futures.TimeoutError as e:
                print(f"Task timed out: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

        gc.collect()  


if __name__ == "__main__":
    main()
