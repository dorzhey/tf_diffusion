import copy
from typing import Any
import os 
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from collections import Counter
from collections import defaultdict
import shutil 
import random
import gc

random.seed(10)
torch.manual_seed(10)
np.random.seed(10)


LABEL_SIZE = 51

file_dir = os.path.dirname(__file__)



train_utils_path = os.path.join(file_dir,'..','train_utils')
training_data_path =  os.path.join(train_utils_path, 'train_data')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)

from utils import (
    extract_motifs,
    calculate_validation_metrics,
    calculate_validation_metrics_parallel,
    calculate_similarity_metric,
    compute_kmer_embeddings,
    calculate_start_end_indices
)

from utils_data import SequenceDataset, get_locus_motifs

import time
from datetime import timedelta


class TrainLoop:
    def __init__(
        self,
        config: dict[str, Any],
        data: dict[str, Any],        
        model: torch.nn.Module,
        accelerator: Accelerator,
        epochs: int = 100,
        log_step_show: int = 1,
        sample_epoch: int = 1,
        save_epoch: int = 5,
        model_name: str = "model_full_conditioning",
        num_sampling_to_compare_cells: int = 10000,
        run_name = ''
    ):
        self.encode_data = data
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_step_show = log_step_show
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch
        self.model_name = model_name
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells
        # pulling parameters from config
        self.batch_size = config.batch_size
        # self.sample_bs = config.sample_bs
        self.seq_length = config.seq_length
        self.get_seq_metrics = config.get_seq_metrics
        self.get_kmer_metrics_bulk = config.get_kmer_metrics_bulk
        self.get_kmer_metrics_labelwise = config.get_kmer_metrics_labelwise
        self.kmer_length = config.kmer_length
        self.min_sample_size = config.min_sample_size
        self.parallel_generating_bs = config.parallel_generating_bs
        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        
        self.label_ratio = data['label_ratio']
        
        self.initialize_sampling()
        self.tf_cluster_state_dict = data['tf_cluster_state_dict']

        # Metrics
        self.train_js, self.test_js, self.shuffle_js = 1, 1, 1
        self.seq_similarity = 1
        self.min_edit_distance, self.knn_distance, self.distance_from_closest = 100, 0, 0
        self.test_js_by_cell_type_global_min = 1
        # for plots
        self.loss_values = []
        self.all_train_js, self.all_test_js, self.all_shuffle_js = [],[],[]
        self.all_seq_similarity = []
        
        self.start_epoch = 1
        if self.get_kmer_metrics_bulk:
            self.accelerator.print("precomputing kmer embeddings")
            self.train_kmer_emb = compute_kmer_embeddings(self.encode_data["train_sequences"], k=self.kmer_length)
        else:
            self.train_kmer_emb = None
        # Dataloader
        seq_dataset = SequenceDataset(seqs=torch.from_numpy(self.encode_data["X_train"]).float(), 
                                      c=torch.from_numpy(self.encode_data["x_train_cell_type"]).float(), 
                                      transform_ddsm = False)
        self.train_dl = DataLoader(seq_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

        self.log_dict = {}
        self.run_name = run_name

    def initialize_sampling(self):
        num_sampling_to_compare_cells = self.num_sampling_to_compare_cells
        min_sample_size = self.min_sample_size

        # Calculate the sampling ratio
        sampling_ratio = {
            (x, y): max(round(z * num_sampling_to_compare_cells), min_sample_size)
            for (x, y), z in self.label_ratio.items()
        }

        # Create a label array based on sampling ratio
        extended_label_array = []
        for (tf_cluster, expression_level), cell_type_sample_size in sampling_ratio.items():
            extended_label_array.extend([(tf_cluster, expression_level)] * cell_type_sample_size)
        
        # Split the extended_label_array among processes
        start_index, end_index = calculate_start_end_indices(len(extended_label_array), self.accelerator.num_processes, self.accelerator.process_index)
        self.process_label_array = extended_label_array[start_index:end_index]

        # Now, split the sampling_ratio dictionary among processes
        keys = list(sampling_ratio.keys())
        start_index_ratio, end_index_ratio = calculate_start_end_indices(len(keys), self.accelerator.num_processes, self.accelerator.process_index)
        process_sampling_keys = keys[start_index_ratio:end_index_ratio]
        
        self.process_sampling_ratio = {key: sampling_ratio[key] for key in process_sampling_keys}

        self.total_number_of_samples = len(extended_label_array) # might be used
        self.total_number_of_labels = len(keys)
        # Shuffle the process_label_array to ensure randomness
        random.shuffle(self.process_label_array)
        self.accelerator.print(f"Total number of samplle generated during each validation step: {self.total_number_of_samples}")
        print(f"Process {self.accelerator.process_index} handles samples from index {start_index} to {end_index}.")
        self.accelerator.wait_for_everyone() # try and group prints
        print(f"Process {self.accelerator.process_index} handles labels from index {start_index_ratio} to {end_index_ratio}.")

        

    def train_loop(self):
        # Prepare for training
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, self.train_dl)
        wandb_config = {
                        'batch_size':self.batch_size, 
                        "num_sampling_to_compare_cells": self.num_sampling_to_compare_cells,
                        'min_sample_size':self.min_sample_size,
                        'parallel_generating_bs':self.parallel_generating_bs,
                        'seq_length':self.seq_length,
                        }
        if self.accelerator.is_main_process:
            # Initialize wandb
            self.accelerator.init_trackers("dnadiffusion_full_cond", 
                                           config=wandb_config)
            
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            # Getting loss of current batch
            for step, batch in enumerate(self.train_dl):
                loss = self.train_step(batch)
            # Sampling
            if epoch % self.sample_epoch == 0:
                #self.sample(epoch)
                generated_bulk_motif, generated_celltype_motif, generated_celltype_sequences = self.create_sample_labelwise_parallel_dynamic(epoch)
                self.calculate_metrics_labelwise_parallel(epoch, generated_bulk_motif, generated_celltype_motif, generated_celltype_sequences)

            # Logging loss
            if epoch % self.log_step_show == 0 and self.accelerator.is_main_process:
                self.log_step(loss, epoch)
            # Saving model
            # if epoch % self.save_epoch == 0 and self.accelerator.is_main_process:
            #     self.save_model(epoch)
            # now saving if new best test_js
            self.accelerator.wait_for_everyone()
            
    def train_step(self, batch):
        x,y = batch
        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            loss = self.model(x, y)

        
        self.accelerator.backward(loss)
        self.optimizer.step()
        if self.accelerator.is_main_process:
            self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))
        return loss

    def create_sample_labelwise_parallel_dynamic(self, epoch):
        self.accelerator.print(f"Sampling at epoch {epoch}")
        self.model.eval()
        diffusion_model = self.accelerator.unwrap_model(self.model)

        nucleotides = ["A", "C", "G", "T"]
        # process_run_name = self.run_name + str(self.accelerator.process_index)
        label_array = self.process_label_array
        # No shuffling if sequential order is desired
        # np.random.shuffle(label_array)
        bulk_fasta_sequences = []
        generated_celltype_sequences = {}
        num_batches = len(label_array) // self.parallel_generating_bs

        for i in tqdm(range(num_batches + 1), desc="generating"):
            batch_labels = label_array[i * self.parallel_generating_bs:(i + 1) * self.parallel_generating_bs]
            if not batch_labels:
                continue

            conditions =  np.array([np.append(self.tf_cluster_state_dict[tf_cluster], expression_level) for tf_cluster, expression_level in batch_labels])
            conditions_tensor = torch.tensor(conditions).float().to(diffusion_model.device)
            sampled_images = diffusion_model.sample(conditions_tensor, shape = (len(conditions), 1, 4, self.seq_length), cond_weight = 1)[-1]

            for j, image in enumerate(sampled_images):
                sequence = "".join([nucleotides[s] for s in np.argmax(image.reshape(4, self.seq_length), axis=0)])
                # header {tf_cluster}_{expression_level}
                header = f">seq_id_{batch_labels[j][0]}_{batch_labels[j][1]}_{i}_{j}"
                fasta_sequence = f"{header}\n{sequence}"
                bulk_fasta_sequences.append(fasta_sequence)

                if batch_labels[j] not in generated_celltype_sequences:
                    generated_celltype_sequences[batch_labels[j]] = [sequence]
                else:
                    generated_celltype_sequences[batch_labels[j]].append(sequence)

        self.accelerator.wait_for_everyone()
        # Synchronize and gather the results from all processes
        gathered_bulk_fasta_sequences = self.accelerator.gather_for_metrics(bulk_fasta_sequences, use_gather_object=True)
        # -------------------- CHECKING GATHERING METHODS
        print(f"Process {self.accelerator.process_index} gathered {len(gathered_bulk_fasta_sequences)} of bulk sequences out of {self.total_number_of_samples}")
        # --------------------
        gathered_celltype_sequences = self.accelerator.gather_for_metrics([generated_celltype_sequences], use_gather_object=True)
        
        # Reconstruct the dictionary for generated_celltype_sequences
        reconstructed_celltype_sequences = {}
        for item in gathered_celltype_sequences:
            reconstructed_celltype_sequences.update(item)
        # -------------------- CHECKING GATHERING METHODS
        print(f"Process {self.accelerator.process_index} gathered {sum(len(value) for value in reconstructed_celltype_sequences.values())} of cell type sequences out of {self.total_number_of_samples}")
        # --------------------
        # now we want to extract_motifs, but only once
        generated_bulk_motif = pd.DataFrame()
        constructed_celltype_motif = {}
        # Now reconstruct the motif counts from a table
        if self.accelerator.is_main_process:
            generated_bulk_motif_table = extract_motifs(gathered_bulk_fasta_sequences, self.run_name, save_bed_file=True, get_table=True)
            generated_bulk_motif, constructed_celltype_motif = self.bulk_motif_to_celltype(generated_bulk_motif_table)
        
        self.accelerator.wait_for_everyone()

        reconstructed_bulk_motif = self.accelerator.gather_for_metrics([generated_bulk_motif], use_gather_object=True)[0]
        # -------------------- CHECKING GATHERING METHODS
        if self.accelerator.is_main_process:
            print(f"Main process gathered {reconstructed_bulk_motif.equals(generated_bulk_motif)} same DataFrame of bulk motif counts")
        # --------------------
        reconstructed_celltype_motif_ = self.accelerator.gather_for_metrics([constructed_celltype_motif], use_gather_object=True)

        reconstructed_celltype_motif = {}
        for item in reconstructed_celltype_motif_:
            reconstructed_celltype_motif.update(item)
        # -------------------- CHECKING GATHERING METHODS
        print(f"Process {self.accelerator.process_index} gathered {len(reconstructed_celltype_motif.keys())} label motif counts out of {self.total_number_of_labels}")
        # --------------------
        self.model.train()
        return reconstructed_bulk_motif, reconstructed_celltype_motif, reconstructed_celltype_sequences
           
    def bulk_motif_to_celltype(self, df_motif_table):
        # we take table sequence_id as 0 column x motif name as first row
        df_motifs = df_motif_table.set_index(0)
        header = df_motifs.iloc[0]
        df_motifs = df_motifs[1:]
        df_motifs.columns = header
        df_motifs = df_motifs.astype(np.float32)
        generated_celltype_motif = {}
        # cut the sequence id to get tf_cluster and cre_expression
        idx_to_label = {sequence_id: list(sequence_id.split('_'))[2:4] for sequence_id in df_motifs.index.values}
           
        for (tf_cluster, cre_expression), _ in self.label_ratio.items():
            idx_subset = [idx for idx, label in idx_to_label.items() 
                          if int(label[0]) == tf_cluster and int(label[1]) == cre_expression]
            subset_motif = df_motifs[df_motifs.index.isin(idx_subset)]
            # turn to motif name as index and then sum, getting motif counts
            subset_motif = pd.DataFrame(subset_motif.T.sum(axis=1))
            # drop empty motifs
            subset_motif = subset_motif[subset_motif[0] != 0]
            generated_celltype_motif[(tf_cluster, cre_expression)] = subset_motif
        bulk_motifs = pd.DataFrame(df_motifs.T.sum(axis=1))

        return bulk_motifs, generated_celltype_motif

    def calculate_metrics_labelwise_parallel(self, epoch, generated_motif, generated_celltype_motif, generated_celltype_sequences):
        self.accelerator.print("calculating metrics at epoch", epoch)
        device = self.accelerator.device
        start_time = time.time()
        generated_sequences = [sequence for sequences in generated_celltype_sequences.values() for sequence in sequences]
    
        start_index_train, end_index_train = calculate_start_end_indices(len(self.encode_data["train_sequences"]), self.accelerator.num_processes, self.accelerator.process_index)
        start_index_generated, end_index_generated = calculate_start_end_indices(len(generated_sequences), self.accelerator.num_processes, self.accelerator.process_index)
        local_train_sequences = self.encode_data["train_sequences"][start_index_train:end_index_train]
        local_generated_sequences = generated_sequences[start_index_generated:end_index_generated]

        local_max_similarity = calculate_similarity_metric(local_train_sequences, local_generated_sequences, self.seq_length, device)

        all_max_similarities = self.accelerator.gather_for_metrics(torch.tensor(local_max_similarity, device=device))

        if self.accelerator.is_main_process:
            seq_similarity = sum(all_max_similarities).item() / self.accelerator.num_processes / self.seq_length
            self.accelerator.print("Similarity :", seq_similarity)
        
        self.accelerator.print(f"calculate_similarity_metric run {str(timedelta(seconds=time.time() - start_time))} seconds")
        

        
        start_time = time.time()
        # first calculate validation metrics in bulk in parallel
        train_js, test_js, shuffle_js, gc_ratio_local, min_edit_distance_local, knn_distance_local, distance_from_closest_local = calculate_validation_metrics_parallel(
            motif_data = {
                "train_motifs":self.encode_data["train_motifs"], 
                "test_motifs": self.encode_data["test_motifs"],
                "shuffle_motifs":self.encode_data["shuffle_motifs"],
            }, 
            generated_motif = generated_motif,
            train_sequences = local_train_sequences if self.get_seq_metrics else None,
            generated_sequences = local_generated_sequences if self.get_seq_metrics else None,
            accelerator = self.accelerator,
            get_seq_metrics = self.get_seq_metrics,
            get_kmer_metrics = self.get_kmer_metrics_bulk,
            train_kmer_emb = self.train_kmer_emb,
            kmer_length = self.kmer_length
        )
        # Gather results across all processes
        self.accelerator.wait_for_everyone()
        
        gc_ratio = self.accelerator.gather_for_metrics(torch.tensor([gc_ratio_local], device=device))
        min_edit_distance = self.accelerator.gather_for_metrics(torch.tensor([min_edit_distance_local], device=device))
        knn_distance = self.accelerator.gather_for_metrics(torch.tensor([knn_distance_local], device=device))
        distance_from_closest = self.accelerator.gather_for_metrics(torch.tensor([distance_from_closest_local], device=device))

        if self.accelerator.is_main_process:
            # -------------------- CHECKING GATHERING METHODS
            print(f"Main process gathered out of {self.accelerator.num_processes} :")
            print(f"{len(all_max_similarities)} local similarity values ")
            if self.get_seq_metrics:
                print(f"{len(gc_ratio)} local gc_ratio values ")
                print(f"{len(min_edit_distance)} local min_edit_distance values")
            if self.get_kmer_metrics_bulk:
                print(f"{len(knn_distance)} local knn_distance values")
                print(f"{len(distance_from_closest)} local distance_from_closest values")
            # --------------------
            # Calculate final averages on the main process
            gc_ratio = gc_ratio.mean().item()
            min_edit_distance = min_edit_distance.mean().item()
            knn_distance = knn_distance.mean().item()
            distance_from_closest = distance_from_closest.mean().item()
        else:
            gc_ratio, min_edit_distance, knn_distance, distance_from_closest = None, None, None, None
        
        if self.accelerator.is_main_process:
            self.all_seq_similarity.append(seq_similarity)
            self.all_train_js.append(train_js)
            self.all_test_js.append(test_js)
            self.all_shuffle_js.append(shuffle_js)
            
            bulk_metrics = {
                "train_js": train_js,
                "test_js": test_js,
                "shuffle_js": shuffle_js,
                "seq_similarity": seq_similarity,
                "gc_ratio":gc_ratio,
                "edit_distance": min_edit_distance,
                "knn_distance":knn_distance,
                "distance_endogenous":distance_from_closest     
            }
            
            metrics_delete = []
            if not self.get_seq_metrics:
                metrics_delete.extend(['gc_ratio', 'edit_distance'])
            if not self.get_kmer_metrics_bulk:
                metrics_delete.extend(['knn_distance','distance_endogenous'])
            for metric in metrics_delete:
                bulk_metrics.pop(metric, None)

            self.accelerator.print(f"calculate_metrics bulk part run {str(timedelta(seconds=time.time() - start_time))} seconds")
            start_time = time.time()
        
            for metric, value in bulk_metrics.items():
                self.accelerator.print(metric,":",value)
        
        # now, generate metrics cell_type-wise per process and then gather and average it
        validation_metric_cell_specific = defaultdict(list)
        for label, _ in self.process_sampling_ratio.items():            
            train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(
                motif_data = {
                "train_motifs":self.encode_data["train_motifs_cell_specific"].get(label), 
                "test_motifs": self.encode_data["test_motifs_cell_specific"].get(label),
                "shuffle_motifs":self.encode_data["shuffle_motifs_cell_specific"].get(label),
                }, 
                train_sequences = None, # self.encode_data["train_sequences_cell_specific"].get(label), 
                generated_motif = generated_celltype_motif[label], 
                generated_sequences = generated_celltype_sequences[label],
                get_kmer_metrics = self.get_kmer_metrics_labelwise,
                train_kmer_emb=self.train_kmer_emb, # if we want label-wise, need to implement train_kmer_emb_labelwise
                kmer_length= self.kmer_length
            )
            # adding to dict
            validation_metric_cell_specific["train_js_by_cell_type"].append(train_js)
            validation_metric_cell_specific["test_js_by_cell_type"].append(test_js)
            validation_metric_cell_specific["shuffle_js_by_cell_type"].append(shuffle_js)
            # validation_metric_cell_specific["edit_distance_by_cell_type"].append(min_edit_distance)
            # validation_metric_cell_specific["gc_ratio_by_cell_type"].append(gc_ratio)
            # not adding knn_distance, distance_from_closest. As we it takes too long, is not included
        self.accelerator.wait_for_everyone()
        validation_metric_cell_specific_gathered = self.accelerator.gather_for_metrics([validation_metric_cell_specific], use_gather_object=True)
        if self.accelerator.is_main_process:
            validation_metric_cell_specific_reconstructed = defaultdict(list)
            for subset_metrics in validation_metric_cell_specific_gathered:
                for metric, values in subset_metrics.items():
                    if metric in validation_metric_cell_specific_reconstructed:
                        validation_metric_cell_specific_reconstructed[metric] = values
                    else:
                        validation_metric_cell_specific_reconstructed[metric].extend(values)
            self.accelerator.print("percentage of None values in metrics:")
            for metric, value_list in validation_metric_cell_specific_reconstructed.items():
                self.accelerator.print(f"{metric} : {np.mean(np.isnan(np.array(value_list))) * 100} %")
            
            validation_metric_average = {x:np.nanmean(y) for x,y in validation_metric_cell_specific_reconstructed.items()}
            if validation_metric_average["test_js_by_cell_type"] < self.test_js_by_cell_type_global_min:
                self.test_js_by_cell_type_global_min = validation_metric_average["test_js_by_cell_type"]
                fasta_file = f"{training_data_path}/{self.run_name}synthetic_motifs.fasta"
                bed_file = f"{training_data_path}/{self.run_name}syn_results_motifs.bed"
                # force renaming even if "best" file exists
                shutil.move(fasta_file, rename_file_to_best(fasta_file, epoch))
                shutil.move(bed_file, rename_file_to_best(bed_file, epoch))
                self.save_model(epoch)
            # saving to logging dictionary

            self.accelerator.print("label-wise metrics:")
            for metric, value in validation_metric_average.items():
                self.accelerator.print(metric,":",value)
        
        
            self.log_dict.update(bulk_metrics)
            self.log_dict.update(validation_metric_average)
            self.accelerator.print(f"calculate_metrics label-wise part run {str(timedelta(seconds=time.time() - start_time))} seconds")

    
    def log_step(self, loss, epoch):
        self.loss_values.append(loss.mean().item())
        if self.accelerator.is_main_process:
            self.log_dict.update({"loss": loss.mean().item()})
            self.accelerator.log(
                self.log_dict,
                step=epoch,
            )
    def save_model(self, epoch):
        print("saving")
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(
            checkpoint_dict,
            f"checkpoints/epoch_{epoch}_{self.model_name}.pt",
        )

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]

        if self.accelerator.is_main_process:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        self.train_loop()

   
class EMA:
    # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        device = new.device
        old = old.to(device)
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 500) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def rename_file_to_best(file_path, epoch):
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    new_file_name = 'best_' + str(epoch) + '_' + file_name + file_ext

    return os.path.join(dir_name, new_file_name)


# for debuggin accelerator gather objects:
        # def ordered_print(message):
        #     rank = self.accelerator.process_index  # Get the rank of the current process
        #     world_size = self.accelerator.num_processes  # Get the total number of processes

        #     for i in range(world_size):
        #         if rank == i:
        #             print(f"Process {rank}: {' '.join([str(part) for part in message])}")
        #         # Synchronize all processes before moving to the next rank
        #         self.accelerator.wait_for_everyone()

        
        # ordered_print(["gathered_celltype_motif", type(gathered_celltype_motif), len(gathered_celltype_motif),  '\n',
        #                type(gathered_celltype_motif[1]), len(gathered_celltype_motif[1]), '\n',
        #             #    type(gathered_celltype_motif[1][0]), gathered_celltype_motif[1][0], '\n',
        #             #    type(gathered_celltype_motif[1][1]), gathered_celltype_motif[1][1]
        #             ])
        
        # ordered_print(["gathered_celltype_sequences", type(gathered_celltype_sequences), len(gathered_celltype_sequences), '\n',
        #                type(gathered_celltype_sequences[1]), len(gathered_celltype_sequences[1]), '\n',
        #             #    type(gathered_celltype_sequences[1][0]), gathered_celltype_sequences[1][0], '\n',
        #             #    type(gathered_celltype_sequences[1][1]), gathered_celltype_sequences[1][1]
        #             ])