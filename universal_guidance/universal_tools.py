import os
import sys
import random
from typing import Any
import shutil 
import numpy as np
import pandas as pd
import copy
import os
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
import wandb
from collections import defaultdict
from functools import partial
import time
from datetime import timedelta

random.seed(10)
torch.manual_seed(10)
np.random.seed(10)


NUM_CLASS = 3

file_dir = os.path.dirname(__file__)
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
checkpoints_dir =  os.path.join(file_dir,'diffusion_checkpoints')



universal_guide_path = os.path.join(file_dir, '..', '..','re_design', 
                                    'Universal-Guided-Diffusion', 'Guided_Diffusion_Imagenet')
if universal_guide_path not in sys.path:
    sys.path.append(universal_guide_path)

# from guided_diffusion import logger
from guided_diffusion.resample import LossAwareSampler, UniformSampler


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

from utils_data import SequenceDataset

from universal_models import EMA

class TrainLoop:
    """Class for managing the training loop for a diffusion model with universal guidance.

    This class encapsulates the entire training procedure including data loading, model optimization, sampling,
    and periodic evaluation. It supports distributed training using the Accelerator framework and integrates
    guidance mechanisms via an external guidance model.
    """

    def __init__(
        self,
        *,
        config : dict[str, Any],
        operation_config: dict[str, Any],
        model: torch.nn.Module,
        diffusion: torch.nn.Module,
        accelerator : Accelerator,
        guide_model,
        data,
        batch_size,
        lr,
        log_interval,
        sample_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        lr_anneal_steps=0,
        run_name='',
    ):
        """Initialize the TrainLoop.

        Initializes the training loop for the diffusion model with universal guidance. This includes setting up the
        data loader, optimizer, learning rate scheduler, EMA model, and various sampling and logging configurations.

        Args:
            config (dict[str, Any]): Configuration dictionary containing training hyperparameters and settings.
            operation_config (dict[str, Any]): Configuration dictionary for operational settings and guidance model parameters.
            model (torch.nn.Module): The diffusion model to be trained.
            diffusion (torch.nn.Module): Diffusion process module that defines the forward/backward diffusion steps.
            accelerator (Accelerator): Accelerator instance to facilitate distributed training.
            guide_model: The guidance model used to modify the loss function or steer the generation process.
            data: Preprocessed training data including encoded sequences and related metadata.
            batch_size (int): Batch size per process.
            lr (float): Initial learning rate.
            log_interval (int): Interval (in training steps) at which logging is performed.
            sample_interval (int): Interval (in training steps) at which samples are generated.
            save_interval (int): Interval (in training steps) at which the model is checkpointed.
            resume_checkpoint: Path to a checkpoint file for resuming training, if applicable.
            schedule_sampler (optional): Sampler for selecting diffusion timesteps; defaults to UniformSampler if not provided.
            lr_anneal_steps (int, optional): Total number of steps over which to anneal the learning rate.
            run_name (str, optional): Identifier for the current run used for logging and file naming.

        Returns:
            None
        """

        self.accelerator = accelerator
        self.config = config
        self.train_cond = config.train_cond
        self.encode_data = data
        labels = torch.from_numpy(self.encode_data["x_train_cell_type"]).int() if self.train_cond else None
        seq_dataset = SequenceDataset(seqs=torch.from_numpy(self.encode_data["X_train"]).float(), 
                                      c=labels, include_labels = self.train_cond, transform_ddsm = False)
        data_loader = DataLoader(seq_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    
        self.lr = lr
        self.current_lr = lr
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = batch_size * self.accelerator.num_processes
        self.run_name = run_name # for gimme scan erros when doing several runs simultaneously 

        self.device = self.accelerator.device
        optimizer = AdamW(
            model.parameters(), lr=self.lr, weight_decay=config.weight_decay
        )
        
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(model, optimizer, data_loader)
        self.diffusion = diffusion
        # self.master_params = list(model.parameters())
        # self.state_dict = list(self.accelerator.get_state_dict(self.model))
        self.operation_config = operation_config
        self.operation_config.loss_func = guide_model
        
        self.run_dataloader = self._run_dataloader() 
        # if self.resume_step:
        #     self._load_optimizer_state()
        # self.ema_rate = config.ema_rate
        
        # self.ema = EMA(config.ema_rate)
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            
        # sampling
        self.parallel_generating_bs = config.parallel_generating_bs
        
        self.seq_length = config.seq_length
        self.get_seq_metrics = config.get_seq_metrics
        self.get_kmer_metrics_bulk = config.get_kmer_metrics_bulk
        self.get_kmer_metrics_labelwise = config.get_kmer_metrics_labelwise
        self.num_cre_counts_per_cell_type = config.num_cre_counts_per_cell_type
        # self.num_sampling_to_compare_cells = config.num_sampling_to_compare_cells
        self.sampling_subset_random = config.sampling_subset_random
        # self.min_sample_size = config.min_sample_size
        self.kmer_length = config.kmer_length
    
        if self.get_kmer_metrics_bulk:
            self.accelerator.print("precomputing kmer embeddings")
            self.train_kmer_emb = compute_kmer_embeddings(self.encode_data["train_sequences"], k=self.kmer_length)
        else:
            self.train_kmer_emb = None
        
        # Metrics
        # self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        # self.seq_similarity = 1
        # self.min_edit_distance, self.knn_distance, self.distance_from_closest = 100, 0, 0
        # self.gc_ratio = 0.5
        # for plots
        self.test_js_by_cell_type_global_min = 1
        self.loss_values = []
        self.all_train_js, self.all_test_js, self.all_shuffle_js = [],[],[]
        self.all_seq_similarity = []
        self.avg_loss = []
        self.log_dict = {}

    def run_loop(self):
        """Run the main training loop.

        Iterates through the training steps, performing forward and backward passes, sampling, logging, and checkpointing.
        If the current process is the main process, it initializes tracking (e.g., via wandb) and logs metrics.

        Returns:
            None
        """

        if self.accelerator.is_main_process:
            # Initialize wandb
            wandb_config = self.config.to_dict()#{"learning_rate": config.lr, "seq_length": config.seq_length, "batch_size": config.batch_size}
            wandb_config.update(self.operation_config.to_dict())
            # logger.configure()
            self.accelerator.init_trackers("universal_diffusion", config= wandb_config )
        # if self.lr_anneal_steps:
        self.accelerator.wait_for_everyone()
        for step in tqdm(range(self.resume_step, self.lr_anneal_steps + 1)):
            self.step = step
            batch = next(self.run_dataloader)
            self.run_step(batch)

            # if self.step> 0 and self.step % self.save_interval == 0 and self.accelerator.is_main_process:
            #     self.save_model()
            #     # Run for a finite amount of time in integration tests.
            #     if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
            #         return
            if step> 0 and step % self.sample_interval == 0:
                self.initialize_sampling()
                generated_bulk_motif, generated_celltype_motif, generated_celltype_sequences = self.create_sample_labelwise_parallel_dynamic()
                self.calculate_metrics_labelwise_parallel(generated_bulk_motif, generated_celltype_motif, generated_celltype_sequences)             
            if step % self.log_interval == 0 and self.accelerator.is_main_process:
                self.log_dict.update({"loss": sum(self.avg_loss)/len(self.avg_loss), "lr":self.current_lr, 
                                      "samples":(step + self.resume_step + 1) * self.global_batch})
                self.accelerator.log(
                    self.log_dict,
                    step=step,
                )
                self.avg_loss = []
                # logger.dumpkvs()
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save_model()

    def run_step(self, batch):
        """Perform a single training step.

        Processes a single batch from the dataloader, computes the diffusion loss using the current model and timestep,
        and updates the model parameters via backpropagation and an optimizer step. Optionally, updates the learning rate
        by annealing.

        Args:
            batch: A batch of training data. If training is conditional, this is expected to be a tuple of (data, condition).

        Returns:
            None
        """

        self.optimizer.zero_grad()
        if self.train_cond:
            batch, cond = batch
            cond = {"y":cond}
        else:
            cond = None
        
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        
        with self.accelerator.autocast():
            losses = self.diffusion.training_losses(
                self.model,
                batch,
                t,
                model_kwargs=cond,
            )
                
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        
        loss = (losses["loss"] * weights).mean()
        self.accelerator.backward(loss)
        self.optimizer.step()
        

        # self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))
        self.current_lr = self._anneal_lr()
        
        self.avg_loss.append(self.accelerator.gather(loss).mean().item())

    
    def initialize_sampling(self):
        """Initialize sampling parameters for generation.

        Randomly subsamples transcription factor clusters from the training data and assigns a subset to the current process.
        Prepares the label array for use during the sampling/generation phase.

        Returns:
            None
        """


        all_tf_clusters = list(self.encode_data['bidir_cre_dict'].keys())

        # randomly subsample from all tf_cluster
        subsampled_tf_clusters_ = []
        if self.accelerator.is_main_process:
            min_len = min(self.sampling_subset_random, len(all_tf_clusters))
            subsampled_tf_clusters_ = random.sample(all_tf_clusters, min_len)
        self.subsampled_tf_clusters = self.accelerator.gather_for_metrics(subsampled_tf_clusters_, use_gather_object=True)

        start_index, end_index = calculate_start_end_indices(len(self.subsampled_tf_clusters), self.accelerator.num_processes, self.accelerator.process_index)
        self.process_tf_clusters = self.subsampled_tf_clusters[start_index:end_index]

        extended_label_array = []
        for tf_cluster in self.process_tf_clusters:
            # get expression levels from both positive and negative strands
            cre_expression_levels = self.encode_data['bidir_cre_dict'][tf_cluster].tolist()
            min_size = min(self.num_cre_counts_per_cell_type, len(cre_expression_levels))
            cre_expression_levels = random.sample(cre_expression_levels, min_size)
            
            extended_label_array.extend([(tf_cluster, *bidir) for bidir in cre_expression_levels])
        # # shuffle the process label array to ensure randomness
        # random.shuffle(extended_label_array)

        self.process_label_array = extended_label_array

        print(f"Process {self.accelerator.process_index} handles {len(extended_label_array)} samples.")

    def create_sample_labelwise_parallel_dynamic(self):
        """Generate synthetic samples and extract motifs in parallel.

        Generates synthetic DNA sequences using the diffusion model with universal guidance by sampling in batches.
        Converts the generated images to nucleotide sequences, aggregates them in FASTA format, and extracts motif counts.
        Gathers and reconstructs cell-type specific motif counts and generated sequences across distributed processes.

        Returns:
            tuple: A tuple containing:
                - generated_bulk_motif (pandas.DataFrame): Aggregated motif counts for all generated sequences.
                - reconstructed_celltype_motif (dict): Dictionary mapping each cell type to its motif counts DataFrame.
                - reconstructed_celltype_sequences (dict): Dictionary mapping each cell type to a list of generated DNA sequences.
        """

        self.accelerator.print(f"Sampling at step {self.step}")
        if self.operation_config.guidance_3:
            self.operation_config.loss_func.predictor_model.to(self.device)
        # model = self.accelerator.unwrap_model(self.model)
        self.model.eval()
        # model.requires_grad_(False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        nucleotides = ["A", "C", "G", "T"]
        # process_run_name = self.run_name + str(self.accelerator.process_index)
        bulk_fasta_sequences = []
        generated_celltype_sequences = {}
        num_batches = len(self.process_label_array) // self.parallel_generating_bs
        config = self.config
        guide_loss_sample = []
        for i in tqdm(range(num_batches + 1), desc="generating"):
            batch_labels = self.process_label_array[i * self.parallel_generating_bs:(i + 1) * self.parallel_generating_bs]
            if not batch_labels:
                continue
            conditions =  np.array([
                np.append(self.encode_data['tf_cluster_state_dict'][tf_cluster], [pos_cre_counts, neg_cre_counts]) 
                for tf_cluster, pos_cre_counts, neg_cre_counts in batch_labels
            ])
            conditions_tensor = torch.as_tensor(conditions, dtype=torch.float32, device=self.device)
            label_tensor = {}
            if config.train_cond:
                label_tensor['y'] = torch.tensor(np.array([tf_cluster for tf_cluster, _,__ in batch_labels]), dtype=torch.int, device=self.device)
            
            sampled_images, guide_loss = self.diffusion.ddim_sample_loop_operation(
                self.model,
                (len(conditions), 1, 4, config.image_size),
                operated_image=conditions_tensor,
                operation=self.operation_config,
                clip_denoised=config.clip_denoised,
                model_kwargs= label_tensor,
                cond_fn=None,
                device=self.device,
                progress=False,
            )
            guide_loss_sample.append(guide_loss)
            for j, image in enumerate(sampled_images.squeeze(1)):
                sequence = "".join([nucleotides[s] for s in np.argmax(image.cpu(), axis=0)]) #image shape 4 x seq_len
                # header >seq_id_tf_cluster, pos_cre_counts, neg_cre_counts
                tf_cluster = batch_labels[j][0]
                header = f">seq_id_{tf_cluster}_{batch_labels[j][1]}_{batch_labels[j][2]}_{i}_{j}"
                fasta_sequence = f"{header}\n{sequence}"
                bulk_fasta_sequences.append(fasta_sequence)
                if tf_cluster not in generated_celltype_sequences:
                    generated_celltype_sequences[tf_cluster] = [sequence]
                else:
                    generated_celltype_sequences[tf_cluster].append(sequence)

        gathered_guide_loss = self.accelerator.gather_for_metrics(guide_loss_sample, use_gather_object=True)
        if self.accelerator.is_main_process:
            self.log_dict.update({"guide_loss":sum(gathered_guide_loss)/len(gathered_guide_loss)})

        gathered_bulk_fasta_sequences = self.accelerator.gather_for_metrics(bulk_fasta_sequences, use_gather_object=True)
    
        gathered_celltype_sequences = self.accelerator.gather_for_metrics([generated_celltype_sequences], use_gather_object=True)
        
        reconstructed_celltype_sequences = {}
        for item in gathered_celltype_sequences:
            reconstructed_celltype_sequences.update(item)
    
        # now we want to extract_motifs, but only once
        generated_bulk_motif = pd.DataFrame()
        constructed_celltype_motif = {}

        if self.accelerator.is_main_process:
            generated_bulk_motif_table = extract_motifs(gathered_bulk_fasta_sequences, self.run_name, save_bed_file=True, get_table=True)
            # now reconstruct the cell_tyoe motif counts from a table
            generated_bulk_motif, constructed_celltype_motif = self.bulk_motif_to_celltype(generated_bulk_motif_table)

        # reconstructed_bulk_motif = self.accelerator.gather_for_metrics([generated_bulk_motif], use_gather_object=True)[0]
        reconstructed_celltype_motif_ = self.accelerator.gather_for_metrics([constructed_celltype_motif], use_gather_object=True)

        reconstructed_celltype_motif = {}
        for item in reconstructed_celltype_motif_:
            reconstructed_celltype_motif.update(item)
        
        self.model.train()
        if self.operation_config.guidance_3:
            self.operation_config.loss_func.predictor_model.to("cpu")
        # self.model.requires_grad_(True)
        # for param in self.model.parameters():
        #     param.requires_grad = True
        
        return generated_bulk_motif, reconstructed_celltype_motif, reconstructed_celltype_sequences
    
    def calculate_metrics_labelwise_parallel(self, generated_motif, generated_celltype_motif, generated_celltype_sequences):
        """Calculate and log label-wise validation metrics in parallel.

        Computes various similarity and motif validation metrics (e.g., Jensen–Shannon divergence, GC content ratio,
        edit distance, k-nearest neighbor distances) between generated and training sequences. Aggregates metrics
        across distributed processes, logs the results, and updates the model checkpoint if an improvement is observed.

        Args:
            generated_motif (pandas.DataFrame): Bulk motif counts from generated sequences.
            generated_celltype_motif (dict): Dictionary of cell-type specific generated motif counts.
            generated_celltype_sequences (dict): Dictionary of cell-type specific generated DNA sequences.

        Returns:
            None
        """

        self.accelerator.print("calculating metrics at step", self.step)
        device = self.device
        start_time = time.time()
        local_generated_sequences = [sequence for tf_cluster, sequences in generated_celltype_sequences.items() if tf_cluster in self.process_tf_clusters for sequence in sequences]
        local_train_sequences = [sequence for tf_cluster, sequences in self.encode_data['train_sequences_cell_specific'].items() if tf_cluster in self.process_tf_clusters for sequence in sequences]

        local_max_similarity = calculate_similarity_metric(local_train_sequences, local_generated_sequences, self.seq_length, device)

        all_max_similarities = self.accelerator.gather_for_metrics([local_max_similarity])

        if self.accelerator.is_main_process:
            seq_similarity = sum(all_max_similarities) / self.accelerator.num_processes / self.seq_length
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
        gc_ratio = self.accelerator.gather_for_metrics([gc_ratio_local])
        min_edit_distance = self.accelerator.gather_for_metrics([min_edit_distance_local])
        knn_distance = self.accelerator.gather_for_metrics([knn_distance_local])
        distance_from_closest = self.accelerator.gather_for_metrics([distance_from_closest_local])

        if self.accelerator.is_main_process:
            # Calculate final averages on the main process
            gc_ratio = sum(gc_ratio)/len(gc_ratio)
            min_edit_distance = sum(min_edit_distance)/len(min_edit_distance)
            knn_distance = sum(knn_distance)/len(knn_distance)
            distance_from_closest = sum(distance_from_closest)/len(distance_from_closest)

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
            # delete unwanted metrics
            for metric in metrics_delete:
                bulk_metrics.pop(metric, None)

            self.accelerator.print(f"calculate_metrics bulk part run {str(timedelta(seconds=time.time() - start_time))} seconds")
            for metric, value in bulk_metrics.items():
                self.accelerator.print(metric,":",value)
        # now, generate metrics cell_type-wise per process and then gather and average it
        start_time = time.time()
        validation_metric_cell_specific = defaultdict(list)
        for tf_cluster in self.process_tf_clusters:            
            train_js, test_js, shuffle_js, gc_ratio, min_edit_distance, knn_distance, distance_from_closest = calculate_validation_metrics(
                motif_data = {
                "train_motifs":self.encode_data["train_motifs_cell_specific"].get(tf_cluster), 
                "test_motifs": self.encode_data["test_motifs_cell_specific"].get(tf_cluster),
                "shuffle_motifs":self.encode_data["shuffle_motifs_cell_specific"].get(tf_cluster),
                },
                train_sequences = None, # self.encode_data["train_sequences_cell_specific"].get(label), 
                generated_motif = generated_celltype_motif.get(tf_cluster), 
                generated_sequences = generated_celltype_sequences.get(tf_cluster),
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
                shutil.move(fasta_file, rename_file_to_best(fasta_file, self.step))
                shutil.move(bed_file, rename_file_to_best(bed_file, self.step))
                self.save_model()
            # saving to logging dictionary

            self.accelerator.print("label-wise metrics:")
            for metric, value in validation_metric_average.items():
                self.accelerator.print(metric,":",value)
        
        
            self.log_dict.update(bulk_metrics)
            self.log_dict.update(validation_metric_average)
            self.accelerator.print(f"calculate_metrics label-wise part run {str(timedelta(seconds=time.time() - start_time))} seconds")

    # def model_fn(self, x, t, y=None):
    #     return self.model(x, t, y if self.config.class_cond else None)

    def bulk_motif_to_celltype(self, df_motifs):
        """Convert a bulk motif table into cell-type specific motif counts.

        Extracts the transcription factor cluster identifier from sequence IDs in the motif table and aggregates motif counts
        for each cell type. Also computes the overall bulk motif counts across all sequences.

        Args:
            df_motifs (pandas.DataFrame): DataFrame containing motif counts with sequence IDs as its index.

        Returns:
            tuple: A tuple containing:
                - bulk_motifs (pandas.DataFrame): Aggregated motif counts for all sequences.
                - generated_celltype_motif (dict): Dictionary mapping each cell type to its corresponding motif counts DataFrame.
        """

        # we take table sequence_id as 0 column x motif name as first row
        generated_celltype_motif = {}
        # cut the sequence id to get tf_cluster
        idx_to_label = {sequence_id: list(sequence_id.split('_'))[2] for sequence_id in df_motifs.index.values}
           
        for tf_cluster in self.subsampled_tf_clusters:
            idx_subset = [idx for idx, label in idx_to_label.items() 
                          if int(label) == tf_cluster]
            subset_motif = df_motifs[df_motifs.index.isin(idx_subset)]
            # turn to motif name as index and then sum, getting motif counts
            subset_motif = pd.DataFrame(subset_motif.T.sum(axis=1))
            # drop empty motifs
            subset_motif = subset_motif[subset_motif[0] != 0]
            generated_celltype_motif[tf_cluster] = subset_motif
        bulk_motifs = pd.DataFrame(df_motifs.T.sum(axis=1))

        return bulk_motifs, generated_celltype_motif
    

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def _run_dataloader(self):
        while True:
            yield from self.train_dl

        
    def save_model(self):
        print("saving")
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(
            checkpoint_dict,
            f"diffusion_checkpoints/step_{self.step}_{self.run_name}.pt",
        )

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.resume_step = checkpoint_dict["step"]
        if self.accelerator.is_main_process:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        self.run_loop()
        
def dict_to_wandb(losses):
    result = {}
    for key, values in losses.items():
        result[key] = values.mean().item()

    return result

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

# def log_loss_dict(diffusion, ts, losses):
#     for key, values in losses.items():
#         logger.logkv_mean(key, values.mean().item())
#         # Log the quantiles (four quartiles, in particular).
#         for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
#             quartile = int(4 * sub_t / diffusion.num_timesteps)
#             logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def rename_file_to_best(file_path, step):
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    new_file_name = 'best_' + str(step) + '_' + file_name + file_ext

    return os.path.join(dir_name, new_file_name)

class OperationArgs:
    """Container for operation-specific arguments and settings for universal guidance.

    This class holds configuration parameters used in the guidance operation, including flags for enabling universal
    guidance, guidance loss functions, optimizer parameters, learning rate, and other hyperparameters.

    Class Attributes:
        guidance_3 (bool): Flag indicating whether to use universal guidance.
        operation_func: Function representing the guidance model.
        loss_func: Loss function for the guidance operation (to be specified later).
        optim_guidance_3_wt (float): Weight for the guidance optimizer.
        num_steps (list): List specifying the number of steps for the guidance operation.
        max_iters (int): Maximum number of iterations.
        loss_cutoff (float): Threshold for loss cutoff.
        optimizer: Optimizer to be used (e.g., 'Adam').
        lr_scheduler: Learning rate scheduler (e.g., 'CosineAnnealingLR').
        lr (float): Learning rate.
        tv_loss (int): Total variation loss parameter.
        warm_start (bool): Flag indicating whether to perform a warm start.
        old_img: Placeholder for a previous image, if applicable.
        fact (float): Scaling factor used in the operation.
        sampling_type: Specifies the type of sampling (e.g., ddpm, fully_random, or None).

    Methods:
        to_dict: Returns a dictionary representation of the operation arguments.
    """

    guidance_3 = False # universal guidance or not
    # original_guidance = False # whether to use original classifier guidance with cond_fn

    operation_func = None # guidance model
    loss_func = None # guidance loss func, specified later

    optim_guidance_3_wt = 0.0001
    num_steps = [1]
    
    max_iters = 0
    loss_cutoff = 1.0
    optimizer = None # 'Adam'
    lr_scheduler = None #'CosineAnnealingLR'
    lr = 1e-2
    tv_loss = 10
    
    warm_start = False
    old_img = None
    fact = 0.5
    sampling_type = None # ddpm, fully_random or None

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'to_dict'}




