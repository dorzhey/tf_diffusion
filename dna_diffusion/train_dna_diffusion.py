import sys
import os
import torch
from accelerate import DataLoaderConfiguration
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
import warnings
import datetime
warnings.filterwarnings("ignore")
# force other processes to wait for the validation metrics to finish calculating
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=5))

file_dir = os.path.dirname(__file__)
save_dir =  os.path.join(file_dir,'train_output')
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')


train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)

from utils import (
    js_heatmap,
    generate_heatmap,
    plot_training_loss,
    plot_training_validation
)

from utils_data import load_TF_data

from dna_tools import (
    TrainLoop
)
from dna_models import UNet, Diffusion


class ModelParameters:
    TESTING_MODE = False
    datafile = f'{data_dir}/sampled_df_columns_renamed.csv'
    device = 'cuda'
    batch_size = 720
    num_workers = 8
    seq_length = 200 #501 is top as min value in data
    subset = None
    num_epochs = 100
    log_step_show = 1
    sample_epoch = 1
    save_epoch = 5
    num_sampling_to_compare_cells = 10000
    min_sample_size = 20 # minimum sequences generated per label
    parallel_generating_bs = 1024 # used for parallel sampling, adjust based on GPU memory capabilities
    get_seq_metrics = False
    get_kmer_metrics_bulk = False
    get_kmer_metrics_labelwise = False # not implemented, takes too long
    kmer_length = 5
    run_name = '200seqlen_try2'  # for gimme scan erros when doing several runs simultaneously
    pikle_filename = "cre_expression_tf_state_200seqlength"
    load_data = True

config = ModelParameters()
    
if config.TESTING_MODE:
    # warnings.filterwarnings("default")
    os.environ['WANDB_SILENT']="true"
    os.environ["WANDB_MODE"] = "offline"
    config.load_data = True
    config.subset = 10000
    config.min_sample_size = 10
    config.seq_length = 500
    config.num_sampling_to_compare_cells = 1000
    config.pikle_filename = "cre_expression_tf_state_500seqlength_10ksubset"


data = load_TF_data(
    data_path=config.datafile,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    to_save_file_name=config.pikle_filename,
    saved_file_name=config.pikle_filename + ".pkl",
    load_saved_data=config.load_data,
)

print("loaded data")
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
dataloader_config = DataLoaderConfiguration(split_batches=False)
device_cpu = config.device == "cpu"
accelerator = Accelerator(
            # kwargs_handlers=[ddp_kwargs],
            dataloader_config = dataloader_config, 
            cpu=device_cpu, 
            # mixed_precision="fp16", 
            log_with=['wandb'])



unet = UNet(
        dim=config.seq_length,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8 # should be largest divisor of seq_length
)

diffusion = Diffusion(
    unet,
    timesteps=50,
    masking=False
)

trainloop = TrainLoop(
    config=config,
    data = data,
    model=diffusion,
    accelerator=accelerator,
    epochs=config.num_epochs,
    log_step_show=config.log_step_show,
    sample_epoch=config.sample_epoch,
    save_epoch=config.save_epoch,
    model_name="model_full_conditioning",
    num_sampling_to_compare_cells=config.num_sampling_to_compare_cells,
    run_name = config.run_name
)
trainloop.train_loop()
print("training done")

plot_training_loss(trainloop.loss_values, save_dir)
validations = [trainloop.all_train_js, trainloop.all_test_js, trainloop.all_shuffle_js]
labels = ["train JS divergence", "test JS divergence", "shuffle JS divergence"]
plot_training_validation(validations, labels, config.sample_epoch, save_dir)
print("training graphs saved")

accelerator.end_training()
# VALIDATION
# iterate over cell types keeping label ratio from train data in generated sequences

# _, generated_celltype_motif, _ = trainloop.create_sample_labelwise()


# label_list = sorted(list(data['label_ratio'].keys()), key=lambda x:x[0])
# # Generate synthetic vs synthetic heatmap
# motif_df = js_heatmap(
#     generated_celltype_motif,
#     generated_celltype_motif,
#     label_list
# )
# generate_heatmap(motif_df, "Generated", "Generated", label_list, save_dir)

# # Generate synthetic vs train heatmap
# motif_df = js_heatmap(
#     generated_celltype_motif,
#     data['train_motifs_cell_specific'],
#     label_list
# )
# generate_heatmap(motif_df, "Generated", "Train", label_list, save_dir)

# print("Finished generating heatmaps")
# accelerator.log({"Test_Train_js_heatmap":wandb.Image(f"{save_dir}/Test_Train_js_heatmap.png")})
# accelerator.log({"Generated_Train_js_heatmap":wandb.Image(f"{save_dir}/Generated_Train_js_heatmap.png")})




# code to only generate heatmaps from saved checkpoint
# trainloop.load("checkpoints/epoch_1200_model_PBMC_3k_3cluster_tcre_kmer_cluster.pt")
# diffusion = trainloop.accelerator.unwrap_model(trainloop.model)
# diffusion.eval()



# motif_df = js_heatmap(
#     data['train_motifs_cell_specific'],
#     data['train_motifs_cell_specific'],
#     label_list
# )
# generate_heatmap(motif_df, "Train", "Train", label_list, save_dir)
# motif_df = js_heatmap(
#     data['test_motifs_cell_specific'],
#     data['train_motifs_cell_specific'],
#     label_list
# )
# generate_heatmap(motif_df, "Test", "Train", label_list, save_dir)
# motif_df = js_heatmap(
#     data['train_motifs_cell_specific'],
#     data['shuffle_motifs_cell_specific'],
#     label_list
# )
# generate_heatmap(motif_df, "Train", "Shuffle", label_list, save_dir)

# accelerator.log({"Test_Train_js_heatmap":wandb.Image(f"{save_dir}/Test_Train_js_heatmap.png")})
