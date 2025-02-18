import os
import sys

from accelerate import DataLoaderConfiguration
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
import torch
import datetime
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(days=5))
file_dir = os.path.dirname(__file__)
save_dir =  os.path.join(file_dir,'train_output')
data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')

train_utils_path = os.path.join(file_dir,'..','train_utils')
if train_utils_path not in sys.path:
    sys.path.append(train_utils_path)
from utils import (
    plot_training_loss,
    plot_training_validation,
    compare_motif_list
)
from utils_data import load_TF_data, load_TF_data_bidir

universal_guide_path = os.path.join(file_dir, '..', '..','re_design', 
                                    'Universal-Guided-Diffusion', 'Guided_Diffusion_Imagenet')
if universal_guide_path not in sys.path:
    sys.path.append(universal_guide_path)

# from guided_diffusion import logger
from guided_diffusion.resample import create_named_schedule_sampler

from universal_models import (
    create_model_and_diffusion,
    scBPnetGuide,
)

from universal_tools import (
    OperationArgs,
    TrainLoop
)

class DiffusionTrainingConfig:
    datafile=f'{data_dir}/sampled_df_columns_renamed_bidir.csv'
    device = 'cuda'
    pikle_filename = "cre_expr_bidir_512seqlength_classcond_12clusters"
    run_name = "512seqlength_bidir_classcond_12clusters" # for gimme scan erros when doing several runs simultaneously 
    guide_checkpoint = "guide_checkpoints/model_e9_b9901.pth"
    load_data = True
    subset = None
    num_workers = 4
    seq_length=512
    
    batch_size=256
    train_cond = True # whether to train conditionally
    n_classes=469
    lr=1e-4
    weight_decay=0.15
    lr_anneal_steps=10000
    
    log_interval=20
    sample_interval=20
    save_interval=1000000
    resume_checkpoint=""
    # use_fp16=False
    # fp16_scale_growth=1e-3

    # Sampling
    # use_ddim=False
    clip_denoised=True
    
    sampling_subset_random = 50 # number of cell types to subset for faster sampling
    num_cre_counts_per_cell_type = 500
    parallel_generating_bs = 256 # used for parallel sampling, adjust based on GPU memory capabilities
    get_seq_metrics = False
    get_kmer_metrics_bulk = False
    get_kmer_metrics_labelwise = False # not implemented, takes too long
    kmer_length = 5   

    # model
    class_cond=train_cond
    use_checkpoint=False # not implemented
    image_size=seq_length
    ema_rate=0.99
    num_channels=128
    num_res_blocks=2
    num_heads=2
    num_heads_upsample=-1
    num_head_channels=-1
    attention_resolutions=""
    # channel_mult=""
    dropout=0.15
    
    use_scale_shift_norm=True
    resblock_updown=True
    use_new_attention_order=False

    # Diffusion
    schedule_sampler="uniform" # or "loss-second-moment"
    learn_sigma=True
    diffusion_steps=100
    noise_schedule="linear" # or "cosine"
    timestep_respacing=""
    sigma_small=False
    use_kl=True
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=False

    # to send config to wandb
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'to_dict'}

# os.environ['WANDB_SILENT']="true"
# os.environ["WANDB_MODE"] = "offline"

config = DiffusionTrainingConfig()

data = load_TF_data_bidir(
    data_path=config.datafile,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    to_save_file_name=config.pikle_filename,
    saved_file_name=config.pikle_filename + ".pkl",
    load_saved_data=config.load_data,
    train_cond = config.train_cond,
    run_name = config.run_name,
)
print("loaded data")

dataloader_config = DataLoaderConfiguration(split_batches=False)
accelerator = Accelerator(
            dataloader_config = dataloader_config, 
            cpu= (config.device == "cpu"), 
            mixed_precision= None, 
            log_with=['wandb'])

accelerator.print("train-test JSD", compare_motif_list(data['train_motifs'],data['test_motifs']))
accelerator.print("train-shuffle JSD", compare_motif_list(data['train_motifs'],data['shuffle_motifs']))
accelerator.print("test-shuffle JSD", compare_motif_list(data['test_motifs'],data['shuffle_motifs']))

guide_model =  scBPnetGuide(config.guide_checkpoint)


model, diffusion = create_model_and_diffusion(config)
schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, diffusion)

operation_config = OperationArgs()

trainloop = TrainLoop(
    config=config,
    operation_config=operation_config,
    model=model,
    diffusion=diffusion,
    accelerator=accelerator,
    guide_model=guide_model,
    data=data,
    batch_size=config.batch_size,
    lr=config.lr,
    log_interval=config.log_interval,
    sample_interval = config.sample_interval,
    save_interval=config.save_interval,
    resume_checkpoint=config.resume_checkpoint,
    schedule_sampler=schedule_sampler,
    lr_anneal_steps=config.lr_anneal_steps,
    run_name = config.run_name,
)

trainloop.run_loop()
plot_training_loss(trainloop.loss_values, save_dir)
validations = [trainloop.all_train_js, trainloop.all_test_js, trainloop.all_shuffle_js, trainloop.all_seq_similarity]
labels = ["train JS divergence", "test JS divergence", "shuffle JS divergence", "Similarity"]
plot_training_validation(validations, labels, config.sample_interval, save_dir)
print("training graphs saved")
accelerator.end_training()
