from utils_data import load_TF_data, load_TF_data_bidir
import os
file_dir = os.path.dirname(__file__)

data_dir = os.path.join(file_dir, '..','data_preprocessing','generated_data')
class ModelParameters:
    datafile = f'{data_dir}/sampled_df_columns_renamed_bidir.csv'
    seq_length = 512
    subset = None # rows
    pikle_filename = "cre_expression_bidir_tf_state_512seqlength_activecre"
    run_name = '512seqlength_bidir_activecre'
    train_cond = False

config = ModelParameters()
data = load_TF_data_bidir(
    data_path=config.datafile,
    seqlen=config.seq_length,
    limit_total_sequences=config.subset,
    to_save_file_name=config.pikle_filename,
    saved_file_name=config.pikle_filename + ".pkl",
    load_saved_data=False,
    train_cond = config.train_cond,
    run_name=config.run_name,
)