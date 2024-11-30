from scbpnet.models import *
from dragonnfruit.models import CellStateController
import pickle

window_length = 1024
device = 'cuda'
checkpoint_path = "/nfs/turbo/umms-welchjd/mkarikom/fwd_gut_midpoint_profile_counts_scbpnet/time.03Oct2024_20.57.29.063176_neighbors.200_window.1024_convlayers.9_ctrl_layers.1_schedMetric.val_jitter.2_trim.256/ckpt_model/model_e0_b201.pth"
example_output_profile = "/nfs/turbo/umms-welchjd/mkarikom/scBPNet/examples/sample_initializing_for_guidance/example_y.pkl"
example_cell_states = "/nfs/turbo/umms-welchjd/mkarikom/scBPNet/examples/sample_initializing_for_guidance/example_c.pkl"
example_input_sequences = "/nfs/turbo/umms-welchjd/mkarikom/scBPNet/examples/sample_initializing_for_guidance/example_X.pkl"

cell_state_model = CellStateController(
		n_inputs=50, 
		n_nodes=256, 
		n_layers=1, 
		n_outputs=64,
 	)

profile_model = RegularizedDynamicBPNetProfileCountsConv(
		controller=cell_state_model,
		n_filters=512, 
		n_layers=np.log2(window_length).astype(int) - 1, 
		trimming=256, 
		dropout_rate=0.2, 
		n_outputs=2,
	)

predictor_model = scBPnetProfileCounts(profile_model, debug_strand=0, name= None)


model_state_dict = torch.load(checkpoint_path)
predictor_model.load_state_dict(model_state_dict)
predictor_model.to(device)

with open(example_input_sequences, 'rb') as f:
    X = pickle.load(f)
with open(example_output_profile, 'rb') as f:
    y = pickle.load(f)
with open(example_cell_states, 'rb') as f:
    c = pickle.load(f)

y_profile, y_counts = predictor_model(X,c)

print(f"predicted y profile: {y_profile.shape}")
print(f"predicted y counts: {y_counts.shape}")

# breakpoint()
optimizer = torch.optim.AdamW(predictor_model.parameters(), lr=0.001)
x_count_target_grad = predictor_model.attribute_count_target(X,c,optimizer,counts_target=y.sum(-1)[:,0].mean()+10)
print(f"X grad for c+100 target: {x_count_target_grad.shape}")