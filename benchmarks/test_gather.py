import torch
from accelerate import Accelerator
import pandas as pd
import numpy as np
from collections import defaultdict
import time

# Initialize Accelerator
accelerator = Accelerator()

# Set up test data for each data type scenario from your script
# data_list = [accelerator.process_index] * 5
# data_dict = {f"key_{accelerator.process_index}": [accelerator.process_index] * 5}
# data_tensor = torch.tensor([accelerator.process_index], dtype=torch.float32, device=accelerator.device)
# data_value= [accelerator.process_index]
# data = {
#     'A': np.random.randint(1, 100, 10),
#     'B': np.random.randn(10),
#     'C': np.random.choice(['X', 'Y', 'Z'], 10),
#     'D': pd.date_range('2024-01-01', periods=10, freq='D')
# }
# main_df = pd.DataFrame()
# if accelerator.is_main_process:
#     main_df = pd.DataFrame(data)

# dct = {accelerator.process_index: pd.DataFrame([accelerator.process_index] * 5)}

main_list = []
if accelerator.is_main_process:
    main_list = [1,2,3,4]
def test_gathering():
    # accelerator.print("Testing various gathering methods across processes...\n")
    # gathered_main_dict = accelerator.gather_for_metrics([dct], use_gather_object=True)
    # reconstructed_main_dict = {}
    # for item in gathered_main_dict:
    #     reconstructed_main_dict.update(item)
    # accelerator.wait_for_everyone()
    # print(f"{accelerator.process_index} gathered data (list from main process):", reconstructed_main_dict) # works
    # accelerator.wait_for_everyone()
    gathered_main_list = accelerator.gather_for_metrics(main_list, use_gather_object=True) # here necessary
    # accelerator.wait_for_everyone()
    print(f"{accelerator.process_index} gathered data (list from main process):", gathered_main_list) # works

    # Test different types of data gathering
    # 1. List of tensors
    # if accelerator.is_main_process:
    #     time.sleep(2.5)

    # gathered_data_list = accelerator.gather_for_metrics(data_list)
    # if accelerator.is_main_process:
    #     print("Gathered data (list of tensors):", gathered_data_list) # works

    # # 2. Dictionary of tensors
    # gathered_data_dict = accelerator.gather_for_metrics([data_dict], use_gather_object=True)
    # if accelerator.is_main_process:
    #     merged_dict = defaultdict(list)
    #     for d in gathered_data_dict:
    #         for k, v in d.items():
    #             merged_dict[k].append(v)
    #     print("Gathered data (dictionary of tensors):", merged_dict) # works

    # # 3. Direct tensor gathering (non-object gather)
    # gathered_data_tensor = accelerator.gather(data_tensor)
    # if accelerator.is_main_process:
    #     print("Gathered data (direct tensor):", gathered_data_tensor) # works

    # # # 4. Numpy array gathering
    # gathered_np_data = accelerator.gather_for_metrics(data_value)
    # if accelerator.is_main_process:
    #     print("Gathered data data_value:", gathered_np_data)

# Run the gathering test
test_gathering()

# Finalize
if accelerator.is_main_process:
    print("\nGathering tests completed.")
