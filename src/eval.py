# train_eval.py
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader as GeoDataLoader
from torch import device as torch_device
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

@torch.no_grad() #disables gradient calculation
def eval(loader, model, device):
    output = [] #will store the concatenated predictions and actual values for each batch.
    smi = [] #will store the SMILES strings for each batch
    model.eval() #Sets the model to evaluation mode
    
    for data in loader: #Iterates over each batch of data provided by the DataLoader loader
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch) #is the model's output predictions
        concatenated_data = torch.cat((out, data.y.view(-1, 1)), dim=1) #Concatenates the model's predictions (out) and the actual values (data.y) along the feature dimension. This combines the predictions and actual values into a single tensor.
        
        # TypeError: can't convert cuda:0 device type tensor to numpy
        output.append(concatenated_data.cpu()) #Move the tensor to CPU and append to the output list
        smi.append(data.smiles) #Appends the SMILES strings for the current batch to the smi list.

    """Stacks the list of tensors along the batch dimension using torch.cat.
       Converts the stacked tensor to a NumPy array.
       Concatenates the list of SMILES strings into a single NumPy array."""

    stacked_output = torch.cat(output, dim=0).cpu().numpy()  # new Convert to numpy array
    stacked_smiles = np.concatenate(smi)

    """Creates a DataFrame from the stacked output and another DataFrame from the stacked SMILES strings. Concatenates these DataFrames along the feature dimension to form a single DataFrame containing predictions, actual values, and SMILES strings."""

    results = pd.concat([pd.DataFrame(stacked_output, columns=['pred', 'actual']), pd.DataFrame(stacked_smiles, columns=['smiles'])], axis=1)

    r2 = r2_score(results['actual'], results['pred'])
    print(f"The R2 score is {r2}")
    return results #Returns the results DataFrame containing predictions, actual values, and SMILES strings.

# Example usage (assuming test_loader, model, and device are defined)
# test_res = eval(test_loader, model, device)
