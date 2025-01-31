# This is a code for A framework for enhancing multi-label deep learning models via oversampling data

### requirements.txt include all package needed

### function.py include some class to calculate the imbalance of the dataset

### Model.py include DELA and CLIF base classifier， and Layers.py and util.py as a supplement



## Main Functions of Label refresh and label mask

### Class: `PLM`

The `PLM` class is responsible for updating class ratios based on the divergence between predicted and actual labels.

#### Attributes:
- `r_c`: A list or array representing the class ratios.
- `lambda_param`: A parameter controlling the rate of update for the class ratios.

#### Methods:
- **`__init__(self, r_c, lambda_param)`**: Initializes the `PLM` class with the given class ratios and lambda parameter.
- **`update_ratios(self, labels, predictions)`**: Updates the class ratios based on the Kullback-Leibler (KL) divergence between the predicted and actual label distributions. The method calculates the divergence for both positive and negative samples, normalizes these divergences, and updates the class ratios accordingly.

### Function: `update_H`

This function updates the prediction history `H` for each sample based on the current predictions.

#### Parameters:
- `H`: A dictionary storing the prediction history for each sample.
- `y_pred`: The current predictions from the model.
- `ids`: The indices of the samples in the current batch.
- `max_history_length`: The maximum length of the history to store for each sample.

#### Returns:
- `H`: The updated prediction history.

### Function: `update_E`

This function updates the entropy matrix `E` based on the prediction history stored in `H`.

#### Parameters:
- `H`: The prediction history dictionary.
- `E`: The entropy matrix to be updated.
- `ids`: The indices of the samples in the current batch.
- `label_dim`: The number of labels (or classes) in the dataset.

#### Returns:
- `E`: The updated entropy matrix.

### Function: `update_dataset_label`

This function updates the dataset labels based on the entropy matrix `E` and a given threshold.

#### Parameters:
- `E`: The entropy matrix.
- `dataset`: The dataset containing the features and labels.
- `ids`: The indices of the samples in the current batch.
- `label_dim`: The number of labels (or classes) in the dataset.
- `threshold`: The threshold value for updating the labels.

#### Returns:
- `updated_dataset`: The dataset with updated labels.

## Usage Example

1. **Initialize the PLM class**:
   ```python
   r_c = [0.5, 0.5]  # Initial class ratios
   lambda_param = 0.1  # Update rate parameter
   plm = PLM(r_c, lambda_param)
   ```

2. **Update class ratios**:
   ```python
   labels = np.array([[1, 0], [0, 1], [1, 1]])  # Actual labels
   predictions = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])  # Predicted labels
   updated_ratios = plm.update_ratios(labels, predictions)
   ```

3. **Update prediction history**:
   ```python
   H = {}  # Initialize prediction history
   y_pred = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])  # Current predictions
   ids = [0, 1, 2]  # Sample indices
   H = update_H(H, y_pred, ids, max_history_length=5)
   ```

4. **Update entropy matrix**:
   ```python
   E = np.zeros((3, 2))  # Initialize entropy matrix
   E = update_E(H, E, ids, label_dim=2)
   ```

5. **Update dataset labels**:
   ```python
   dataset = TensorDataset(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), torch.tensor([[1, 0], [0, 1], [1, 1]]))
   threshold = 0.5  # Entropy threshold
   updated_dataset = update_dataset_label(E, dataset, ids, label_dim=2, threshold=threshold)
   ```

## Notes

- The `PLM` class is designed to work with binary classification problems, but it can be extended to multi-class problems with appropriate modifications.
- The `update_H`, `update_E`, and `update_dataset_label` functions are designed to work together to dynamically update the dataset labels based on the model's prediction history and entropy calculations.
- Ensure that the `labels` and `predictions` arrays are properly aligned and have the same shape when using the `update_ratios` method.

