import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    
    Parameters:
        y_true : array-like of shape (N,)
                 Correct class indices
        y_pred : array-like of shape (N, K)
                 Predicted probabilities (rows sum ≈ 1)
                 
    Returns:
        float : average cross-entropy loss
    """
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    N = y_true.shape[0]
    
    # Select probabilities of correct classes
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute average negative log likelihood
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss