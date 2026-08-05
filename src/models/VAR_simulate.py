import numpy as np

def transition_mat(dimension: int, order: int, stable: bool | None = False,
                   alpha: float | None = None, seed: int | None = 1776):
    """
    Generate random transition matrices in the companion form for simulating
    VAR processes.
    
    Parameters
    ----------
    dimension : int
        Number of variables.
    order : int
        Order of the process.
    stable : bool | None, optional
        Whether to generate stable transition matrices. The default is False.
    alpha : float | None, optional
        Scaling factor for the eigenvalues. The default value is 0.95.
    seed : int | None, optional
        Random seed for the generator. The default is 1776.

    Raises
    ------
    ValueError
        If the scaling factor is zero or larger than 1 in absolute value.

    Returns
    -------
    J : np.Array
        Companion matrix of the VAR process.

    """
    
    if alpha is not None:
        if abs(alpha) > 1:
            raise ValueError('The scaling factor alhpa must be less than one in absolute value.')
        if alpha == 0:
            raise ValueError('The sacling factor alhpa must be non-zero.')
    else:
        alpha = 0.95

    rng = np.random.default_rng(seed=seed)
    
    ## Generate random transition matrices
    A = [rng.standard_normal((dimension,dimension)) for _ in range(order)]
    
    ## Generate the companion matrix
    J = np.zeros((dimension*order, dimension*order))
    
    ## Populate the first row of the companion matrix with transition matrices
    
    for i,A_mat in enumerate(A):
        J[:dimension, i*dimension:(i+1)*dimension] = A_mat
        
    ## Fill the bottom left block of the companion matrix with an identity matrix of the same size
    
    J[dimension:, :dimension*(order-1)] = np.identity(dimension*(order-1))
    
    ## If a stable VAR is desired, rescale the companion matrix by its spectral radius to ensure stability
    
    if stable:
        _eigvals = np.linalg.eigvals(J)
        spectral_radius = np.max(np.abs(_eigvals))
        
        J[:dimension, i*dimension:(i+1)*dimension] = J[:dimension, i*dimension:(i+1)*dimension] * alpha/spectral_radius
        
        _eigvals_stable = np.linalg.eigvals(J)
        
        assert np.all(np.abs(_eigvals_stable) < 1), "The Companion Matrix is not Stable"
        
    return J


    