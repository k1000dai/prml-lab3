import numpy as np
import numpy.matlib
import scipy

class GaussD:
    """
    Represents a Gaussian probability distribution for a random vector.

    This class can model a Gaussian distribution with either:
    1. Statistically independent components (diagonal covariance matrix).
    2. Correlated components (full covariance matrix).

    Attributes:
        means (np.ndarray): The mean vector of the Gaussian distribution. Shape (dataSize,).
        stdevs (np.ndarray): Standard deviations of the components.
                             If covariance is diagonal, these are the sqrt of diagonal elements.
                             If covariance is full, these are sqrt of eigenvalues of covariance matrix.
                             Shape (dataSize,).
        cov (np.ndarray): The covariance matrix. Shape (dataSize, dataSize).
        variance (np.ndarray): Variance of each component (stdevs**2). Shape (dataSize,).
        dataSize (int): The dimensionality of the random vector.
        covEigen (np.ndarray or int): Eigenvectors of the covariance matrix if it's full,
                                      or 1 if it's diagonal (implicitly identity eigenvectors scaled by stdevs).
    """
    def __init__(self, means: np.ndarray, stdevs: np.ndarray = None, cov: np.ndarray = None):
        """
        Initializes a GaussD object.

        Args:
            means (np.ndarray): The mean vector. Shape (dataSize,).
            stdevs (np.ndarray, optional): Standard deviations for each component.
                                           Required if `cov` is not provided (for diagonal covariance).
                                           Shape (dataSize,). Defaults to None.
            cov (np.ndarray, optional): The full covariance matrix.
                                        If provided, `stdevs` will be derived from its eigenvalues.
                                        Shape (dataSize, dataSize). Defaults to None.

        Raises:
            ValueError: If neither `stdevs` nor `cov` is provided, or if shapes are inconsistent.
        """
        self.means = np.array(means, dtype=float)
        self.dataSize = len(self.means)

        if cov is None:
            if stdevs is None:
                raise ValueError("Either stdevs or cov must be provided.")
            self.stdevs = np.array(stdevs, dtype=float)
            if len(self.stdevs) != self.dataSize:
                raise ValueError("Means and stdevs must have the same length.")
            
            self.variance = self.stdevs**2
            # For diagonal covariance, cov is an identity matrix scaled by variance
            self.cov = np.eye(self.dataSize) * self.variance
            # covEigen = 1 signifies diagonal covariance (eigenvectors are standard basis vectors)
            self.covEigen = 1
        else:
            self.cov = np.array(cov, dtype=float)
            if self.cov.shape != (self.dataSize, self.dataSize):
                raise ValueError("Covariance matrix shape must match (dataSize, dataSize).")
            
            # Symmetrize covariance for numerical stability before eigendecomposition
            symmetric_cov = 0.5 * (self.cov + self.cov.T)
            eigenvalues, eigenvectors = np.linalg.eig(symmetric_cov)
            
            # Eigenvalues represent variances along principal components
            # Ensure eigenvalues are non-negative before sqrt due to potential float inaccuracies
            self.stdevs = np.sqrt(np.maximum(eigenvalues, 0))
            self.variance = self.stdevs**2 # Variances along principal components
            self.covEigen = eigenvectors
    
   
    def rand(self, nData: int) -> np.ndarray:
        """
        Generates random vectors drawn from this Gaussian distribution.

        Args:
            nData (int): The number of random data vectors to generate.

        Returns:
            np.ndarray: A matrix where each column is a random vector.
                        Shape (dataSize, nData).
        """
        # Generate standard normal random numbers
        R_std_normal = np.random.randn(self.dataSize, nData) # Shape (dataSize, nData)
        
        # Scale by standard deviations (or sqrt of eigenvalues if full cov)
        # This step effectively colors the noise if covariance is diagonal
        R_scaled = np.diag(self.stdevs) @ R_std_normal
        
        # If covariance is full, rotate by eigenvectors to introduce correlations
        if not isinstance(self.covEigen, int): # covEigen is matrix of eigenvectors
            R_correlated = self.covEigen @ R_scaled
        else: # Diagonal covariance, R_scaled is already correct
            R_correlated = R_scaled

        # Add mean vector (broadcasted to each column)
        R_final = R_correlated + self.means.reshape(-1, 1) # self.means to (dataSize, 1)

        return R_final
    
    def init(self, data: np.ndarray) -> 'GaussD':
        """
        Initializes the GaussD object's parameters (mean, covariance, etc.) from data.

        Args:
            data (np.ndarray): A NumPy array where rows are observations and columns are features.
                               Shape (nSamples, dataSize).

        Returns:
            GaussD: The instance itself, with updated parameters.

        Raises:
            TypeError: If `data` is not a NumPy array.
            ValueError: If `data` contains less than one observation or has inconsistent dimensions.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input 'data' must be a NumPy array.")

        if data.ndim == 1:
            # If data is 1D (e.g., shape (N,)), treat as N observations of 1 feature.
            data = data.reshape(-1, 1) # Reshape to (N, 1)
        
        if data.shape[0] < 1: # nSamples must be at least 1
            raise ValueError("Input 'data' must contain at least one observation.")
            
        # Calculate sample mean for each feature (column-wise mean)
        self.means = np.mean(data, axis=0) # Shape (dataSize,)
        
        # Update dataSize based on the number of features in the data
        self.dataSize = data.shape[1]
        if self.means.shape[0] != self.dataSize : # Should match if data.shape[1] was used for dataSize
             raise ValueError("Mismatch between data columns and inferred dataSize.")


        # Calculate sample covariance matrix.
        # rowvar=False: each column represents a variable, while rows contain observations.
        # ddof=1 (default) for unbiased estimate, but can lead to NaN if data.shape[0] == 1.
        # If only one sample, covariance is undefined or typically set to zero/small value.
        # For robustness with few samples, one might use ddof=0 (bias=True).
        # Here, we use default behavior of np.cov.
        if data.shape[0] > 1:
            self.cov = np.cov(data, rowvar=False)
        else: # Single sample: covariance is zero or small positive diagonal.
              # np.cov with one sample and ddof=1 yields NaNs.
              # Let's define it as zero matrix for a single sample.
            self.cov = np.zeros((self.dataSize, self.dataSize))


        # Ensure covariance is a 2D matrix, even for a single feature.
        # np.cov(data_with_1_feature, rowvar=False) returns a 1x1 matrix.
        # If somehow it became scalar (e.g. if dataSize was 0 and then set to 1), ensure it's 2D.
        if self.cov.ndim == 0: 
            self.cov = np.array([[self.cov]])
        
        # Update related attributes based on the new covariance matrix
        if self.dataSize == 0 : # Should not happen if data is valid
            self.stdevs = np.array([])
            self.variance = np.array([])
            self.covEigen = 1 # Default for empty case
            return self

        # Symmetrize covariance matrix before eigendecomposition for numerical stability
        symmetric_cov = 0.5 * (self.cov + self.cov.T)
        eigenvalues, eigenvectors = np.linalg.eig(symmetric_cov)
        
        # Eigenvalues represent variances along principal components.
        # Ensure eigenvalues are non-negative before sqrt due to float inaccuracies.
        self.stdevs = np.sqrt(np.maximum(eigenvalues, 0))
        self.variance = self.stdevs**2 # Variances along principal components
        self.covEigen = eigenvectors
        
        # If one wants stdevs to be sqrt of diagonal of self.cov (variances of original features):
        # diag_cov = np.diag(self.cov)
        # self.stdevs = np.sqrt(np.maximum(diag_cov, 0))
        # self.variance = self.stdevs**2
        # And covEigen handling would need to align with whether cov is treated as full or diagonal.
        # Current implementation makes stdevs reflect principal component variances, consistent with __init__ for full cov.

        return self

    def logprob(self, values: np.ndarray) -> np.ndarray:
        """
        Calculates the log probability density of given observation(s) under this Gaussian distribution.

        Args:
            values (np.ndarray): A NumPy array of observations.
                                 Can be a single observation (1D array of shape (dataSize,))
                                 or multiple observations (2D array of shape (nSamples, dataSize)).

        Returns:
            np.ndarray: The log probability density for each observation.
                        If `values` is 1D, returns a scalar.
                        If `values` is 2D, returns a 1D array of shape (nSamples,).
        """
        # scipy.stats.multivariate_normal.logpdf handles single and multiple observations.
        return scipy.stats.multivariate_normal.logpdf(values, mean=self.means, cov=self.cov, allow_singular=True)

    def prob(self, values: np.ndarray) -> np.ndarray:
        """
        Calculates the probability density of given observation(s) under this Gaussian distribution.

        Args:
            values (np.ndarray): A NumPy array of observations.
                                 Can be a single observation (1D array of shape (dataSize,))
                                 or multiple observations (2D array of shape (nSamples, dataSize)).

        Returns:
            np.ndarray: The probability density for each observation.
                        If `values` is 1D, returns a scalar.
                        If `values` is 2D, returns a 1D array of shape (nSamples,).
        """
        # scipy.stats.multivariate_normal.pdf handles single and multiple observations.
        return scipy.stats.multivariate_normal.pdf(values, mean=self.means, cov=self.cov, allow_singular=True)
    
    def plotCross(self) -> None:
        """Placeholder for plotting a cross representing the distribution. Not yet implemented."""
        # This would typically plot ellipses for 2D Gaussians.
        pass

    def adaptStart(self) -> None:
        """Placeholder for initializing adaptation parameters. Not yet implemented."""
        pass
    
    def adaptSet(self, data: np.ndarray) -> None: # Assuming data might be needed
        """Placeholder for setting adaptation data or parameters. Not yet implemented."""
        pass

    def adaptAccum(self, data_point: np.ndarray, weight: float = 1.0) -> None: # Example params
        """Placeholder for accumulating statistics for adaptation. Not yet implemented."""
        # This would typically involve weighted updates to mean and covariance.
        pass