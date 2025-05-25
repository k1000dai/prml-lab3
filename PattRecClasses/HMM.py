import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain

class HMM:
    """
    Represents a Hidden Markov Model (HMM).

    An HMM is a statistical model that describes a system with hidden states
    that generate observable outputs. It consists of a state transition model
    (a Markov chain) and output probability distributions for each state.

    Attributes:
        stateGen (MarkovChain): The underlying Markov chain for state transitions.
        outputDistr (list[GaussD]): A list of output probability distributions (e.g., GaussD objects),
                                     one for each state.
        nStates (int): The number of hidden states in the model.
        dataSize (int): The dimensionality of the observation vectors.

    References:
        Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
        Rabiner, L. R. (1989) A tutorial on hidden Markov models
            and selected applications in speech recognition. Proc IEEE 77, 257-286.
    """
    def __init__(self, mc: MarkovChain, distributions: list[GaussD]):
        """
        Initializes a Hidden Markov Model.

        Args:
            mc (MarkovChain): An instance of MarkovChain representing the state
                              transition dynamics.
            distributions (list[GaussD]): A list of distribution objects (e.g., GaussD)
                                         representing the output probabilities for each state.
                                         All distributions must have the same dataSize.
        """
        self.stateGen: MarkovChain = mc
        self.outputDistr: list[GaussD] = distributions # Output distributions for each state

        self.nStates: int = mc.nStates # Number of states
        # Data dimensionality, assumes all distributions have the same dataSize
        self.dataSize: int = distributions[0].dataSize if distributions else 0


    def _calculate_observation_probabilities(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the observation probability matrix pX.

        pX[t, i] = P(X(t)=x(t) | S(t)=i), the probability of observing x[t] given state i.

        Args:
            x (np.ndarray): A matrix or vector of observations.
                            If a matrix, rows are time steps (observations), columns are features.
                            If a vector, it's treated as a sequence of 1D observations.

        Returns:
            np.ndarray: A 2D array where pX[t, i] is the probability of observation x[t]
                        given the model is in state i. Shape (T, nStates), where T is
                        the number of observations.
        """
        # Ensure x is a 2D array (T, D) where T is number of observations, D is dataSize
        if x.ndim == 1:
            # If x is 1D, reshape to (T, 1) if dataSize is 1, or (1, D) if it's a single D-dim observation.
            # Assuming x is (T,) for T observations of 1 feature.
            x = x.reshape(-1, self.dataSize if self.dataSize > 0 else 1)
        
        T = x.shape[0] # Number of observations
        pX = np.zeros((T, self.nStates)) # pX[t, i] = P(x[t] | S[t]=i)

        # Iterate over each observation vector x[t]
        for t in range(T):
            # Iterate over each state to calculate P(x[t] | S[t]=i)
            for i_state in range(self.nStates):
                pX[t, i_state] = self.outputDistr[i_state].prob(x[t])
        return pX

    def observationProb(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of observing each sample in x for each state.
        Alias for _calculate_observation_probabilities.

        Args:
            x (np.ndarray): A matrix or vector of output data samples.
                            Rows correspond to time steps/observations, columns to features.

        Returns:
            np.ndarray: Matrix `pX` where `pX[t, i] = P(X(t)=x[t] | S(t)=i)`.
                        Shape is (number of samples, nStates).
        """
        return self._calculate_observation_probabilities(x)
    
    def rand(self, nSamples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a random sequence of data from the Hidden Markov Model.

        Args:
            nSamples (int): The maximum number of output samples (observations) to generate.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - X (np.ndarray): A matrix with output data samples (observations).
                                  Shape (dataSize, nS), where nS is the actual number of samples.
                - S (np.ndarray): A row vector with the corresponding integer state values
                                  for each observation. Shape (nS,).
                                  nS <= nSamples, especially if the state generator is a
                                  finite-duration MarkovChain.
        """
        # Generate a sequence of states from the underlying Markov chain
        S = self.stateGen.rand(nSamples)
        nS_actual = len(S) # Actual number of states/samples generated
        
        # Initialize the observation matrix X
        X = np.zeros((self.dataSize, nS_actual))
        
        # For each generated state, draw an observation from its output distribution
        for i in range(nS_actual):
            # outputDistr[S[i]] is the distribution for the state S[i]
            # .rand(1) generates one sample from this distribution
            X[:, i] = self.outputDistr[S[i]].rand(1).flatten()
        
        return X, S
    
    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """
        Finds the most probable sequence of hidden states given observations using the Viterbi algorithm.

        Args:
            x (np.ndarray): A matrix or vector of observations.
                            Rows are time steps, columns are features.

        Returns:
            np.ndarray: The most probable sequence of states (state path). Shape (T,),
                        where T is the number of observations.
        """
        # Calculate observation probabilities P(x[t] | S[t]=i)
        pX = self.observationProb(x)
        # Delegate to the MarkovChain's Viterbi algorithm
        return self.stateGen.viterbi(pX)

    def train(self, x: np.ndarray, nIter: int = 10, threshold: float = 0.1, debug: bool = False) -> None:
        """
        Trains the HMM parameters using the Baum-Welch algorithm (an Expectation-Maximization algorithm).

        The Baum-Welch algorithm iteratively re-estimates the HMM parameters (initial state
        probabilities, transition probabilities, and output distribution parameters) to
        maximize the likelihood of the observed data `x`.

        Args:
            x (np.ndarray): A matrix of observation sequences. Rows are time steps (observations),
                            columns are features.
            nIter (int, optional): The maximum number of iterations for the Baum-Welch algorithm.
                                   Defaults to 10.
            threshold (float, optional): The convergence threshold for the log-likelihood.
                                         Training stops if the change in log-likelihood between
                                         iterations is less than this threshold. Defaults to 0.1.
            debug (bool, optional): If True, prints debug information (parameter values, log-likelihood)
                                    during training. Defaults to False.
        
        Side Effects:
            Modifies the HMM's parameters in-place (stateGen.q, stateGen.A, outputDistr means/cov).
        """
        prev_loglike = -np.inf # Initialize previous log-likelihood for convergence check

        for iteration in range(nIter):
            # Calculate observation probabilities P(x[t] | S[t]=i) based on current model parameters
            pX = self.observationProb(x)
            
            # E-step: Calculate expected values (gamma, alpha_hat, beta_hat, c)
            gamma, a_hat, b_hat, c = self._e_step(x, pX)

            # M-step: Re-estimate model parameters
            # Update initial state probabilities (pi or q)
            q_new = self._update_initial_probabilities(gamma)
            self.stateGen.q = q_new
            if debug:
                print(f"Iteration {iteration + 1}: Updated q (initial probabilities):\n{self.stateGen.q}")

            # Update transition probabilities (A)
            # Note: 'c' (scaling factors) might be needed for a more standard xi calculation,
            # but current implementation follows original structure which doesn't use it here.
            A_new = self._update_transition_probabilities(x, pX, a_hat, b_hat, c)
            self.stateGen.A = A_new
            if debug:
                print(f"Iteration {iteration + 1}: Updated A (transition probabilities):\n{self.stateGen.A}")
            
            # Update output distributions (e.g., means and covariances for Gaussian distributions)
            self._update_output_distributions(x, gamma, debug)
            
            # Calculate log-likelihood of the observations given the new model parameters
            # log(P(X|model)) = sum(log(c_t))
            current_loglike = np.sum(np.log(c))
            
            if debug or (iteration + 1) % 10 == 0 : # Print every 10 iterations or if debug
                print(f"Iteration {iteration + 1}: Log-likelihood = {current_loglike}")

            # Check for convergence
            if np.abs(current_loglike - prev_loglike) < threshold:
                print(f"Convergence reached at iteration {iteration + 1}.")
                break
            prev_loglike = current_loglike
        else: # Executed if loop completes without break
            print(f"Training finished after {nIter} iterations (max iterations reached).")


    def _e_step(self, x: np.ndarray, pX: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the E-step (Expectation step) of the Baum-Welch algorithm.

        This step calculates the forward probabilities (alpha_hat), backward probabilities (beta_hat),
        state occupation probabilities (gamma), and scaling factors (c).

        Args:
            x (np.ndarray): The sequence of observations. Shape (T, D).
            pX (np.ndarray): The observation probability matrix P(x[t] | S[t]=i). Shape (T, nStates).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - gamma (np.ndarray): State occupation probabilities P(S_t=i | X, model),
                                      using the specific scaling from the original implementation.
                                      Shape (T, nStates).
                - a_hat (np.ndarray): Scaled forward probabilities (alpha_hat). Shape (T, nStates).
                - b_hat (np.ndarray): Scaled backward probabilities (beta_hat). Shape (T, nStates).
                - c (np.ndarray): Scaling factors used in the forward pass. Shape (T,).
        """
        # alpha_hat are scaled forward variables: alpha_hat[t,i] = P(S[t]=i, x[1..t] | model) / P(x[1..t] | model)
        # c are scaling factors: c[t] = P(x[t] | x[1..t-1], model)
        a_hat, c = self.stateGen.forward(pX)
        
        # beta_hat are scaled backward variables: beta_hat[t,i] = P(x[t+1..T] | S[t]=i, model) / (product of future c's)
        b_hat = self.stateGen.backward(c, pX)
        
        # Calculate gamma_t(i) = P(S_t=i | X, model_lambda)
        # The original implementation uses gamma[t,j] = a_hat[t,j] * b_hat[t,j] * c[t].
        # This specific form of gamma is used in subsequent M-step calculations.
        # Note: Standard Rabiner definition of gamma_t(i) is alpha_t(i)*beta_t(i) / P(O|lambda),
        # or using scaled coefficients: (alpha_hat_t(i) * beta_hat_t(i)) / sum_j(alpha_hat_t(j) * beta_hat_t(j)).
        # The sum_j(...) term should normalize gamma_t(i) to sum to 1 for each t.
        # The product a_hat[t,j] * b_hat[t,j] itself is P(S_t=j | X, model) if scaling is consistent.
        # The multiplication by c[t] here is specific to this implementation's update rules.
        T = x.shape[0]
        gamma = np.zeros((T, self.nStates))
        for t in range(T): # For each time step
            for j_state in range(self.nStates): # For each state
                gamma[t, j_state] = a_hat[t, j_state] * b_hat[t, j_state] * c[t]
        
        return gamma, a_hat, b_hat, c

    def _update_initial_probabilities(self, gamma: np.ndarray) -> np.ndarray:
        """
        Updates the initial state probabilities (q or pi) during the M-step.

        Args:
            gamma (np.ndarray): State occupation probabilities P(S_t=i | X, model)
                                (using the specific scaling from _e_step). Shape (T, nStates).

        Returns:
            np.ndarray: The new initial state probability vector. Shape (nStates,).
        """
        # The original code re-estimates q (initial probabilities) as the overall proportion
        # of time spent in each state: sum_t(gamma_t(j)) / sum_t sum_k(gamma_t(k)).
        # This is somewhat unusual, as pi_i (initial state prob) is typically gamma_1(i).
        # However, this method maintains the original algorithm's logic.
        
        # sum_gamma_per_state[j] = sum_t gamma[t,j]
        sum_gamma_per_state = np.sum(gamma, axis=0) # Sum over time for each state
        total_sum_gamma = np.sum(sum_gamma_per_state) # Total sum of gamma over all states and time
        
        if total_sum_gamma == 0: # Avoid division by zero if gamma is all zeros
            return self.stateGen.q # Return current q or a uniform distribution
            
        q_new = sum_gamma_per_state / total_sum_gamma
        return q_new

    def _update_transition_probabilities(self, x: np.ndarray, pX: np.ndarray, 
                                         a_hat: np.ndarray, b_hat: np.ndarray, 
                                         c_scaling_factors: np.ndarray) -> np.ndarray:
        """
        Updates the state transition probabilities (A) during the M-step.

        This involves calculating xi_t(i,j) = P(S_t=i, S_{t+1}=j | X, model).

        Args:
            x (np.ndarray): The sequence of observations. Shape (T, D).
            pX (np.ndarray): Observation probability matrix P(x[t] | S[t]=i). Shape (T, nStates).
            a_hat (np.ndarray): Scaled forward probabilities (alpha_hat). Shape (T, nStates).
            b_hat (np.ndarray): Scaled backward probabilities (beta_hat). Shape (T, nStates).
            c_scaling_factors (np.ndarray): Scaling factors from the forward pass. Shape (T,).
                                            Currently unused in this specific xi calculation,
                                            matching original implementation.

        Returns:
            np.ndarray: The new state transition probability matrix A. Shape (nStates, nStates).
        """
        T = x.shape[0]
        nStates = self.nStates
        
        # xi[i,j,t] = P(S_t=i, S_{t+1}=j | X, model_lambda)
        # The formula used here: a_hat[t,i] * A[i,j] * pX[t+1,j] * b_hat[t+1,j]
        # This is proportional to xi_t(i,j) * P(O|lambda) / (c[t] * c[t+1]) based on Rabiner's scaled coefficients.
        # Or, more directly, it's (alpha_t(i) * A[i,j] * b_j(O_{t+1}) * beta_{t+1}(j)) / (c[t] * c[t+1])
        # where b_j(O_{t+1}) is pX[t+1,j].
        # The normalization for A_ij = sum_t xi_t(i,j) / sum_t gamma_t(i) usually assumes xi_t(i,j)
        # is the true conditional probability. The division by sum_j(sum_t(xi[i,j,t])) in the
        # original code handles the normalization implicitly based on this definition of xi.
        xi = np.zeros((nStates, nStates, T - 1)) # xi[i_state, j_state, t_time]
        
        for i_state in range(nStates):
            for j_state in range(nStates):
                for t_time in range(T - 1):
                    xi[i_state, j_state, t_time] = a_hat[t_time, i_state] * \
                                                   self.stateGen.A[i_state, j_state] * \
                                                   b_hat[t_time + 1, j_state] * \
                                                   pX[t_time + 1, j_state]
        
        # Sum xi over time: xi_ij[i,j] = sum_t xi[i,j,t]
        xi_ij_sum_over_t = np.sum(xi, axis=2) # Shape (nStates, nStates)
        
        # Update A_new[i,j] = xi_ij_sum_over_t[i,j] / sum_k(xi_ij_sum_over_t[i,k])
        # sum_k(xi_ij_sum_over_t[i,k]) is the expected number of transitions from state i.
        sum_xi_ij_for_each_i = np.sum(xi_ij_sum_over_t, axis=1, keepdims=True) # Shape (nStates, 1)
        
        A_new = np.zeros((nStates, nStates))
        # Avoid division by zero if a state is never visited or has no transitions from it.
        # np.divide handles this with the 'where' and 'out' arguments.
        A_new = np.divide(xi_ij_sum_over_t, sum_xi_ij_for_each_i, 
                          out=np.zeros_like(A_new), where=sum_xi_ij_for_each_i != 0)
        
        # Fallback for rows that sum to zero (e.g. if a state was never left)
        # This could mean keeping the old row or setting a uniform distribution.
        # For now, it results in a row of zeros if sum_xi_ij_for_each_i is 0 for that row.
        # Consider if self.stateGen.A should be used as a fallback for such rows.
        # E.g., if sum_xi_ij_for_each_i[i] == 0, A_new[i,:] = self.stateGen.A[i,:] (or uniform)

        return A_new

    def _update_output_distributions(self, x: np.ndarray, gamma: np.ndarray, debug: bool):
        """
        Updates parameters of the output distributions for each state during the M-step.

        This example assumes Gaussian output distributions (GaussD) and updates their
        means and covariances. For other distribution types, this method would need
        to be adjusted accordingly.

        Args:
            x (np.ndarray): The sequence of observations. Shape (T, D).
            gamma (np.ndarray): State occupation probabilities P(S_t=i | X, model)
                                (using the specific scaling from _e_step). Shape (T, nStates).
            debug (bool): If True, prints debug information about updated means and covariances.
        
        Side Effects:
            Modifies `self.outputDistr[i_state].means` and `self.outputDistr[i_state].cov` in-place.
        """
        T = x.shape[0] # Number of observations
        nStates = self.nStates
        dataSize = self.dataSize
        
        # Sum of gamma over time for each state: sum_gamma_over_time_per_state[i] = sum_t gamma[t,i]
        # This represents the expected number of times state i is visited (scaled by sum of all gammas).
        sum_gamma_over_time_per_state = np.sum(gamma, axis=0) # Shape (nStates,)
        
        epsilon = 1e-10 # Small value to prevent division by zero
        
        # Store new means temporarily as they are used in covariance update
        mu_new_arr = np.zeros((nStates, dataSize))

        # Update means for each state's output distribution
        for i_state in range(nStates):
            if sum_gamma_over_time_per_state[i_state] > epsilon:
                # Numerator for mu_i: sum_t (gamma[t,i] * x[t])
                # weighted_sum_x is sum of observations weighted by gamma for current state i_state
                weighted_sum_x = np.dot(gamma[:, i_state], x) # Shape (dataSize,)
                mu_new_arr[i_state,:] = weighted_sum_x / sum_gamma_over_time_per_state[i_state]
            else:
                # If state is not visited (sum_gamma is near zero), keep existing mean
                mu_new_arr[i_state,:] = self.outputDistr[i_state].means
            
            self.outputDistr[i_state].means = mu_new_arr[i_state,:] # Update distribution mean
            if debug:
                print(f"Updated mu (state {i_state+1}):\n{self.outputDistr[i_state].means}")

        # Update covariances for each state's output distribution
        for i_state in range(nStates):
            if sum_gamma_over_time_per_state[i_state] > epsilon:
                # Use the newly computed mean for this state (mu_new_arr[i_state,:])
                x_minus_mu = x - mu_new_arr[i_state,:] # Shape (T, dataSize)
                
                # Numerator for cov_i: sum_t (gamma[t,i] * (x[t]-mu_i)(x[t]-mu_i)^T)
                # weighted_x_minus_mu[t,:] = (x[t]-mu_i) * gamma[t,i]
                weighted_x_minus_mu = x_minus_mu * gamma[:, i_state][:, np.newaxis] # Shape (T, dataSize)
                # cov_new_i = sum_t (gamma[t,i] * outer(x[t]-mu_i, x[t]-mu_i))
                # This is equivalent to (weighted_x_minus_mu).T @ (x_minus_mu)
                cov_new_i = np.dot(weighted_x_minus_mu.T, x_minus_mu) # Shape (dataSize, dataSize)
                
                self.outputDistr[i_state].cov = cov_new_i / sum_gamma_over_time_per_state[i_state]
            # Else: If state is not visited, keep existing covariance (no update)
            
            if debug:
                print(f"Updated cov (state {i_state+1}):\n{self.outputDistr[i_state].cov}")
                
    def stateEntropyRate(self) -> float:
        """Placeholder for state entropy rate calculation. Not yet implemented."""
        # This would typically involve the stationary distribution and transition probabilities.
        pass
        return 0.0 # Placeholder return

    def setStationary(self) -> None:
        """Placeholder for setting the HMM to a stationary version. Not yet implemented."""
        # This might involve adjusting initial state probabilities to be the stationary
        # distribution of the Markov chain.
        pass

    def log_like(self, x: np.ndarray) -> float:
        """
        Calculates the log-likelihood of the observation sequence x given the HMM.

        This is done using the forward algorithm and its scaling factors.
        log P(X | model) = sum_t log(c_t), where c_t are scaling factors.

        Args:
            x (np.ndarray): A matrix or vector of observations.
                            Rows are time steps, columns are features.

        Returns:
            float: The log-likelihood of the observation sequence.
        """
        # pX[t,i] = P(x[t] | S[t]=i)
        pX = self._calculate_observation_probabilities(x)
        
        # a_hat are scaled forward probabilities, c are scaling factors
        # from the forward algorithm (self.stateGen.forward)
        _, c = self.stateGen.forward(pX)
        
        # Log-likelihood is sum of log of scaling factors
        # np.log will produce -inf if any c_i is 0, which is correct.
        # Add a small constant to c if strict positivity is needed before log,
        # but sum(log(c)) is standard.
        log_likelihood = np.sum(np.log(c)) # Ensure c contains positive values
        return log_likelihood

    def adaptStart(self) -> None:
        """Placeholder for starting adaptation. Not yet implemented."""
        pass

    def adaptSet(self) -> None:
        """Placeholder for setting adaptation parameters. Not yet implemented."""
        pass

    def adaptAccum(self) -> None:
        """Placeholder for accumulating adaptation statistics. Not yet implemented."""
        pass