import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    Represents a first-order discrete Markov chain.

    A Markov chain models a sequence of states S(t), t=1...T, where the probability
    of transitioning to the next state depends only on the current state.
    It is defined by initial state probabilities and state transition probabilities.
    Finite duration chains include a special END state.

    Attributes:
        q (np.ndarray): Initial state probabilities P(S(1)=j). Shape (nStates,).
        A (np.ndarray): State transition probabilities P(S(t)=j | S(t-1)=i).
                        Shape (nStates, nStates) for infinite duration chains,
                        or (nStates, nStates+1) for finite duration chains
                        (where the last column represents transitions to the END state).
        nStates (int): The number of states (excluding the END state).
        is_finite (bool): True if the Markov chain has a finite duration (includes an END state).
    """
    def __init__(self, initial_prob: np.ndarray, transition_prob: np.ndarray):
        """
        Initializes a MarkovChain object.

        Args:
            initial_prob (np.ndarray): A 1D array representing the initial state
                                       probabilities P(S(1)=j). Sums to 1.
                                       Shape (nStates,).
            transition_prob (np.ndarray): A 2D array representing the state transition
                                          probabilities P(S(t)=j | S(t-1)=i).
                                          Rows sum to 1 (or less if finite duration with END state).
                                          Shape (nStates, nStates) for infinite duration, or
                                          (nStates, nStates+1) for finite duration.
        """
        self.q: np.ndarray = initial_prob  # Initial state probabilities P(S(1)=i)
        self.A: np.ndarray = transition_prob # Transition probabilities A[i,j] = P(S(t)=j | S(t-1)=i)

        self.nStates: int = transition_prob.shape[0] # Number of actual states (excluding END state)

        self.is_finite: bool = False
        # If the transition matrix has more columns than rows, it implies an END state column
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True
        
    def probDuration(self, tmax: int) -> np.ndarray:
        """
        Calculates the probability mass function (PMF) of sequence durations t=1...tmax.

        This is meaningful only for finite-duration Markov Chains, as pD(t) will be
        zero for all t for infinite-duration chains (they don't end).
        The duration is the time step at which the END state is entered.

        Args:
            tmax (int): The maximum duration to calculate probabilities for.

        Returns:
            np.ndarray: A 1D array `pD` of shape (tmax,), where `pD[t-1]` is the
                        probability that the chain duration is `t`.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            # pSt[j] = P(S(t)=j, S(t+1)=END_STATE), probability of ending at time t from state j
            # This is effectively (I - A^T)q initially for t=1.
            # It represents the probability flux into the end state at each time step.
            # q_eff = (I-A^T)q where A is only the state-to-state part.
            # Here, A is (nStates, nStates), so A.T is also (nStates, nStates).
            # Let's use the formulation from the reference.
            # The probability of being in state j at time t and transitioning to END state at t+1.
            # This is P(S(t)=j) * P(S(t+1)=END | S(t)=j)
            # Let alpha_t be P(S(t)=j). alpha_1 = q. alpha_t = alpha_{t-1} @ A_states (A without end column)
            # pD[t-1] = P(Duration = t) = sum_j P(S(t)=j) * A[j, END_STATE_COL]
            
            # Original code: pSt = (np.eye(self.nStates)-self.A.T)@self.q seems related to fundamental matrix.
            # Let's trace the original logic:
            # pSt_0 = (I-A_states^T)q
            # pD[0] (duration 1) = sum(pSt_0) IF pSt_0 means prob of ending at t=1.
            # The probability of ending at time t=1 is q @ A[:, self.nStates]
            # The probability of being in state j at time t is alpha[t-1, j]
            
            # Using the formula from Leijon, Problem 4.8:
            # pi_t = q^T * (A_states)^{t-1} (row vector of state probabilities at time t)
            # P(Duration = t) = pi_t * A_end (where A_end is the column for transitions to END state)
            
            # Let's try to match the provided implementation's spirit:
            # pSt seems to represent the probability of being in a state *and* terminating at the current step.
            # If A is (N x N) (states only), A_end is a separate vector (N x 1).
            # Here, A can be (N x N+1). A_states = self.A[:, :self.nStates]
            # A_end_col = self.A[:, self.nStates]
            
            # Let alpha[t] be the probability distribution over states at time t (row vector)
            alpha = self.q.copy() # P(S(1)=j)
            for t in range(tmax): # t from 0 to tmax-1, so duration is t+1
                # Probability of ending at duration (t+1)
                # P(Duration = t+1) = sum_j P(S(t+1)=j) * A[j, END_STATE_COL] (if S is 0-indexed time)
                # Or P(S(t)=j) * A[j, END_STATE_COL] if S is 1-indexed time.
                # The problem asks for t=1...tMax. So pD[k-1] is for duration k.
                if alpha is None: # Should not happen if loop condition is correct
                    break
                prob_ending_now = np.sum(alpha @ self.A[:, self.nStates]) # alpha is (N,), A is (N,N+1) -> (N+1,)
                pD[t] = prob_ending_now
                
                # Probability of transitioning to next non-terminal states
                alpha = alpha @ self.A[:, :self.nStates] # Update alpha for next step
                if np.sum(alpha) < 1e-15: # If all probability mass has exited
                    break
        return pD

    def probStateDuration(self, tmax: int) -> np.ndarray:
        """
        Calculates the probability mass P(D_j=t) for spending t time steps in each state j.
        This assumes a geometric distribution of durations for each state, based on
        the diagonal elements of the transition matrix A (self-loop probabilities).

        Args:
            tmax (int): The maximum duration `t` to calculate probabilities for.

        Returns:
            np.ndarray: A 2D array `pD` of shape (nStates, tmax), where `pD[j, t-1]`
                        is the probability P(D_j=t) that state `j` has a duration of `t`.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        # t_values represents durations from 1 to tmax
        t_values = np.arange(1, tmax + 1).reshape(1, -1) # Shape (1, tmax)
        # a_ii contains probabilities P(S(t+1)=i | S(t)=i), i.e., self-loop probabilities
        a_ii = np.diag(self.A[:, :self.nStates]).reshape(-1, 1) # Shape (nStates, 1)
        
        # P(Duration in state i = t) = (a_ii)^(t-1) * (1 - a_ii) for t >= 1
        # The original code uses t in log(a_ii)*t. If t is 0-indexed for duration 1, then a_ii^t * (1-a_ii).
        # If t_values is 1-indexed (1 to tmax), then (a_ii)^(t_values-1) * (1-a_ii)
        # Original: log(aii)*t + log(1-aii). If t is 0..tmax-1 for durations 1..tmax
        # then this is (a_ii)^t * (1-a_ii).
        # This corresponds to P(D=d) = p^(d-1)*(1-p) where d is duration (1,2,...) and p is self-loop prob.
        # So, if t_values is 0-indexed for d=1, then p^t_values * (1-p).
        # The arange was (tmax), so 0 to tmax-1.
        t_indices = np.arange(tmax).reshape(1,-1) # Durations represented by indices 0 to tmax-1
                                                 # index k corresponds to duration k+1
        
        # log P(D_i = k+1) = k * log(a_ii) + log(1 - a_ii)
        log_pD = t_indices * np.log(a_ii) + np.log(1 - a_ii)
        pD = np.exp(log_pD)

        return pD # pD[j, k] is P(Duration in state j = k+1)

    def meanStateDuration(self) -> np.ndarray:
        """
        Calculates the expected number of time samples spent in each state (mean duration).
        This is based on the self-loop probabilities: 1 / (1 - a_ii).

        Returns:
            np.ndarray: A 1D array where element `i` is the mean duration of state `i`.
                        Shape (nStates,).
        """
        # a_ii = P(S(t+1)=i | S(t)=i)
        a_ii = np.diag(self.A[:, :self.nStates])
        # Mean duration for state i is 1 / (probability of leaving state i)
        # Probability of leaving state i is (1 - a_ii)
        return 1.0 / (1.0 - a_ii)
    
    def rand(self, tmax: int) -> np.ndarray:
        """
        Generates a random state sequence from the Markov chain.

        Args:
            tmax (int): The maximum length of the desired state sequence.
                        For infinite-duration chains, the sequence length will be `tmax`.
                        For finite-duration chains, the sequence may be shorter if the
                        END state is reached before `tmax` samples.

        Returns:
            np.ndarray: An integer row vector representing the random state sequence.
                        The END state is not included in the sequence.
                        Length is `tmax` for infinite-duration chains, or `<= tmax` for
                        finite-duration chains. Shape (actual_length,).
        """
        S = np.zeros(tmax, dtype=int) # Pre-allocate state sequence array
        
        # First state S(0) is chosen according to initial probabilities q
        S[0] = np.random.choice(self.nStates, p=self.q)
        
        # Generate subsequent states
        for t in range(1, tmax):
            current_state = S[t-1]
            if self.is_finite:
                # For finite chains, transition probabilities A are (nStates, nStates+1)
                # The (nStates)-th column (0-indexed) corresponds to the END state.
                # Possible next states are 0 to nStates (inclusive for END state).
                possible_next_states = np.arange(self.nStates + 1)
                S[t] = np.random.choice(possible_next_states, p=self.A[current_state, :])
                
                if S[t] == self.nStates: # If END state is reached
                    S = S[:t] # Truncate sequence, END state is not included
                    break
            else:    
                # For infinite chains, A is (nStates, nStates)
                possible_next_states = np.arange(self.nStates)
                S[t] = np.random.choice(possible_next_states, p=self.A[current_state, :])
        return S

    def viterbi(self, pX: np.ndarray) -> np.ndarray:
        """
        Finds the most likely sequence of hidden states given observation probabilities
        using the Viterbi algorithm.

        Args:
            pX (np.ndarray): Observation probability matrix. `pX[t, j]` is P(observation at t | state j).
                             Shape (T, nStates), where T is the number of observations.

        Returns:
            np.ndarray: The most probable sequence of states (path). Shape (T,).
        """
        T, N = pX.shape # T = number of observations, N = number of states
        
        # viterbi_prob[t, j]: max probability of any path ending at state j at time t
        viterbi_prob = np.zeros((T, N))
        # viterbi_backpointer[t, j]: state at t-1 that maximizes probability to reach state j at time t
        viterbi_backpointer = np.zeros((T, N), dtype=int)
        
        # Initialization step (t=0)
        # Prob of being in state i at t=0 = initial_prob[i] * observation_prob[0,i]
        viterbi_prob[0, :] = self.q * pX[0, :]
        # No backpointers for t=0 as there's no previous state. (Often set to 0 or special value)
        # The original code sets viterbi_backpointer[0,i] = 0, which is fine.

        # Recursion step (t=1 to T-1)
        for t in range(1, T):
            for j in range(N): # For each current state j at time t
                # Find max_i { viterbi_prob[t-1, i] * A[i,j] * pX[t,j] }
                probs_to_j = viterbi_prob[t-1, :] * self.A[:, j] * pX[t, j]
                viterbi_prob[t, j] = np.max(probs_to_j)
                viterbi_backpointer[t, j] = np.argmax(probs_to_j)
            
            # Normalization (optional but helps prevent underflow for long sequences)
            # If not normalized, viterbi_prob values can become extremely small.
            # Normalization does not affect the selection of the most likely path.
            # The original code normalizes, so we keep it.
            current_sum = np.sum(viterbi_prob[t, :])
            if current_sum > 0:
                 viterbi_prob[t, :] /= current_sum
            # else: if sum is zero, all probs are zero, path is undefined or has zero probability.

        # Termination: Find the most likely final state at T-1
        viterbi_path = np.zeros(T, dtype=int)
        viterbi_path[T-1] = np.argmax(viterbi_prob[T-1, :])
        
        # Path backtracking
        for t in range(T-2, -1, -1): # Iterate backwards from T-2 down to 0
            # The state at time t is the backpointer from the state chosen at t+1
            viterbi_path[t] = viterbi_backpointer[t+1, viterbi_path[t+1]]
            
        return viterbi_path
            
    
    def stationaryProb(self) -> np.ndarray:
        """Placeholder for calculating stationary probabilities. Not yet implemented."""
        # This typically involves finding the eigenvector of A.T corresponding to eigenvalue 1.
        pass
        return np.zeros(self.nStates) # Placeholder
    
    def stateEntropyRate(self) -> float:
        """Placeholder for calculating state entropy rate. Not yet implemented."""
        # H(S) = - sum_i pi_i * sum_j A_ij * log2(A_ij), where pi is stationary distribution.
        pass
        return 0.0 # Placeholder
    
    def setStationary(self) -> None:
        """Placeholder for setting initial probabilities to stationary distribution. Not yet implemented."""
        # self.q = self.stationaryProb()
        pass

    def logprob(self, S: np.ndarray) -> float:
        """
        Calculates the log probability of a given state sequence S.
        P(S) = P(S_1) * P(S_2|S_1) * ... * P(S_T|S_{T-1})

        Args:
            S (np.ndarray): A 1D array of integers representing the state sequence.

        Returns:
            float: The log probability of the sequence.
        """
        if not S.any():
            return -np.inf # Or handle as error
        
        log_p = np.log(self.q[S[0]])
        for t in range(1, len(S)):
            log_p += np.log(self.A[S[t-1], S[t]])
        return log_p

    def join(self, N_extra_states: int) -> 'MarkovChain':
        """Placeholder for joining Markov chains. Not yet implemented."""
        # This would likely involve creating a new MarkovChain by combining self with another.
        pass
        # Example return, actual implementation would be complex:
        return MarkovChain(self.q, self.A) 

    def initLeftRight(self, n_states: int, end_state: bool = False) -> None:
        """Placeholder for initializing a left-right Markov chain. Not yet implemented."""
        # A left-right model typically only allows transitions to the same or next state.
        pass
    
    def initErgodic(self, n_states: int, self_loops: bool = True) -> None:
        """Placeholder for initializing an ergodic Markov chain. Not yet implemented."""
        # An ergodic chain means every state is reachable from every other state.
        pass

    def forward(self, pX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the forward algorithm to calculate scaled forward probabilities (alpha_hat)
        and scaling factors (c).

        alpha_hat[t,i] = P(S(t)=i, X(1..t) | Model) / P(X(1..t) | Model)
                       = alpha_t(i) / sum_j(alpha_t(j))
        c[t] = P(X(t) | X(1..t-1), Model) = sum_j(alpha_t(j)) (sum of unscaled alpha at time t)

        Args:
            pX (np.ndarray): Observation probability matrix. `pX[t,j]` is P(observation at t | state j).
                             Shape (T, nStates).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - alpha_hat (np.ndarray): Scaled forward probabilities. Shape (T, nStates).
                - c (np.ndarray): Scaling factors. Shape (T,) or (T+1,) if finite_duration.
        """
        T, N = pX.shape # T = number of observations, N = number of states
        
        c = np.zeros(T) # Scaling factors c_t = P(x_t | x_1 ... x_{t-1})
        alpha_hat = np.zeros((T, N)) # Scaled forward probabilities alpha_hat(t,i)
        
        # Initialization (t=0)
        # alpha_0(i) = q_i * pX(0,i)
        alpha_0 = self.q * pX[0, :]
        c[0] = np.sum(alpha_0) # c_0 = sum_i alpha_0(i)
        alpha_hat[0, :] = alpha_0 / c[0] if c[0] != 0 else alpha_0 # Scale alpha_0
        
        # Recursion (t=1 to T-1)
        for t in range(1, T):
            # alpha_t(j) = [ sum_i alpha_hat(t-1,i) * A(i,j) ] * pX(t,j)
            # Sum over previous states i: sum_i (alpha_hat_{t-1}(i) * A[i,j])
            # This is effectively (alpha_hat[t-1, :] @ A[:, j]) but loop is clearer here.
            alpha_t_unscaled = np.zeros(N)
            for j_curr_state in range(N): # For each current state j at time t
                sum_prev_alpha_times_A = 0
                for i_prev_state in range(N): # Sum over all previous states i at time t-1
                    sum_prev_alpha_times_A += alpha_hat[t-1, i_prev_state] * self.A[i_prev_state, j_curr_state]
                alpha_t_unscaled[j_curr_state] = sum_prev_alpha_times_A * pX[t, j_curr_state]
            
            c[t] = np.sum(alpha_t_unscaled) # c_t = sum_j alpha_t(j) (unscaled)
            alpha_hat[t, :] = alpha_t_unscaled / c[t] if c[t] != 0 else alpha_t_unscaled # Scale alpha_t
        
        # For finite duration, an additional scaling factor c[T] might be calculated
        # representing probability of transitioning to END state from T-1.
        # Original code: if self.is_finite: c = np.append(c,np.sum(alpha_hat[-1,:]*self.A[:,N]))
        # This implies A has N+1 columns. self.A[:,N] would be the transitions to END state.
        if self.is_finite:
            # Prob of terminating after last observation: sum_i alpha_hat(T-1,i) * A(i, END)
            # This assumes c was (T,1) initially, so c becomes (T+1,1)
            # My c is (T,), so it will become (T+1,)
            prob_to_end_state = np.sum(alpha_hat[T-1, :] @ self.A[:, self.nStates])
            c = np.append(c, prob_to_end_state) # c[T] (0-indexed)
        
        return alpha_hat, c.reshape(-1,1) # Return c as column vector to match original

    def finiteDuration(self) -> bool:
        """Checks if the Markov chain has a finite duration. Not fully implemented."""
        # Placeholder - relies on is_finite attribute.
        # A more robust check might analyze the transition matrix A.
        pass
        return self.is_finite # Returns the attribute set in __init__
    
    def backward(self, c: np.ndarray, pX: np.ndarray) -> np.ndarray:
        """
        Performs the backward algorithm to calculate scaled backward probabilities (beta_hat).

        beta_hat[t,i] = P(X(t+1..T) | S(t)=i, Model) / product_k=t+1^T (c_k)
                      = (beta_t(i) / c_t) / product_k=t+1^T (c_k) (if beta_t(i) scaled by c_t)
        More commonly, beta_hat[t,i] is scaled by c[t] from the forward pass: beta_t(i) / c[t]

        Args:
            c (np.ndarray): Scaling factors obtained from the forward algorithm.
                            Shape (T,) or (T+1,) if finite_duration.
            pX (np.ndarray): Observation probability matrix. `pX[t,j]` is P(observation at t | state j).
                             Shape (T, nStates).

        Returns:
            np.ndarray: Scaled backward probabilities beta_hat. Shape (T, nStates).
        """
        T, N  = pX.shape # T = number of observations, N = number of states
        c = c.flatten() # Ensure c is 1D array for consistent indexing
        
        beta_hat = np.zeros((T,N)) # Scaled backward probabilities beta_hat(t,i)
        
        # Initialization (t=T-1)
        # For finite duration, beta_hat(T-1,i) = A(i,END) / (c(T-1)*c(T)) if c has T+1 elements
        # The c from forward pass might have T or T+1 elements.
        # If c has T elements (c[0]..c[T-1]), then c[T-1] is the last scaling factor for observations.
        # If c has T+1 elements (c[0]..c[T]), c[T] is P(END | X_1..T).
        if self.is_finite:
            # Original: beta_hat[T-1,:] = self.A[:,N]/(c[T-1]*c[T])
            # This implies c has T+1 elements (indexed 0 to T).
            # A[:,N] is the column of transitions to END state.
            # c[T-1] is the observation scaling factor at T-1.
            # c[T] is the final transition-to-END scaling factor.
            if len(c) == T + 1 : # c includes scaling for terminal transition
                 beta_hat[T-1,:] = self.A[:, self.nStates] / (c[T-1] * c[T]) if (c[T-1]*c[T]) != 0 else self.A[:, self.nStates]
            else: # c only has T elements, assume standard beta_hat[T-1,i] = 1, then scale by c[T-1]
                 beta_hat[T-1,:] = 1.0 / c[T-1] if c[T-1] != 0 else 1.0
        else: # Infinite duration
            # beta_hat(T-1,i) = 1, then scaled by c(T-1)
            beta_hat[T-1,:] = 1.0 / c[T-1] if c[T-1] != 0 else 1.0
            
        # Recursion (t=T-2 down to 0)
        for t in range(T-2,-1,-1):
            for i_curr_state in range(N): # For each current state i at time t
                sum_A_pX_beta = 0
                for j_next_state in range(N): # Sum over all next states j at time t+1
                    sum_A_pX_beta += self.A[i_curr_state, j_next_state] * \
                                     pX[t+1, j_next_state] * \
                                     beta_hat[t+1, j_next_state]
                beta_hat[t, i_curr_state] = sum_A_pX_beta / c[t] if c[t] != 0 else sum_A_pX_beta
                
        return beta_hat

    def adaptStart(self) -> None:
        """Placeholder for initializing adaptation parameters. Not yet implemented."""
        pass

    def adaptSet(self, x: np.ndarray) -> None: # Assuming x might be needed
        """Placeholder for setting adaptation data or parameters. Not yet implemented."""
        pass

    def adaptAccum(self, pX: np.ndarray, hmm) -> None: # Assuming HMM object might be needed
        """Placeholder for accumulating statistics for adaptation. Not yet implemented."""
        # This would typically be part of Baum-Welch for a single sequence in an HMM context.
        pass
