import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        S = np.zeros(tmax, dtype=int)
        S[0] = np.random.choice(self.nStates, p=self.q)
        for t in range(1, tmax):
            if self.is_finite:
                # nStates+1 is the END state
                S[t] = np.random.choice(self.nStates + 1, p=self.A[S[t-1]])
                if S[t] == self.nStates:
                    S = S[:t]
                    break
            else:    
                S[t] = np.random.choice(self.nStates, p=self.A[S[t-1]])
        return S

    def viterbi(self, pX):
        print(pX.shape)
        N, T = pX.shape
        viterbi_prob = np.zeros((N, T))
        viterbi_prob[:, 0] = self.q * pX[:, 0]
        for t in range(1, T):
            for j in range(N):
                viterbi_prob[j, t] = np.max(viterbi_prob[:, t-1] * self.A[:, j]) * pX[j, t]
            
        # Backtrack to find the most likely state sequence
        viterbi_path = np.zeros(T, dtype=int)
        viterbi_path[-1] = np.argmax(viterbi_prob[:, -1])
        for t in range(T-2, -1, -1):
            viterbi_path[t] = np.argmax(viterbi_prob[:, t] * self.A[:, viterbi_path[t+1]])
        
        viterbi_likelihood = np.max(viterbi_prob[:, -1])
        return viterbi_path, viterbi_likelihood
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        # pX is a matrix of size (N, D)
        # where T is the number of time samples, N is the number of states.
        T, N   = pX.shape
        c = np.zeros((T,1))
        alpha_hat = np.zeros((T, N))
        #initialization
        for i in range(N):
            alpha_hat[0,i] = self.q[i]*pX[0][i]
        c[0] = np.sum(alpha_hat[0,:])
        alpha_hat[0,:] = alpha_hat[0,:]/c[0]
        
        for t in range(1,T):
            alpha_temp = np.zeros((N))
            for j in range(N):
                sum = 0
                for i in range(N):
                    sum+=alpha_hat[t-1][i]*self.A[i][j]
                alpha_temp[j] = sum * pX[t][j]
            c[t] = np.sum(alpha_temp)
            alpha_hat[t,:] = alpha_temp/c[t]
        
        if self.is_finite: #add c_[T+1]
            c = np.append(c,np.sum(alpha_hat[-1,:]*self.A[:,N]))
        
        return alpha_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self,c, pX):
        T, N  = pX.shape
        beta_hat = np.zeros((T,N))
        if self.is_finite:
            beta_hat[T-1,:] = self.A[:,N]/(c[T-1]*c[T])
        else:
            beta_hat[T-1,:] = 1/c[T-1]
        for t in range(T-2,-1,-1):
            for i in range(N):
                sum = 0
                for j in range(N):
                    sum+=self.A[i][j]*pX[t+1][j]*beta_hat[t+1][j]
                beta_hat[t][i] = sum/c[t]  
        return beta_hat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
