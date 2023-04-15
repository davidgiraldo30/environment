import numpy as np

class tsne:
    def __init__(self, n_components=2, perplexity=15.0, max_iter=50, momentum = 1.0, learning_rate=10,random_state=1234):
        """
        T-SNE: A t-Distributed Stochastic Neighbor Embedding implementation. Built based on https://github.com/nlml/tsne_raw
        It's a tool to visualize high-dimensional data. It converts
        similarities between data points to joint probabilities and tries
        to minimize the Kullback-Leibler divergence between the joint
        probabilities of the low-dimensional embedding and the
        high-dimensional data. 
        Parameters:
        ----------
        max_iter : int, default 300
        perplexity : float, default 15.0
        n_components : int, default 2
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter    
        self.momentum = momentum
        self.lr = learning_rate
        self.seed=random_state

    def fit(self, X):
        self.Y = np.random.RandomState(self.seed).normal(0., 0.0001, [X.shape[0], self.n_components])
        self.Q, self.distances = self.q_tsne()
        self.P=self.p_joint(X)

    def transform(self, X):
        if self.momentum:
            Y_m2 = self.Y.copy()
            Y_m1 = self.Y.copy()

        for i in range(self.max_iter):

            # Get Q and distances (distances only used for t-SNE)
            self.Q, self.distances = self.q_tsne()
            # Estimate gradients with respect to Y
            grads = self.tsne_grad()

            # Update Y
            self.Y = self.Y - self.lr * grads

            if self.momentum:  # Add momentum
                self.Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = self.Y.copy()
        return self.Y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def p_joint(self, X):
        """Given a data matrix X, gives joint probabilities matrix.
        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        def p_conditional_to_joint(P):
            """Given conditional probabilities matrix P, return
            approximation of joint distribution probabilities."""
            return (P + P.T) / (2. * P.shape[0])
        def calc_prob_matrix(distances, sigmas=None, zero_index=None):
            """Convert a distances matrix to a matrix of probabilities."""
            if sigmas is not None:
                two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                return self.softmax(distances / two_sig_sq, zero_index=zero_index)
            else:
                return self.softmax(distances, zero_index=zero_index)
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas()
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        self.P = p_conditional_to_joint(p_conditional)
        return self.
