import numpy as np
from matplotlib import pyplot
from scipy import optimize

class evaluateNN():
    #initialize evaluation neural network
    def __init__(self, X, Y, costFunction):
        self.X = X
        np.random.shuffle(X)
        self.X_train, self.X_test, self.X_val = np.split(X, [int(0.6*len(X)),int(0.8*len(X))])
        self.Y = Y
        np.random.shuffle(Y)
        self.Y_train, self.Y_test, self.Y_val = np.split(Y, [int(0.6*len(X)),int(0.8*len(X))])

        self.costFunction = costFunction

    # Detect Bias-variance: 
    # Learning curve plots training and cross validation error as a function of training set size (gap indicates high variance, while no gap indicates high bias)
    # !Important: High bias doesn't mean adding more training examples!
    # !Important: adding more training examples could help in case of high variance 
    def learningCurve(self):
        # Number of training examples
        m = self.Y.size

        # You need to return these values correctly
        error_train = np.zeros(m)
        error_val   = np.zeros(m)

        for i in range(1, m + 1):
            theta_t = self.trainLinearReg(self.costFunction, self.X_train[:i], self.Y_train[:i], lambda_ = 0)
            error_train[i - 1], _ = self.costFunction(self.X_train[:i], self.Y_train[:i], theta_t, lambda_ = 0)
            error_val[i - 1], _ = self.costFuntion(self.X_val, self.Y_val, theta_t, lambda_ = 0)
            
        return error_train, error_val
    def learningCurvePlot(self):
        error_train, error_val = self.learningCurve()

        m = self.Y.size

        pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
        pyplot.title('Learning curve for linear regression')
        pyplot.legend(['Train', 'Cross Validation'])
        pyplot.xlabel('Number of training examples')
        pyplot.ylabel('Error')
        pyplot.axis([0, 13, 0, 150])


    # Depending on bias-variance -> adding more or less features
    # F.e. High bias (underfitting) -> adding more features
    def evaluatePol(self):
        pol_vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        lambda_ = 100

        error_train = np.zeros(len(pol_vec))
        error_val = np.zeros(len(pol_vec))

        cur_error = 0
        pol_ideal = 1
        for i in range(len(pol_vec)):
            X_poly = self.polyFeatures(self.X, i)
            X_poly, mu, sigma = self.featureNormalize(X_poly)
            X_poly = np.concatenate([np.ones((self.Y_train.size, 1)), X_poly], axis=1)

            theta_t = self.trainLinearReg(self.costFunction, X_poly, self.Y_train, lambda_=lambda_, maxiter=55)
            error_val[i], _ = self.costFunction(self.X_val, self.Y_val, theta_t, lambda_ = 0)
            error_train[i], _ = self.costFunction(self.X_train, self.Y_train, theta_t, lambda_ = 0)

            if (error_val[i] >= cur_error):
                pol_ideal = pol_vec[i]
                cur_error = error_val[i]
    
        return pol_vec, error_val, _, pol_ideal
    def evaluatePolplot(self):
        pol_vec, error_val, error_train, pol_ideal = self.evaluatePol()

        pyplot.plot(pol_vec, error_train, '-o', pol_vec, error_val, '-o', lw=2)
        pyplot.legend(['Train', 'Cross Validation'])
        pyplot.xlabel('lambda')
        pyplot.ylabel('Error')
            

    # Depending on bias-variance -> higher or lower regularisation term
    # F.e. High bias -> lower regularisation term
    def validationCurve(self):
        # Selected values of lambda (you should not change this)
        lambda_vec = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10]

        # You need to return these variables correctly.
        error_train = np.zeros(len(lambda_vec))
        error_val = np.zeros(len(lambda_vec))


        cur_error = 0
        lambda_ideal = 1
        for i in range(len(lambda_vec)):
            lambda_try = lambda_vec[i]
            theta_t = self.trainLinearReg(self.costFunction, self.X_train, self.Y_train, lambda_ = lambda_try)
            error_train[i], _ = self.costFunction(self.X_train, self.Y_train, theta_t, lambda_ = 0)
            error_val[i], _ = self.costFunction(self.X_val, self.y_val, theta_t, lambda_ = 0)
            #choose lambda with min(error_val)
            if (error_val[i] >= cur_error):
                lambda_ideal = lambda_vec[i]
                cur_error = error_val[i]

        return lambda_vec, error_train, error_val, lambda_ideal

    def validationCurvePlot(self):
        lambda_vec, error_train, error_val, _ = self.validationCurve(self.X, self.Y, self.X_val, self.Y_val)

        pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
        pyplot.legend(['Train', 'Cross Validation'])
        pyplot.xlabel('lambda')
        pyplot.ylabel('Error')

    #calculate test-error (evaluate model on a test set that was not used in any part of training)
    def testError(self, lambda_ideal):
        theta_t = self.trainLinearReg(self.costFunction, self.X_train, self.Y_train, lambda_ = lambda_ideal)
        error_test = self.costFunction(self.X_test, self.Y_test, theta_t, lambda_ = 0)

        return error_test


    # other tips for neural network architectures and over-/underfitting
    # small neural network: more prone to underfitting (comp cheaper)
    # large neural network: more prone to overfitting (comp more expensive)

    #after this: manual error analysis

    
    
    
    #other help functions
    def featureNormalize(X):
        mu = np.mean(X, axis=0)
        X_norm = X - mu

        sigma = np.std(X_norm, axis=0, ddof=1)
        X_norm /= sigma
        return X_norm

    def polyFeatures(X, p):
        X_poly = np.zeros((X.shape[0], p))

        for i in range(p):
            X_poly[:, i] = X[:, 0] ** (i + 1)

        return X_poly

    #find theta parameter
    def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
        # Initialize Theta
        initial_theta = np.zeros(X.shape[1])

        # Create "short hand" for the cost function to be minimized
        costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

        # Now, costFunction is a function that takes in only one argument
        options = {'maxiter': maxiter}

        # Minimize using scipy
        res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
        return res.x


    