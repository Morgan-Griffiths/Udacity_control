import numpy as np

def initialize_N(T):
    """
    Orstein Uhlenbeck process
    theta = 0.15
    sigma = 0.2
    mu = 0
    dX = theta(mu-X) dt + sigma * dW
    """
    theta = 0.15
    sigma = 0.2
    mu = 0
    tau = 0.5
    dt = 1
    n = int(T / dt)
    
    t = np.linspace(0.,T,n)
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n)
    
    for i in range(1,n):
        x[i] = x[i-1] + dt * (-(x[i-1] - mu)/tau) + \
            sigma_bis * sqrtdt * np.random.randn()
        
    return x
#     N = theta(-X) * dt + sigma * W
    
#     X += dt * (-(X - mu) / tau) + \
#         sigma * np.random.randn(ntrials)