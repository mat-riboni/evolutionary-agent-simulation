import numpy as np

class Brain:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        
        self.w2 = np.random.randn(hidden_size, hidden_size)
        self.b2 = np.random.randn(hidden_size)
        
        self.w3 = np.random.randn(hidden_size, output_size)
        self.b3 = np.random.randn(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1) 
        
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.tanh(z2)
        
        z3 = np.dot(a2, self.w3) + self.b3
        a3 = np.tanh(z3)
        
        return a3