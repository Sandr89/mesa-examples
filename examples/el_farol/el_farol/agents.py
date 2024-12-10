import mesa
import numpy as np

class BarCustomer(mesa.Agent):
    def __init__(self, model, unique_id, memory_size, crowd_threshold, num_strategies, bias_factor=0):
        super().__init__(unique_id, model)
        # Random values from -1.0 to 1.0
        self.strategies = np.random.rand(num_strategies, memory_size + 1) * 2 - 1
        self.best_strategy = self.strategies[0]
        self.attend = False
        self.memory_size = memory_size
        self.crowd_threshold = crowd_threshold
        self.utility = 0
        self.bias_factor = bias_factor  # New bias factor
        self.update_strategies()

    def update_attendance(self):
        prediction = self.predict_attendance(
            self.best_strategy, self.model.history[-self.memory_size :]
        )
        prediction += self.bias_factor  # Adjust prediction using bias factor
        if prediction <= self.crowd_threshold:
            self.attend = True
            self.model.attendance += 1
        else:
            self.attend = False

    def update_strategies(self):
        # Pick the best strategy based on new history window
        best_score = float("inf")
        for strategy in self.strategies:
            score = 0
            for week in range(self.memory_size):
                last = week + self.memory_size
                prediction = self.predict_attendance(
                    strategy, self.model.history[week:last]
                )
                score += abs(self.model.history[last] - prediction)
            if score <= best_score:
                best_score = score
                self.best_strategy = strategy
        should_attend = self.model.history[-1] <= self.crowd_threshold
        if should_attend != self.attend:
            self.utility -= 1
        else:
            self.utility += 1

    def predict_attendance(self, strategy, subhistory):
        # Predict attendance based on strategy and history
        return strategy[0] * 100 + np.dot(strategy[1:], subhistory)
