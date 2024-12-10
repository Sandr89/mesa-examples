import mesa
import numpy as np

from .agents import BarCustomer

class ElFarolBar(mesa.Model):
    def __init__(
        self,
        crowd_threshold=60,
        num_strategies=10,
        memory_size=10,
        N=100,
        bias_factor=0,
    ):
        super().__init__()
        self.running = True
        self.num_agents = N

        # Initialize the previous attendance randomly so the agents have a history
        # to work with from the start.
        self.history = np.random.randint(0, 100, size=memory_size * 2).tolist()
        self.attendance = self.history[-1]
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents with bias factor
        for i in range(self.num_agents):
            agent = BarCustomer(self, i, memory_size, crowd_threshold, num_strategies, bias_factor)
            self.schedule.add(agent)

        self.datacollector = mesa.DataCollector(
            model_reporters={"Customers": "attendance"},
            agent_reporters={"Utility": "utility", "Attendance": "attend"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.attendance = 0
        self.schedule.step()
        # Maintain constant history length
        self.history.pop(0)
        self.history.append(self.attendance)