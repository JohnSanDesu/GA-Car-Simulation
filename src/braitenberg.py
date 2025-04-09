import numpy as np

class Braitenberg:
    """
    This class holds the genotype and position of the Braitenberg vehicle.
    """
    def __init__(self, initial_pos, initial_bearing, geno):
        # Initialize the agent with starting position, bearing, and genotype.
        self.geno = geno
        self.initial_bearing = initial_bearing
        self.pos = initial_pos

    def get_geno(self):
        """
        Returns the genotype of the vehicle.
        """
        return self.geno
