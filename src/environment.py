import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(data, mean=0.0, stddev=0.1):
    """
    Adds Gaussian noise to the given data.
    """
    noise = np.random.normal(mean, stddev, data.shape)
    return data + noise

class Environment:
    """
    This class simulates an environment for running Braitenberg vehicle trials.
    """
    def __init__(self, sig=0.1, noise_stddev=0.1):
        # Time step for simulation
        self.dt = 0.05  
        # Radius of the vehicle
        self.R = 0.05  
        # Sensor angle in degrees
        self.b = 45  
        # Standard deviation for motor noise
        self.sig = sig  
        # Standard deviation for positional noise
        self.noise_stddev = noise_stddev  

    def run(self, T, agent, motor_gain=0.5, show=True, S=(0, 0)):
        """
        Runs the simulation for time T with the given Braitenberg agent.
        S defines the position of the light source.
        Returns a 2D array of positions and light intensities along the trajectory.
        """
        # Convert genotype parameters from the agent
        w_ll, w_lr, w_rl, w_rr, bl, br = agent.get_geno()
        sl_pos = np.zeros((2, 1))  # left sensor position
        sr_pos = np.zeros((2, 1))  # right sensor position
        S = np.array(S)
        sensor_gain = 0.5  # gain factor for sensors
        vl = 0  # initial left motor speed
        vr = 0  # initial right motor speed

        # Convert initial bearing and sensor angle to radians
        initial_bearing = agent.initial_bearing / 360 * 2 * np.pi
        b = self.b / 360 * 2 * np.pi

        # Preallocate arrays for positions, bearings, and light intensities
        steps = int(T / self.dt)
        pos = np.zeros((2, steps))
        bearing = np.zeros((1, steps))
        pos[:, 0] = agent.pos
        bearing[:, 0] = initial_bearing
        lightIntensity = np.zeros((2, steps))

        for i in range(1, steps):
            # Calculate vehicle's central velocity and angular velocity
            vc = (vl + vr) / 2
            va = (vr - vl) / (2 * self.R)

            # Update position and bearing based on current state
            pos[0, i] = pos[0, i - 1] + self.dt * vc * np.cos(bearing[0, i - 1])
            pos[1, i] = pos[1, i - 1] + self.dt * vc * np.sin(bearing[0, i - 1])
            bearing[0, i] = np.mod(bearing[0, i - 1] + self.dt * va, 2 * np.pi)

            # Calculate left sensor position
            sl_pos[0] = pos[0, i] + self.R * np.cos(bearing[0, i] + b)
            sl_pos[1] = pos[1, i] + self.R * np.sin(bearing[0, i] + b)

            # Calculate right sensor position
            sr_pos[0] = pos[0, i] + self.R * np.cos(bearing[0, i] - b)
            sr_pos[1] = pos[1, i] + self.R * np.sin(bearing[0, i] - b)

            # Calculate distances from the sensors to the light source
            dl = np.sqrt((sl_pos[0] - S[0]) ** 2 + (sl_pos[1] - S[1]) ** 2)
            dr = np.sqrt((sr_pos[0] - S[0]) ** 2 + (sr_pos[1] - S[1]) ** 2)

            # Calculate local light intensities at the sensor positions
            il = sensor_gain / dl
            ir = sensor_gain / dr
            lightIntensity[0, i] = il
            lightIntensity[1, i] = ir

            # Compute motor outputs: weighted sum from sensor values plus bias
            lm = il * w_ll + ir * w_rl + bl
            rm = il * w_lr + ir * w_rr + br

            # Add Gaussian noise to each motor command
            lm = lm + np.random.normal(0, self.sig)
            rm = rm + np.random.normal(0, self.sig)

            # Scale by motor gain
            vl = motor_gain * lm
            vr = motor_gain * rm

        # Add final Gaussian noise to position data
        pos = add_gaussian_noise(pos, mean=0.0, stddev=self.noise_stddev)

        if show:
            self.show(T, pos, sl_pos, sr_pos, bearing, b)
        return pos, lightIntensity

    def show(self, T, pos, sl_pos, sr_pos, bearing, b):
        """
        Plots the vehicle's trajectory along with sensor and body positions.
        """
        plt.plot(pos[0, :], pos[1, :])
        # Final vehicle position and bearing
        x = pos[0, int(T / self.dt) - 1]
        y = pos[1, int(T / self.dt) - 1]
        f_bearing = bearing[0, int(T / self.dt) - 1]

        # Compute final sensor positions
        sl_pos[0] = x + self.R * np.cos(f_bearing + b)
        sl_pos[1] = y + self.R * np.sin(f_bearing + b)
        sr_pos[0] = x + self.R * np.cos(f_bearing - b)
        sr_pos[1] = y + self.R * np.sin(f_bearing - b)

        # Plot light source (yellow dot)
        plt.plot(0, 0, marker='.', markersize=30, color='yellow')
        plt.plot(0, 0, marker='o', markersize=10, color='black')

        # Plot sensor positions (red dots)
        plt.plot(sl_pos[0], sl_pos[1], marker='.', markersize=10, color='red')
        plt.plot(sr_pos[0], sr_pos[1], marker='.', markersize=10, color='red')

        # Plot vehicle body (blue dot and outline)
        plt.plot(x, y, marker='.', markersize=10, color='blue')
        plt.plot(x, y, marker='o', markersize=10, color='black')
        plt.pause(0.05)
        return pos
