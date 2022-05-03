# Kalman filter implementation
from pykalman import KalmanFilter
import numpy as np
import sys
from os.path import exists
import matplotlib.pyplot as plt

# KF class
class KF:
    def __init__(self):
        self.initial_state_mean = [-119.807374818, 0, 39.542342345, 0]

        self.transition_matrix = [[1, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]]

        self.observation_matrix = [[1, 0, 0, 0],
                            [0, 0, 1, 0]]

        # created by using moving robot data
        self.sample_observation_covariance = np.array([ [7.47577593e-02, 2.40432407e-14],
                                                [2.40419520e-14, 7.47577593e-02] ])

        self.initial_state_covariance = np.eye(4)

        self.n_dim_state = len(self.transition_matrix)

        self.kf = KalmanFilter(transition_matrices = self.transition_matrix,
                    observation_matrices = self.observation_matrix,
                    initial_state_mean = self.initial_state_mean,
                    observation_covariance = 10*self.sample_observation_covariance,
                    em_vars=['transition_covariance', 'initial_state_covariance'])

        # sample measurements for testing
        self.measurements = np.asarray([ (-119.807377167, 39.542380667),
                                        (-119.807377167, 39.542380667),
                                        (-119.807377167, 39.542380667),
                                        (-119.807378, 39.542380833),
                                        (-119.807378, 39.542380833),
                                        (-119.807378, 39.542380833),
                                        (-119.807377667, 39.542376833),
                                        (-119.807377667, 39.542376833),
                                        (-119.807377667, 39.542376833),
                                        (-119.807377167, 39.542374667),
                                        (-119.807377167, 39.542374667),
                                        (-119.807377167, 39.542374667),
                                        (-119.8073775, 39.542372167),
                                        (-119.8073775, 39.542372167),
                                        (-119.8073775, 39.542372167),
                                        (-119.807381667, 39.54237),
                                        (-119.807381667, 39.54237) ])

    def filterUpdate(self, lat, long, mean, cov):

        measurement = np.asarray( (lat, long) )
        prev_state_mean = mean
        prev_state_cov = cov

        state_mean, state_covariance = (
            self.kf.filter_update(prev_state_mean, prev_state_cov, measurement)
        )

        return state_mean, state_covariance


# Test using previously made measurements

# if __name__ == '__main__':
#     if len(sys.argv) == 3:
        
#         plt.figure(1)

#         #plt.plot(times, measurements[:, 0], 'bo',
#         # times, measurements[:, 1], 'ro',
#         # old_times, filtered_state_means[:, 0], 'b--',
#         # old_times, filtered_state_means[:, 2], 'r--',
#         # new_times, x_new[:, 0], 'b-',
#         # new_times, x_new[:, 2], 'r-')

#         kf = KF()
#         i = 0
#         for measurement in range(len(kf.measurements)):
#             #mean, cov = kf.filterUpdate(kf.measurements[measurement][0], kf.measurements[measurement][1])
#             #print(mean)
#             if exists('last_mean.npy') and exists('last_cov.npy'):
#                 mean, cov = kf.filterUpdate(kf.measurements[measurement][0], kf.measurements[measurement][1], np.load('last_mean.npy'), np.load('last_cov.npy'))
#             else:
#                 mean, cov = kf.filterUpdate(kf.measurements[measurement][0], kf.measurements[measurement][1], kf.initial_state_mean, kf.initial_state_covariance)
#             # mean comes out as masked array, fix to make normal np array
#             mean = np.asarray([mean[0], mean[1], mean[2], mean[3]])
#             print(mean)
#             #print(cov)
#             val = kf.measurements[measurement][1]
#             print(val)
#             #plt.plot(i, mean[2], 'bo')
#             #            i, kf.measurements[measurement][0], 'b--')
#             plt.plot(i, val, 'bo')
#             i += 1
#             np.save('last_mean.npy', mean)
#             np.save('last_cov.npy', cov)

#         plt.savefig("filtered.png")
#         #mean, cov = kf.filterUpdate(lat, long)
        

#         #tempmean = np.load('last_mean.npy')
#         #tempcov = np.load('last_cov.npy')
#         #print(tempcov)
#         #print(tempmean)

#     else:
#         print("Error: incorrect num of args")

# Run as subprocess filter
if __name__ == '__main__':
    if len(sys.argv) == 3:
        # change into correct type
        lat = float(sys.argv[1])
        long = float(sys.argv[2])
        kf = KF()
        # check for existing data, if exists use it, otherwise generate it
        if exists('last_mean.npy') and exists('last_cov.npy'):
            mean, cov = kf.filterUpdate(lat, long, np.load('last_mean.npy'), np.load('last_cov.npy'))
        else:
            mean, cov = kf.filterUpdate(lat, long, kf.initial_state_mean, kf.initial_state_covariance)

        # mean comes out as masked array, fix to make normal np array
        mean = np.asarray([mean[0], mean[1], mean[2], mean[3]])
        
        np.save('last_mean.npy', mean)
        np.save('last_cov.npy', cov)

        # print lat, long to stdout, which is picked up by subprocess.checkoutput in awsros
        print("{}, {}".format(mean[0], mean[2]))

    else:
        print("Error: incorrect num of args")

# ex python3.8 filter.py -119.807374833 39.542342333
