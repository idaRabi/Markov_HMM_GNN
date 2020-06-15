import numpy as np
from scipy.stats import multivariate_normal
import XML_converter
import math
import sys

class HMM:

    def __init__(self, number_of_states, mixture_size, observation_dimension):
        self.khiar = 0


        self.number_of_states = number_of_states
        self.mixture_size = mixture_size
        self.observation_dimension = observation_dimension
        self.log_transition = np.random.rand(number_of_states, number_of_states)
        self.log_initial_probs = np.random.rand(number_of_states, 1)

        self.log_transition /= self.log_transition.sum(axis=1, keepdims=1)
        self.log_transition = np.log(self.log_transition)
        self.log_initial_probs /= self.log_initial_probs.sum()
        self.log_initial_probs = np.log(self.log_initial_probs)

        self.log_c = np.random.rand(self.number_of_states, self.mixture_size)
        self.log_c /= self.log_c.sum(axis=1, keepdims=1)
        self.log_c = np.log(self.log_c)

        self.mu = np.random.rand(self.number_of_states, self.mixture_size, self.observation_dimension)
        self.variance = np.random.rand(self.number_of_states, self.mixture_size, self.observation_dimension)

    def set_arrays_per_observation(self, end_of_time, observations):
        self.observations = observations
        self.end_of_time = end_of_time
        self.log_alpha = np.zeros(shape=(self.number_of_states, end_of_time))
        self.log_betha = np.zeros(shape=(self.number_of_states, end_of_time))
        self.khi = np.zeros(shape=(self.number_of_states, self.number_of_states, end_of_time - 1))
        self.gamma = np.zeros(shape=(self.number_of_states, end_of_time - 1))
        self.gamma_k = np.zeros(shape=(self.number_of_states, end_of_time, self.mixture_size))

    def get_sum_alpha_in_end_of_time(self):
        result = 0
        for state_num in range(self.number_of_states):
            result += self.log_alpha[state_num, self.end_of_time - 1]
        return result


    #todo shaiad az log back up nemi
    #called + logalized
    def cal_log_alpha_t_i(self):
        for state_num in range(self.number_of_states):
            self.log_alpha[state_num, 0] = self.log_initial_probs[state_num, 0]

        for time in range(1, self.end_of_time):
            for current_state in range(self.number_of_states):
                log_alpha_plus_log_transition_i_j = []
                for prev_state in range(self.number_of_states):
                    log_alpha_plus_log_transition_i_j.append(self.log_alpha[prev_state, time - 1] + self.log_transition[prev_state, current_state])
                log_alpha_plus_log_transition_i_j = np.array(log_alpha_plus_log_transition_i_j) + \
                                                      self.cal_log_emission_state_and_output(current_state, self.observations[time][:])
                self.log_alpha[current_state, time] = self.max_log_sum(log_alpha_plus_log_transition_i_j)


    #called + logalized
    def cal_betha_t_plus_1_i(self):
        for state_num in range(self.number_of_states):
            self.log_betha[state_num, self.end_of_time - 1] = math.log(1)

        for time in range(self.end_of_time - 2, -1, -1):
            for current_state in range(self.number_of_states):
                log_betha_plus_log_transition_i_j = []
                for next_state in range(self.number_of_states):
                    log_betha_plus_log_transition_i_j.append(self.log_transition[current_state, next_state] + \
                                                self.cal_log_emission_state_and_output(next_state, self.observations[time + 1][:]) + \
                                                self.log_betha[next_state, time + 1])
                log_betha_plus_log_transition_i_j = np.array(log_betha_plus_log_transition_i_j)
                max_index = np.unravel_index(np.argmax(log_betha_plus_log_transition_i_j, axis=None), log_betha_plus_log_transition_i_j.shape)
                maxx = log_betha_plus_log_transition_i_j[max_index]
                log_betha_plus_log_transition_i_j -= maxx
                exp_log_betha_plus_log_transition_i_j = np.exp(log_betha_plus_log_transition_i_j)
                sum_exp_log_betha_plus_log_transition_i_j = np.log(exp_log_betha_plus_log_transition_i_j.sum())
                self.log_betha[current_state, time] = maxx + sum_exp_log_betha_plus_log_transition_i_j

    #called + logalized
    def cal_log_emission_state_and_output(self, state_num, observation_t):
        list = []
        for mixture_index in range(self.mixture_size):
            list.append(self.log_c[state_num, mixture_index] + np.log(multivariate_normal.pdf(observation_t,
                                                                             self.mu[state_num, mixture_index, :],
                                                                          self.variance[state_num, mixture_index, :], allow_singular=True)))
        result = self.max_log_sum(list)
        return result

    def max_log_sum(self, array):
        array = np.array(array)
        max_index = np.unravel_index(np.argmax(array,axis=None), array.shape)
        maxx = array[max_index]
        array -= maxx
        array = np.exp(array)
        arrayResult = np.log(array.sum())
        return maxx + arrayResult

    #called + logalized
    def cal_khi_t_i_j(self):
        for time in range(self.end_of_time - 1):
            denominator_in_t = []
            for state_num in range(self.number_of_states):
                denominator_in_t.append(self.log_alpha[state_num, time] + self.log_betha[state_num, time])

            denominator_in_t = self.max_log_sum(denominator_in_t)

            for state_i in range(self.number_of_states):
                khiArray = []
                for state_j in range(self.number_of_states):
                    self.khi[state_i, state_j, time] = self.log_alpha[state_i][time] + \
                                                       self.log_transition[state_i][state_j] + \
                                                       self.cal_log_emission_state_and_output(state_j, self.observations[time + 1][:]) + \
                                                       self.log_betha[state_j][time + 1] - denominator_in_t

    #called + logalized
    def cal_gamma_t_i(self):
        for time in range(self.end_of_time - 1):
            for state_i in range(self.number_of_states):
                khi_sum = []
                for state_j in range(self.number_of_states):
                    khi_sum.append(self.khi[state_i, state_j, time])
                self.gamma[state_i, time] = self.max_log_sum(khi_sum)

    #called + logalized
    def cal_num_of_mov_from_i(self, state_num):
        result = []
        for time in range(self.end_of_time - 1):
            result.append(self.gamma[state_num, time])

        return self.max_log_sum(result)

    #called + logalized
    def cal_num_of_mov_from_i_to_j(self, state_i, state_j):
        result = []
        for time in range(self.end_of_time - 1):
            result.append(self.khi[state_i, state_j, time])

        return self.max_log_sum(result)

    #called + logalized
    def cal_gamma_t_i_k(self):
        for time in range(self.end_of_time):
            for state_i in range(self.number_of_states):
                left_denominator = []
                for state_j in range(self.number_of_states):
                    left_denominator.append(self.log_alpha[state_j][time] + self.log_betha[state_j][time])

                left_denominator = self.max_log_sum(left_denominator)

                right_denominator = []
                for mixture_index in range(self.mixture_size):
                    right_denominator.append(self.log_c[state_i, mixture_index] + np.log(multivariate_normal.pdf(self.observations[time][:],
                                                                                                  self.mu[state_i, mixture_index, :],
                                                                                                  self.variance[state_i, mixture_index, :],
                                                                                                                 allow_singular=True)))
                right_denominator = self.max_log_sum(right_denominator)

                left_part_of_equation = self.log_alpha[state_i, time] + self.log_betha[state_i, time] - left_denominator

                for mixture_index in range(self.mixture_size):
                    right_part_of_equation = self.log_c[state_i, mixture_index] + np.log(multivariate_normal.pdf(self.observations[time][:],
                                                                                                      self.mu[state_i, mixture_index, :],
                                                                                                      self.variance[state_i, mixture_index, :],
                                                                                                                 allow_singular=True)) \
                                             - right_denominator
                    self.gamma_k[state_i, time, mixture_index] = left_part_of_equation + right_part_of_equation
                    if math.isnan(self.gamma_k[state_i, time, mixture_index]):
                        print("2")

    #called + logalized
    def update_initial_prob(self):
        for state in range(self.number_of_states):
            self.log_initial_probs[state, 0] = self.gamma[state, 0]

    #called + logalized
    def update_transition_prob(self):
        for state_i in range(self.number_of_states):
            denominator = self.cal_num_of_mov_from_i(state_i)
            for state_j in range(self.number_of_states):
                self.log_transition[state_i, state_j] = self.cal_num_of_mov_from_i_to_j(state_i, state_j) - denominator

    #called + logalized
    def update_emission_prob(self):
        for state_num in range(self.number_of_states):
            denominator = []
            for time in range(self.end_of_time):
                for mixture_index in range(self.mixture_size):
                    denominator.append(self.gamma_k[state_num, time, mixture_index])

            denominator = self.max_log_sum(denominator)

            for mixture_index in range(self.mixture_size):
                t_sum = []
                c_t_sum = [0, 0]
                var_t_sum = [0, 0]
                for time in range(self.end_of_time):
                    t_sum.append(self.gamma_k[state_num, time, mixture_index])
                    c_t_sum += np.exp(self.gamma_k[state_num, time, mixture_index]) * self.observations[time][:]
                    o_t_minus_mu_hat_j_k = (self.observations[time] - self.mu[state_num,mixture_index])
                    var_t_sum += np.exp(self.gamma_k[state_num, time, mixture_index]) * np.matmul(o_t_minus_mu_hat_j_k,
                                                                                          o_t_minus_mu_hat_j_k.transpose())
                t_sum = self.max_log_sum(t_sum)

                self.log_c[state_num, mixture_index] = t_sum - denominator

                self.mu[state_num, mixture_index] = c_t_sum / np.exp(denominator)
                self.variance[state_num, mixture_index] = var_t_sum / np.exp(denominator)


np.random.seed(0)
vowels = ["a", "i", "e"]
vowel_train = {}
vowel_test = {}
vowel_HMM = {}
NUMBER_OF_STATES = 10
MIXTURE_SIZE = 1
OBSERVATION_DIMENSION = 2
for vowel in vowels:
    train, test = XML_converter.get_train_and_test(vowel)
    vowel_train[vowel] = train
    vowel_test[vowel] = test
    vowel_HMM[vowel] = HMM(number_of_states=NUMBER_OF_STATES, mixture_size=MIXTURE_SIZE, observation_dimension=OBSERVATION_DIMENSION)

for vowel in vowels:
    model = vowel_HMM[vowel]
    print("training vowel %s" % vowel)
    #t#odo khate paEno pak
    #model = HMM(number_of_states=NUMBER_OF_STATES, mixture_size=MIXTURE_SIZE, observation_dimension=OBSERVATION_DIMENSION)
    train_set = vowel_train[vowel]

    for epoch_num in range(2):
        for observations in train_set:
            model.set_arrays_per_observation(end_of_time=len(observations), observations=observations)
            model.cal_log_alpha_t_i()
            model.cal_betha_t_plus_1_i()
            model.cal_khi_t_i_j()
            model.cal_gamma_t_i()
            model.cal_gamma_t_i_k()
            model.update_initial_prob()
            model.update_transition_prob()
            model.update_emission_prob()
        print("training epoch num: %d" % epoch_num)

# testing
print("testing")
confusion_matrix = np.zeros(shape=(len(vowels),len(vowels)))
for test_vowel in vowels:
    print("testing vowel %s" % test_vowel)
    test_set = vowel_test[test_vowel]
    #test_set = vowel_train[test_vowel]
    for test_observations in test_set:
        models_result = []
        for trained_vowel in vowels:
            trained_model = vowel_HMM[trained_vowel]
            trained_model.set_arrays_per_observation(end_of_time=len(test_observations), observations=test_observations)
            trained_model.cal_log_alpha_t_i()
            models_result.append(trained_model.get_sum_alpha_in_end_of_time())

        selected_vowel = models_result.index(max(models_result))
        confusion_matrix[vowels.index(test_vowel)][selected_vowel] += 1

print(confusion_matrix)