import numpy as np
from scipy.stats import multivariate_normal
import XML_converter

class HMM:

    def __init__(self, number_of_states, mixture_size, observation_dimension):
        self.number_of_states = number_of_states
        self.mixture_size = mixture_size
        self.observation_dimension = observation_dimension
        self.transition = np.random.rand(number_of_states, number_of_states)
        self.initial_probs = np.random.rand(number_of_states, 1)

        self.transition /= self.transition.sum(axis=1, keepdims=1)
        self.initial_probs /= self.initial_probs.sum()

        self.c = np.random.rand(self.number_of_states, self.mixture_size)
        self.c /= self.c.sum(axis=1, keepdims=1)
        self.mu = np.random.rand(self.number_of_states, self.mixture_size, self.observation_dimension)
        self.variance = np.random.rand(self.number_of_states, self.mixture_size, self.observation_dimension)
        self.tokhmi_shomar = 0

    def set_arrays_per_observation(self, end_of_time, observations):
        self.observations = observations
        self.end_of_time = end_of_time
        self.alpha = np.zeros(shape=(self.number_of_states, end_of_time))
        self.betha = np.zeros(shape=(self.number_of_states, end_of_time))
        self.khi = np.zeros(shape=(self.number_of_states, self.number_of_states, end_of_time - 1))
        self.gamma = np.zeros(shape=(self.number_of_states, end_of_time - 1))
        self.gamma_k = np.zeros(shape=(self.number_of_states, end_of_time, self.mixture_size))

    #called
    def cal_alpha_t_i(self):
        for state_num in range(self.number_of_states):
            self.alpha[state_num, 0] = self.initial_probs[state_num, 0]

        for time in range(1, self.end_of_time):
            for current_state in range(self.number_of_states):
                alpha_mul_transition_i_j = 0
                for prev_state in range(self.number_of_states):
                    alpha_mul_transition_i_j += self.alpha[prev_state, time - 1] * self.transition[prev_state, current_state]
                self.alpha[current_state, time] = alpha_mul_transition_i_j * \
                                                  self.cal_emission_state_and_output(current_state, self.observations[time][:])

    #called
    def cal_betha_t_plus_1_i(self):
        for state_num in range(self.number_of_states):
            self.betha[state_num, self.end_of_time - 1] = 1

        for time in range(self.end_of_time - 2, -1, -1):
            for current_state in range(self.number_of_states):
                betha_mul_transition_i_j = 0
                for next_state in range(self.number_of_states):
                    betha_mul_transition_i_j += self.transition[current_state, next_state] * \
                        self.cal_emission_state_and_output(next_state, self.observations[time + 1][:]) * \
                        self.betha[next_state, time + 1]
                self.betha[current_state, time] = betha_mul_transition_i_j

    #called
    def cal_emission_state_and_output(self, state_num, observation_t):
        self.tokhmi_shomar += 1
        print(self.tokhmi_shomar)
        if self.tokhmi_shomar == 52295:
            print("salam")
        result = 0
        for mixture_index in range(self.mixture_size):
            result += self.c[state_num, mixture_index] * multivariate_normal.pdf(observation_t,
                                                                             self.mu[state_num, mixture_index, :],
                                                                             self.variance[state_num, mixture_index, :])
        return result

    #called
    def cal_khi_t_i_j(self):
        for time in range(self.end_of_time - 1):
            denominator_in_t = 0
            for state_num in range(self.number_of_states):
                denominator_in_t += self.alpha[state_num, time] * self.betha[state_num, time]

            for state_i in range(self.number_of_states):
                for state_j in range(self.number_of_states):
                    self.khi[state_i, state_j, time] = self.alpha[state_i][time] * self.transition[state_i][state_j] * \
                                                       self.cal_emission_state_and_output(state_j, self.observations[time + 1][:]) \
                                                       * self.betha[state_j][time + 1] / denominator_in_t
                    if self.khi[state_i, state_j, time] == 0:
                        print("salam")

    #called
    def cal_gamma_t_i(self):
        for time in range(self.end_of_time - 1):
            for state_i in range(self.number_of_states):
                khi_sum = 0
                for state_j in range(self.number_of_states):
                    khi_sum += self.khi[state_i, state_j, time]
                self.gamma[state_i, time] = khi_sum

    #called
    def cal_num_of_mov_from_i(self, state_num):
        result = 0
        for time in range(self.end_of_time - 1):
            result += self.gamma[state_num, time]

        return result

    #called
    def cal_num_of_mov_from_i_to_j(self, state_i, state_j):
        result = 0
        for time in range(self.end_of_time - 1):
            result += self.khi[state_i, state_j, time]
        return result

    #called
    def cal_gamma_t_i_k(self):
        for time in range(self.end_of_time):
            for state_i in range(self.number_of_states):
                left_denominator = 0
                for state_j in range(self.number_of_states):
                    left_denominator += self.alpha[state_j][time] * self.betha[state_j][time]

                right_denominator = 0
                for mixture_index in range(self.mixture_size):
                    right_denominator += self.c[state_i, mixture_index] * multivariate_normal.pdf(self.observations[time][:],
                                                                                                  self.mu[state_i, mixture_index, :],
                                                                                                  self.variance[state_i, mixture_index, :])

                left_part_of_equation = self.alpha[state_i, time] * self.betha[state_i, time] / left_denominator

                for mixture_index in range(self.mixture_size):
                    right_part_of_equation = self.c[state_i, mixture_index] * multivariate_normal.pdf(self.observations[time][:],
                                                                                                      self.mu[state_i, mixture_index, :],
                                                                                                      self.variance[state_i, mixture_index, :])\
                                             / right_denominator
                    self.gamma_k[state_i, time, mixture_index] = left_part_of_equation * right_part_of_equation
                    if self.gamma_k[state_i,:,0].sum() == 0.0 and state_i == 19 and mixture_index == 0:
                        print("daal ensemble")


    #called
    def update_initial_prob(self):
        for state in range(self.number_of_states):
            self.initial_probs[state, 0] = self.gamma[state, 0]

    #called
    def update_transition_prob(self):
        for state_i in range(self.number_of_states):
            denominator = self.cal_num_of_mov_from_i(state_i)
            for state_j in range(self.number_of_states):
                self.transition[state_i, state_j] = self.cal_num_of_mov_from_i_to_j(state_i, state_j) / denominator

    #called
    def update_emission_prob(self):
        for state_num in range(self.number_of_states):
            denominator = 0
            for time in range(self.end_of_time):
                for mixture_index in range(self.mixture_size):
                    denominator += self.gamma_k[state_num, time, mixture_index]

            for mixture_index in range(self.mixture_size):
                t_sum = 0
                c_t_sum = [0, 0]
                var_t_sum = [0, 0]
                for time in range(self.end_of_time):
                    t_sum += self.gamma_k[state_num, time, mixture_index]
                    c_t_sum += self.gamma_k[state_num, time, mixture_index] * self.observations[time][:]
                    o_t_minus_mu_hat_j_k = (self.observations[time] - self.mu[state_num,mixture_index])
                    var_t_sum += self.gamma_k[state_num, time, mixture_index] * np.matmul(o_t_minus_mu_hat_j_k,
                                                                                          o_t_minus_mu_hat_j_k.transpose())
                self.c[state_num, mixture_index] = t_sum / denominator

                self.mu[state_num, mixture_index] = c_t_sum / denominator
                self.variance[state_num, mixture_index] = var_t_sum / denominator

                if self.mu[state_num, mixture_index,:].sum() == 0.0 and state_num == 19 and mixture_index == 0:
                    print("peiroe zartosht bodi ya masih")

np.random.seed(0)
vowels = ["a", "e", "i", "o", "u"]
vowel_train = {}
vowel_test = {}
vowel_HMM = {}
NUMBER_OF_STATES = 25
MIXTURE_SIZE = 3
OBSERVATION_DIMENSION = 2
for vowel in vowels:
    train, test = XML_converter.get_train_and_test(vowel)
    vowel_train[vowel] = train
    vowel_test[vowel] = test
    vowel_HMM[vowel] = HMM(number_of_states=NUMBER_OF_STATES, mixture_size=MIXTURE_SIZE, observation_dimension=OBSERVATION_DIMENSION)

for vowel in vowels:
    model = vowel_HMM[vowel]

    #t#odo khate paEno pak
    #model = HMM(number_of_states=NUMBER_OF_STATES, mixture_size=MIXTURE_SIZE, observation_dimension=OBSERVATION_DIMENSION)
    train_set = vowel_train[vowel]
    test_set = vowel_test[vowel]

    for observations in train_set:
        model.set_arrays_per_observation(end_of_time=len(observations),observations=observations)
        model.cal_log_alpha_t_i()
        model.cal_betha_t_plus_1_i()
        model.cal_khi_t_i_j()
        model.cal_gamma_t_i()
        model.cal_gamma_t_i_k()
        model.update_initial_prob()
        model.update_transition_prob()
        model.update_emission_prob()
