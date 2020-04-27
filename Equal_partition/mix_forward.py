import numpy as np
from itertools import combinations
from scipy.special import comb, perm


class mix_forward(object):
    '''
    About the parameters:
    # Nm is the number of modality
    # N is the dimension of stimulus input
    # M is the dimension of contexts input
    # P is the independent sample number of stimulus
    # K is the independent sample number of contexts
    # Nc is the neurons on cortical layer
    '''

    def __init__(self, Nm, N, M, P, K, Nc):
        self.num_modality = Nm
        self.dim_stimuli = int(N)
        self.dim_context = int(M)
        self.num_sti_sample = int(P)
        self.num_con_sample = int(K)
        self.dim_cortical = int(Nc)

    # ------- initialize random data input-------#
    def generate_input(self):

        Nm = self.num_modality
        P = self.num_sti_sample
        K = self.num_con_sample
        N = self.dim_stimuli
        M = self.dim_context

        # binary data, stimulus input
        self.data_0 = np.sign(np.random.randn(P, N))

        for i in range(1, Nm):
            # dynamically generate instance as input data
            self.__dict__[f'data_{i}'] = np.sign(np.random.randn(K, M))

    # -----fix the sparsity in the cortical layer-----#
    def fix_sparsity(self, v, f=0.5):

        threshold = np.sort(v.flatten())[int((1 - f) * v.size)]

        exite = v >= threshold
        inhibit = v < threshold

        v[exite] = 1
        v[inhibit] = 0

        return v

    # ---------generate random_connectin matrixs for each partition--------#
    def random_connection(self, m):

        # ------claim the parameters---------#
        Nm = self.num_modality
        N = self.dim_stimuli
        Nc = self.dim_cortical
        M = self.dim_context

        # generate a dim_list for all modalities, convenient for later calculation of partition dimension
        expr1 = lambda i: N if i == 0 else M
        dim_list = [expr1(i) for i in range(Nm)]

        # total partition number
        p = comb(Nm, m, exact=True)

        # number of partition with task-relevant stimulus
        # p_t = comb(Nm - 1, m - 1, exact=True)

        # total partition list
        p_list = list(combinations(range(Nm), m))

        # dimension of each partion on cortical layer
        dim_order_m = int(Nc / p)

        # --------generate random connection matrix for each partition--------#
        for i in range(1, p):
            # dimension of the partition on input layer
            dim_i = np.sum([dim_list[j] for j in p_list[i]])

            # dynamic variable naming
            self.__dict__[f'J_{i}'] = np.random.normal(0, 1 / np.sqrt(m * dim_i), size=(dim_order_m, dim_i))

        # random connection matrix for first partition, Now Nc dont have to be interger times of p
        dim_0 = np.sum([dim_list[j] for j in p_list[0]])
        self.J_0 = np.random.normal(0, 1 / np.sqrt(m * dim_0), size=(Nc - (p - 1) * dim_order_m, dim_0))

    # -------initialize the input_data and random connection matrix------#
    def initialize(self, m):

        self.generate_input()
        self.random_connection(m)

    # recursively realize the traverse of all independent input data
    def dynloop_rcsn(self, sample_list, mix_layer_data, modality_index=0, sample_index=[]):

        # ------claim the parameters---------#
        Nm = self.num_modality
        p = self.p
        p_list = self.p_list

        # ----------recursion as dynamic loop realization------------------#
        # traverse the stimulus pattern
        for i in range(sample_list[modality_index]):

            # append the index of current sample in current modality
            sample_index.append(i)

            # deepest loop
            if modality_index == Nm - 1:
                # -----------feed forward the current sample, with the sample index------#

                # create a temp array record the vector on cortical layer from current sample
                cort_temp = np.array([])

                ##--------calculate the cortical vector for each partition------##
                # j is the index of current partition
                for j in range(p):

                    # create a temp array record the vector on input layer from current partition
                    input_temp = np.array([])

                    ###-------concatenate to create the input vector for partition j-----#
                    # k is the index of modality in current partition
                    for k in p_list[j]:
                        # sample data of the current modality, self.data_k[sample_index[k]]
                        input_temp = np.concatenate((input_temp, self.__dict__[f'data_{k}'][sample_index[k]]))

                    ###-----feed the input vector forward with the connection matrix self.J_j---#
                    part_cort = np.matmul(self.__dict__[f'J_{j}'], input_temp)

                    cort_temp = np.concatenate((cort_temp, part_cort))

                ##--------write the current vector into the whole effective data matrix----##
                mix_layer_data[self.count, :] = cort_temp
                self.count = self.count + 1

            # ---------if not the deepest loop, then do one more recursion-----#
            else:
                self.dynloop_rcsn(sample_list, mix_layer_data, modality_index + 1, sample_index)

            sample_index.pop()

    ## need to generate_input and random_before run the order_m mixing
    def order_m(self, m, f=0.5, initial_data=False):

        # ------------claim parameters-------------#
        Nm = self.num_modality
        P = self.num_sti_sample
        K = self.num_con_sample
        Nc = self.dim_cortical

        # ------initialize the input_data and random_connection matrix------#
        ## Now J_0 to J_p is the random connection matrix for each partition
        ## self.indata_0 to self.indata_Nm are the input data
        if initial_data: self.generate_input()
        self.random_connection(m)

        ##----generate a sample number list for all modalities, convenient for later mixing-----##
        expr2 = lambda i: P if i == 0 else K
        sample_list = [expr2(i) for i in range(Nm)]

        ##----generate partition number and partition list convenient for later mixing-----##
        self.p = comb(Nm, m, exact=True)  # total partition number
        self.p_list = list(combinations(range(Nm), m))  # total partition list

        ##-----initialize the effective data matrix on cortical layer------##
        mix_layer_data = np.zeros((P * K ** (Nm - 1), Nc))
        self.count = 0

        # -----use recursion to dynamically traverse all possible data and mix feedforwar them-----#
        self.dynloop_rcsn(sample_list, mix_layer_data, modality_index=0, sample_index=[])

        mix_layer_data = self.fix_sparsity(mix_layer_data,f=f)

        return mix_layer_data
