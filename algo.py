class ADALINE:
    def __init__(self, A, B, weight, beta, desired_outputs = (-1, 1)):
        restart = True
        weight_count = 0
        while restart:
            count = 0
            print(f'Current weight (w{weight_count}): {weight}')
            print()
            for a, b in zip(A, B):
                count += 1
                error_E = self.get_error(a, weight, desired_outputs[1])
                if error_E != 0:
                    weight = self.get_new_weight(a, weight, beta, error_E)
                    weight_count += 1
                    print(f'after A{count}')
                    break

                error_E = self.get_error(b, weight, desired_outputs[0])
                if error_E != 0:
                    weight = self.get_new_weight(b, weight, beta, error_E)
                    weight_count += 1
                    print(f'after B{count}')
                    break
            else:
                restart = False
                print('finished')

    def get_error(self, pattern, weight, desired_output):
        net_input_I = self.dot_product(pattern, weight)
        if net_input_I > 0:
            output = 1
        else:
            output = -1
        return desired_output - output

    def get_new_weight(self, pattern, weight, beta, error):
        L = self.get_distance(pattern)
        delta = beta * error * pattern[0] / L, beta * error * pattern[1] / L
        return weight[0] + delta[0], weight[1] + delta[1]

    @staticmethod
    def dot_product(pattern, weight):
        total = 0
        for p, w in zip(pattern, weight):
            total += p * w
        return total

    @staticmethod
    def get_distance(pattern):
        total = 0
        for x in pattern:
            total += x**2
        return total


# Example from "Understanding Neural Networks, Vol 1: Basic Networks"
# by Maureen Caudill and Charles Butler
A = [(0.3,0.7),(0.4,0.9),(0.5,0.5),(0.7,0.3)]
B = [(-0.6,0.3),(-0.4,-0.2),(0.3,-0.4),(-0.2,-0.8)]
weight = (-0.6,0.8)       
beta = 0.5

ADALINE(A, B, weight, beta)         
