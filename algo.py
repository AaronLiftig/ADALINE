class ADALINE:
    def __init__(self, A, B, weight, beta, pattern_rnd_int, weight_rnd_int,
                 desired_outputs = (-1, 1), max_iter=50):
        restart = True
        weight_count = 0
        while restart and weight_count <= max_iter - 1:
            count = 0
            print(f'Current weight (w{weight_count}): {weight}')
            print()
            for a, b in zip(A, B):
                count += 1
                error_E = self.get_error(a, weight, desired_outputs[1])
                if error_E != 0:
                    weight = self.get_new_weight(a, weight, beta, error_E, 
                                                pattern_rnd_int, weight_rnd_int)
                    weight_count += 1
                    print(f'after A{count}')
                    break

                error_E = self.get_error(b, weight, desired_outputs[0])
                if error_E != 0:
                    weight = self.get_new_weight(b, weight, beta, error_E, 
                                                pattern_rnd_int, weight_rnd_int)
                    weight_count += 1
                    print(f'after B{count}')
                    break
            else:
                restart = False
                print('finished')
                exit(1)
        print('Reached max_iter')
        exit(0)

    def get_error(self, pattern, weight, desired_output):
        net_input_I = self.dot_product(pattern, weight)
        if net_input_I > 0:
            output = 1
        else:
            output = -1
        return desired_output - output

    def get_new_weight(self, pattern, weight, beta, error, pattern_rnd_int, weight_rnd_int):
        L = self.get_distance(pattern, pattern_rnd_int)
        delta = (round(beta * error * pattern[0] / L, weight_rnd_int),
                round(beta * error * pattern[1] / L, weight_rnd_int))
        print(f'beta: {beta}, E: {error}, x_0: {pattern[0]}, x_1: {pattern[1]}, L: {L}')
        print(f'Delta: {delta}')
        return (round(weight[0] + delta[0], weight_rnd_int), 
                round(weight[1] + delta[1], weight_rnd_int))

    @staticmethod
    def dot_product(pattern, weight):
        total = 0
        for p, w in zip(pattern, weight):
            total += p * w
        return total

    @staticmethod
    def get_distance(pattern, pattern_rnd_int):
        total = 0 
        for x in pattern:
            total += round(x**2, pattern_rnd_int)
        return round(total, pattern_rnd_int)


# Example from "Understanding Neural Networks, Vol 1: Basic Networks"
# by Maureen Caudill and Charles Butler
A = [(0.3,0.7),(0.4,0.9),(0.5,0.5),(0.7,0.3)]
B = [(-0.6,0.3),(-0.4,-0.2),(0.3,-0.4),(-0.2,-0.8)]
weight = (-0.6,0.8)       
beta = 0.5

ADALINE(A, B, weight, beta, 2, 1)
