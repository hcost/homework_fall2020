import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


if __name__ == '__main__':

    dqn_names = ['hw3_q2_dqn_1_LunarLander-v3_19-10-2020_14-48-13', 'hw3_q2_dqn_2_LunarLander-v3_19-10-2020_14-53-06', 'hw3_q2_dqn_3_LunarLander-v3_19-10-2020_15-10-02']
    ddqn_names = ['hw3_q2_doubledqn_3_LunarLander-v3_19-10-2020_15-10-16', 'hw3_q2_doubledqn_1_LunarLander-v3_19-10-2020_15-08-06', 'hw3_q2_doubledqn_2_LunarLander-v3_19-10-2020_14-43-33']

    for i in range(len(dqn_names)):
        name = dqn_names[i]
        dqn_names[i] = f'data/{name}/events*'

    for i in range(len(ddqn_names)):
        name = ddqn_names[i]
        ddqn_names[i] = f'data/{name}/events*'

    dqn_files = [glob.glob(dqn_names[i])[0] for i in range(len(dqn_names))]
    ddqn_files = [glob.glob(ddqn_names[i])[0] for i in range(len(ddqn_names))]
    dqn_results = [get_section_results(dqn_files[i]) for i in range(len(dqn_names))]
    ddqn_results = [get_section_results(ddqn_files[i]) for i in range(len(ddqn_names))]

    i = 0
    dqn_rets = {}
    for result in dqn_results:
        returns = []
        X, Y = result[0], result[1]
        for j, (x, y) in enumerate(zip(X, Y)):
            returns.append(y)
        dqn_rets[i] = returns
        i += 1

    i = 0
    ddqn_rets = {}
    for result in ddqn_results:
        returns = []
        X, Y = result[0], result[1]
        for j, (x, y) in enumerate(zip(X, Y)):
            returns.append(y)
        ddqn_rets[i] = returns
        i += 1