from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt


class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]


def display_train_stats(fl_stats, communication_rounds):
    clear_output(wait=True)
    plt.figure(figsize=(6, 4))

    acc_mean = np.mean(fl_stats.acc_clients, axis=1)
    acc_std = np.std(fl_stats.acc_clients, axis=1)
    plt.fill_between(
        fl_stats.rounds, acc_mean - acc_std, acc_mean + acc_std, alpha=0.5, color="C0"
    )
    plt.plot(fl_stats.rounds, acc_mean, color="C0")

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")

    plt.xlim(0, communication_rounds)
    plt.ylim(0, 1)

    plt.show()
