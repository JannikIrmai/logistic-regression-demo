import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent, MouseEvent


def compute_loss(x0, x1, theta):
    loss = np.zeros_like(theta)
    for x in x0:
        loss += np.log2(1 + 2**(theta*x))
    for x in x1:
        loss += - theta*x + np.log2(1 + 2**(theta*x))
    return loss


def compute_regularizer(theta, sigma):
    return theta**2 * np.log2(np.e) / (2 * sigma**2)


class LogRegDemo:

    def __init__(self):
        self.n0 = 10
        self.n1 = 10

        self.x0 = np.linspace(0, 1-self.n0, self.n0)
        self.x1 = np.linspace(0, self.n1-1, self.n1)
        self.data = [self.x0, self.x1]

        self.sigma = 1
        self.theta = np.linspace(-10, 10, 1001)
        self.x = np.linspace(-self.n0, self.n1, 1001)

        self.loss = compute_loss(self.x0, self.x1, self.theta)
        self.regularizer = compute_regularizer(self.theta, self.sigma)
        self.objective = self.loss + self.regularizer

        self.opt = self.theta[np.argmin(self.objective)]
        self.prob = 1 / (1 + 2**(-self.opt*self.x))

        self.fig, self.ax = plt.subplots(2)
        # axis for data and logistic distribution
        self.ax[0].set_xlabel("x")
        self.ax[0].set_ylabel("y")
        # axis for objective function
        self.ax[1].set_xlabel(r"$\theta$")
        self.fig.tight_layout()

        # plot data and logistic curve
        self.prob_handle = self.ax[0].plot(self.x, self.prob, color="black")[0]
        self.scatter0_handle = self.ax[0].scatter(self.x0, [0] * self.n0, color="tab:blue", picker=True)
        self.scatter1_handle = self.ax[0].scatter(self.x1, [1] * self.n1, color="tab:red", picker=True)
        self.scatters = [self.scatter0_handle, self.scatter1_handle]

        # plot loss, regularizer and objective
        self.loss_handle = self.ax[1].plot(self.theta, self.loss, label="loss")[0]
        self.regularizer_handle = self.ax[1].plot(self.theta, self.regularizer, label="regularizer")[0]
        self.objective_handle = self.ax[1].plot(self.theta, self.objective, label="objective")[0]
        self.ax[1].legend()

        self.selected = [None, 0]

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        plt.show()

    def on_pick(self, event: PickEvent):
        self.selected[0] = 0 if event.artist == self.scatter0_handle else 1
        self.selected[1] = event.ind[0]

    def on_release(self, event: MouseEvent):
        self.selected[0] = None

    def on_move(self, event: MouseEvent):
        if self.selected[0] is None:
            return
        if event.xdata is None:
            return
        self.data[self.selected[0]][self.selected[1]] = event.xdata
        self.update()

    def update(self):
        for i in range(2):
            self.scatters[i].set_offsets(np.c_[self.data[i], [i]*self.data[i].size])

        self.loss = compute_loss(self.x0, self.x1, self.theta)
        self.regularizer = compute_regularizer(self.theta, self.sigma)
        self.objective = self.loss + self.regularizer
        self.loss_handle.set(ydata=self.loss)
        self.regularizer_handle.set(ydata=self.regularizer)
        self.objective_handle.set(ydata=self.objective)

        self.opt = self.theta[np.argmin(self.objective)]
        self.prob = 1 / (1 + 2 ** (-self.opt * self.x))
        self.prob_handle.set(ydata=self.prob)

        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    demo = LogRegDemo()

