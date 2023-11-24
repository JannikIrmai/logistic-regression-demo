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
        self.n = [10, 10]

        self.data = [np.linspace(0, 1-self.n[0], self.n[0]), np.linspace(0, self.n[1]-1, self.n[1])]

        self.sigma = 1
        self.theta = np.linspace(-10, 10, 1001)
        self.x = np.linspace(-self.n[0], self.n[1], 1001)

        self.loss = compute_loss(self.data[0], self.data[1], self.theta)
        self.regularizer = compute_regularizer(self.theta, self.sigma)
        self.objective = self.loss + self.regularizer

        self.opt = self.theta[np.argmin(self.objective)]
        self.prob = 1 / (1 + 2**(-self.opt*self.x))

        self.fig = plt.figure()
        self.ax_data = self.fig.add_axes([0.25, 0.6, 0.7, 0.35])
        self.ax_obj = self.fig.add_axes([0.25, 0.1, 0.7, 0.35])
        self.fig.subplots_adjust(left=0.25)
        # axis for data and logistic distribution
        self.ax_data.set_xlabel("x")
        self.ax_data.set_ylabel("y")
        # axis for objective function
        self.ax_obj.set_xlabel(r"$\theta$")

        # add sliders
        self.ax_sigma_slider = self.fig.add_axes([0.025, 0.1, 0.025, 0.8])
        self.sigma_slider = plt.Slider(ax=self.ax_sigma_slider, label=r"$\sigma$", valmin=-10, valmax=10,
                                       valinit=np.log2(self.sigma), orientation="vertical")
        self.sigma_slider.on_changed(self.on_slide_sigma)
        self.ax_n0_slider = self.fig.add_axes([0.075, 0.1, 0.025, 0.8])
        self.n0_slider = plt.Slider(ax=self.ax_n0_slider, label=r"n0", valmin=0, valmax=self.n[0],
                                    valinit=self.n[0], orientation="vertical", valstep=1)
        self.n0_slider.on_changed(self.on_slide_n0)
        self.ax_n1_slider = self.fig.add_axes([0.125, 0.1, 0.025, 0.8])
        self.n1_slider = plt.Slider(ax=self.ax_n1_slider, label=r"n1", valmin=0, valmax=self.n[1],
                                    valinit=self.n[1], orientation="vertical", valstep=1)
        self.n1_slider.on_changed(self.on_slide_n1)

        # plot data and logistic curve
        self.prob_handle = self.ax_data.plot(self.x, self.prob, color="black")[0]
        self.scatter0_handle = self.ax_data.scatter(self.data[0], [0] * self.n[0], color="tab:blue", picker=True)
        self.scatter1_handle = self.ax_data.scatter(self.data[1], [1] * self.n[1], color="tab:red", picker=True)
        self.scatters = [self.scatter0_handle, self.scatter1_handle]

        # plot loss, regularizer and objective
        self.loss_handle = self.ax_obj.plot(self.theta, self.loss, label="loss")[0]
        self.regularizer_handle = self.ax_obj.plot(self.theta, self.regularizer, label="regularizer")[0]
        self.objective_handle = self.ax_obj.plot(self.theta, self.objective, label="objective")[0]
        self.ax_obj.legend()

        self.selected = [None, 0]

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_slide_sigma(self, val):
        self.sigma = 2**val
        self.sigma_slider.valtext.set_text(f"{self.sigma:.3f}")
        self.update()

    def on_slide_n0(self, val):
        self.n[0] = val
        self.update()

    def on_slide_n1(self, val):
        self.n[1] = val
        self.update()

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
            alpha = np.zeros_like(self.data[i])
            alpha[:self.n[i]] = 1
            self.scatters[i].set_alpha(alpha)

        self.loss = compute_loss(self.data[0][:self.n[0]], self.data[1][:self.n[1]], self.theta)
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
    plt.show()

