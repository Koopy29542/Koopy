import numpy as np
import george
from george import kernels
from scipy.optimize import minimize


class GaussianProcessPredictor:
    def __init__(self, prediction_len, history_len, dt):

        self._history_len = history_len
        self._prediction_len = prediction_len

        assert self._prediction_len > 0

        # smoothing factor for computing the EMAs of the trajectories
        self._dt = dt

        pass

    def __call__(self, tracking_result):


        dt = self._dt
        history_len = self._history_len
        prediction_len = self._prediction_len

        prediction_result = {}

        for object_id, t in tracking_result.items():
            if not isinstance(t, np.ndarray):
                t = np.array(t)

            if t.shape[0] >= history_len + 1:

                kernel_x = kernels.ExpSquaredKernel(metric=1.0, ndim=2)
                kernel_y = kernels.ExpSquaredKernel(metric=1.0, ndim=2)

                gp_vel_x = george.GP(kernel_x, solver=george.HODLRSolver)
                gp_vel_y = george.GP(kernel_y, solver=george.HODLRSolver)

                v = t[1:] - t[:-1]

                Y_train_x = v[-history_len:, 0] + 1e-4 * np.random.randn(history_len)
                Y_train_y = v[-history_len:, 1] + 1e-4 * np.random.randn(history_len)

                X_train = t[-history_len:]

                # regression
                gp_vel_x.compute(X_train, Y_train_x)
                gp_vel_y.compute(X_train, Y_train_y)

                # hyperparameter optimization
                self._optimize_hyperparameters(gp_vel_x, gp_vel_y, Y_train_x, Y_train_y)

                ps = []
                state = t[-1]
                for t_pred in range(self._prediction_len):

                    # ignore covariance information

                    vel_x, _ = gp_vel_x.predict(y=Y_train_x, t=np.expand_dims(state, axis=0))
                    vel_y, _ = gp_vel_y.predict(y=Y_train_y, t=np.expand_dims(state, axis=0))



                    # single integrator model
                    state += np.concatenate([vel_x, vel_y])

                    ps.append(np.copy(state))

                prediction_result[object_id] = np.array(ps)

                # TODO: EKF or UT (or Invariant KF)

            else:
                # Don't generate prediction for the trajectories that are too short
                prediction_result[object_id] = np.repeat(t[-1:, :], prediction_len, axis=0)
                pass

        return prediction_result

    @staticmethod
    def _optimize_hyperparameters(gp_vel_x, gp_vel_y, Y_train_x, Y_train_y):
        # hyperparameter optimization routine
        # -------------------------------------------------------------------- #
        def nll_x(p):
            # negative log-likelihood
            gp_vel_x.set_parameter_vector(p)
            return -gp_vel_x.log_likelihood(Y_train_x)

        def grad_nll_x(p):
            # grad. of negative log-likelihood
            gp_vel_x.set_parameter_vector(p)
            return -gp_vel_x.grad_log_likelihood(Y_train_x)

        param_x = gp_vel_x.get_parameter_vector()
        results_x = minimize(nll_x, param_x, jac=grad_nll_x, method="L-BFGS-B")
        gp_vel_x.set_parameter_vector(results_x.x)

        def nll_y(p):
            # negative log-likelihood
            gp_vel_y.set_parameter_vector(p)
            return -gp_vel_y.log_likelihood(Y_train_y)

        def grad_nll_y(p):
            # grad. of negative log-likelihood
            gp_vel_y.set_parameter_vector(p)
            return -gp_vel_y.grad_log_likelihood(Y_train_y)

        param_y = gp_vel_y.get_parameter_vector()
        results_y = minimize(nll_y, param_y, jac=grad_nll_y, method="L-BFGS-B")
        gp_vel_y.set_parameter_vector(results_y.x)
        # -------------------------------------------------------------------- #
        return
