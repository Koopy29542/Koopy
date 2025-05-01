import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    dir_to_save = './fig_dir'
    os.makedirs(dir_to_save, exist_ok=True)
    t = 15

    types = ['alpha', 'coverage', 'interval', 'score']
    models = ['linear', 'gp', 'trajectron', 'eigen', 'koopman']

    colors = ['red', 'purple', 'blue', 'green', 'brown']
    labels = ['Linear', 'GP', 'Trajectron++', 'EigenTrajectory + STGCNN', 'Koopman']
    cp_types = ['split', 'adaptive']

    type_titles = {
        'alpha': r'$\alpha_{{t+{}|t}}$: effective miscoverage'.format(t),
        'coverage': 'coverage',
        'interval': r'$R_{{t+{}|t}}$: interval length'.format(t),
        'score': 'score'


    }

    model_colors = {
        'linear': 'red',
        'gp': 'purple',
        'trajectron': 'blue',
        'eigen': 'green',
        'koopman': 'brown'
    }

    model_labels = {
        'linear': 'Linear',
        'gp': 'GP',
        'trajectron': 'Trajectron++',
        'eigen': 'EigenTrajectory + STGCNN',
        'koopman': 'Koopman'

    }

    cp_linestyles = {'split': 'dashed',
                     'adaptive': 'solid'}

    scenarios_id = [str(i) for i in range(90, 95)]
    scenarios_ood = [str(i) for i in range(95, 97)]


    for type in types:


        # split vs adaptive (for in-distribution)
        plt.clf(), plt.cla()
        fig, ax = plt.subplots()

        for cp in cp_types:
            for model in models:
                model_data = []
                for scenario in scenarios_id:
                    data = np.load(os.path.join('./CP_stats', '{}_{}_{}_{}.npy'.format(type, model, cp, scenario)))
                    model_data.append(data)

                # shape: (scenario length, # prediction horizon)
                model_data_mean = np.mean(model_data, axis=0)
                x = np.arange(model_data_mean.shape[0])

                if cp == 'adaptive':
                    ax.plot(
                        x, model_data_mean[:, t],
                        label=model_labels[model],
                        color=model_colors[model],
                        linestyle=cp_linestyles[cp]
                    )
                else:
                    ax.plot(
                        x, model_data_mean[:, t],
                        color=model_colors[model],
                        linestyle=cp_linestyles[cp]
                    )

        ax.set_xlim(0, 120)
        ax.set_title(type_titles[type])
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        fig.savefig(os.path.join(dir_to_save, '{}_CP.png'.format(type)))
        # id vs ood

    for cp in cp_types:
        for type in types:
            # split vs adaptive (for in-distribution)
            plt.clf(), plt.cla()
            fig, ax = plt.subplots()

            for model in models:
                model_data = []
                for scenario in scenarios_id:
                    data = np.load(os.path.join('./CP_stats', '{}_{}_{}_{}.npy'.format(type, model, cp, scenario)))
                    model_data.append(data)

                # shape: (scenario length, # prediction horizon)
                model_data_mean = np.mean(model_data, axis=0)
                x = np.arange(model_data_mean.shape[0])


                ax.plot(
                    x, model_data_mean[:, t],
                    label=model_labels[model],
                    color=model_colors[model],
                    linestyle='solid'
                )

                model_data = []
                for scenario in scenarios_ood:
                    data = np.load(os.path.join('./CP_stats', '{}_{}_{}_{}.npy'.format(type, model, cp, scenario)))
                    model_data.append(data)

                # shape: (scenario length, # prediction horizon)
                model_data_mean = np.mean(model_data, axis=0)
                x = np.arange(model_data_mean.shape[0])

                ax.plot(
                    x, model_data_mean[:, t],
                    color=model_colors[model],
                    linestyle='dashed'
                )

            ax.set_xlim(0, 120)
            ax.set_title(type_titles[type])
            ax.legend()
            ax.grid(True)

            fig.tight_layout()

            fig.savefig(os.path.join(dir_to_save, '{}_{}_OOD.png'.format(type, cp)))
            # id vs ood


if __name__ == '__main__':
    main()