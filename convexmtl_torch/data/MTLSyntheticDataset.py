import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint

from icecream import ic

from abc import ABC, abstractmethod

class MTLSyntheticRegressionDataset(ABC):
    """
    A class to generate a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise to add to the target.
    random_state : int or None
        The seed to use for the random number generator.
    """

    def __init__(self, n_samples_per_task, noise, n_tasks, random_state, start, stop):
        self.n_samples_per_task = n_samples_per_task
        self.noise = noise
        self.n_tasks = n_tasks
        self.random_state = random_state
        self.start = start
        self.stop = stop
        np.random.seed(self.random_state)
        self.tasks = self._generate_tasks()
        self.colors = self.get_colors()
        self.task_functions = self._generate_task_functions()
        self.X, self.y = self._generate_data()

    def _generate_data(self):
        """
        Generate the synthetic data.
        """        
        m_t = self.n_samples_per_task
        N = m_t * self.n_tasks
        
        x_l = []
        y_l = []
        t_l = []
        dic_w = {}

        rng = np.random.RandomState(self.random_state)

        for t in self.tasks:            
            f_t = self.task_functions[t]
            x_task = np.linspace(self.start, self.stop, m_t)[:, None]

            
            y_task = self._function_task(x_task, t) + + rng.randn(self.n_samples_per_task)[:, None] * self.noise
            t_task = np.array([t] * len(y_task)).reshape(y_task.shape)

            x_l.append(x_task)
            y_l.append(y_task)
            t_l.append(t_task)


        X_data = np.concatenate(x_l, axis=0)
        ic(X_data.shape)
        t = np.concatenate(t_l, axis=0)
        X = np.concatenate((X_data, t), axis=1)
        y = np.concatenate(y_l, axis=0).flatten()

        return X, y

    def generate_new_data(self, n_samples_per_task=None, n_tasks=None, noise=None, random_state=None):
        """
        Generates new dataset with given parameters.
        """
        if n_samples_per_task is None:
            n_samples_per_task = self.n_samples_per_task
        if n_tasks is None:
            n_tasks = self.n_tasks
        if noise is None:
            noise = self.noise
        if random_state is None:
            random_state = self.random_state
        self.n_samples_per_task = n_samples_per_task
        self.n_tasks = n_tasks
        self.noise = noise
        self.random_state = random_state
        self.X, self.y = self._generate_data()

    # def _generate_data(self):
    #     """
    #     Generate the synthetic data.
    #     """
    #     rng = np.random.RandomState(self.random_state)
    #     X = rng.rand(self.n_samples, 1)
    #     y = 2 * X.ravel() + rng.randn(self.n_samples) * self.noise
    #     return X, y

    # def generate_new_data(self, n_samples=None, noise=None, random_state=None):
    #     """
    #     Generates new dataset with given parameters.
    #     """
    #     if n_samples is None:
    #         n_samples = self.n_samples
    #     if noise is None:
    #         noise = self.noise
    #     if random_state is None:
    #         random_state = self.random_state
    #     self.n_samples = n_samples
    #     self.noise = noise
    #     self.random_state = random_state
    #     self.X, self.y = self._generate_data()

    def plot_functions(self, ax=None, show_legend=True):
        x_linsp = np.linspace(self.start, self.stop)
        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = plt.gca()
        # plt.plot(np.linspace(self.start, self.stop), f(np.linspace(self.start, self.stop)), color='black', label='sin(x)')
        for t in self.tasks:
            f_t = self.task_functions[t]
            ax.plot(x_linsp, self._function_task(x_linsp, t), color=self._get_color(t), linestyle='--', label=t)
            #plt.scatter(data_x[i*m_r:(i+1)*m_r], data_y[i*m_r:(i+1)*m_r], label='task {}'.format(t), color=self._get_color(t))
        if show_legend:
            plt.legend()
        # plt.title('Multi-Task Toy Problem'.format(data_name))
        # plt.tight_layout()
        # plt.savefig('toyproblem_{}.pdf'.format(data_name))

    def plot_data(self, X, y, ax=None, show_legend=True):
        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = plt.gca()
        # plt.plot(np.linspace(self.
        X_data = X[:, :-1]
        t_col = X[:, -1]
        for t in self.tasks:
            t_idx = (t_col == t)
            ax.scatter(X_data[t_idx].astype(float), y[t_idx], color=self._get_color(t), marker=self._get_marker(t), label=t)
            #plt.scatter(data_x[i*m_r:(i+1)*m_r], data_y[i*m_r:(i+1)*m_r], label='task {}'.format(t), color=self._get_color(t))
        if show_legend:
            plt.legend()
        # plt.title('Multi-Task Toy Problem'.format(data_name))
        # plt.tight_layout()
        # plt.savefig('toyproblem_{}.pdf'.format(data_name))

    def plot_data_functions(self, X, y, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = plt.gca()
        self.plot_data(X, y, ax=ax)
        self.plot_functions(ax=ax, show_legend=False)


class MTLFunctionsRegression(MTLSyntheticRegressionDataset):
    """
    A class to generate a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise to add to the target.
    random_state : int or None
        The seed to use for the random number generator.
    """   
    def __init__(self, n_samples_per_task=25, noise=0.1, n_tasks=4, 
                 random_state=42, start=1., stop=10.):
        super().__init__(n_samples_per_task=n_samples_per_task, 
                         noise=noise, n_tasks=n_tasks, 
                         random_state=random_state, start=start, stop=stop)

    def get_colors(self):
        colors = {t: cm.tab10(t) for t in self.tasks}
        return colors
    
    def _get_color(self, t):
        return self.colors[t]
    
    def _get_marker(self, t):
        return "."
        
    def _generate_tasks(self):
        return range(self.n_tasks)
    
    def _function_task(self, x, t=None):
        y = np.sin(x)
        if t is not None:
            y += (1e-6 * self.fun_tasks[t](x))
        return y

    def _generate_task_functions(self):
        self.fun_tasks = {}
        for t in self.tasks:
            shape = np.random.randint(1, 10, size=1)
            coefs = np.random.randn(shape[0])
            self.fun_tasks[t] = np.poly1d(coefs)

        ic(self.fun_tasks)

        task_functions = {}
        for t in self.tasks:
            task_functions[t] = (lambda x: self._function_task(x, t=t))

        return task_functions

    

class MTLClustersARegression(MTLSyntheticRegressionDataset):
    """
    A class to generate a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise to add to the target.
    random_state : int or None
        The seed to use for the random number generator.
    """

    def __init__(self, n_samples_per_task=25, noise=0.1, n_tasks=3,
                  random_state=42, start=-2., stop=2.):
        super().__init__(n_samples_per_task=n_samples_per_task, 
                         noise=noise, n_tasks=n_tasks, 
                         random_state=random_state, start=start, stop=stop)
        
    def get_colors(self):
        colors = {t: cm.tab10(int(t/10)) for t in self.tasks}
        ic(colors)
        # print(colors)
        return colors
    
    def _get_color(self, t):
        color = cm.tab10(int(t/10))
        return color
    
    def _get_marker(self, t):
        markers = ['.', 'o', 'v', '*', '^']
        return markers[int(t%10)]
        
    def _generate_tasks(self):
        tasks = []
        for t in range(self.n_tasks):
            m_t = randint(1,5)
            tasks.extend([t*10+j for j in range(m_t)])
        ic(tasks)
        return tasks
        
    def _function_task(self, x, t=None):
        real_t = int(t/10)
        y = self.fun_tasks[real_t](x)
        # if t is not None:
        #     y += (1e-6 * self.fun_tasks[t](x))
        return y

    def _generate_task_functions(self):
        self.fun_tasks = {
            0: np.vectorize(lambda x: 10 * np.sin(x)),
            1: np.vectorize(lambda x: x**2),
            2: np.vectorize(lambda x: x**3)
        }

        task_functions = {}
        for t in self.tasks:
            task_functions[t] = (lambda x: self._function_task(x, t=t))

        return task_functions


class MTLCommonRegression(MTLSyntheticRegressionDataset):
    """
    A class to generate a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise to add to the target.
    random_state : int or None
        The seed to use for the random number generator.
    """

    def __init__(self, n_samples_per_task=25, noise=0.1, n_tasks=1,
                  random_state=42, start=-2., stop=2.):
        super().__init__(n_samples_per_task=n_samples_per_task, 
                         noise=noise, n_tasks=n_tasks, 
                         random_state=random_state, start=start, stop=stop)
        
    def get_colors(self):
        colors = {t: cm.tab10(int(t/10)) for t in self.tasks}
        ic(colors)
        # print(colors)
        return colors
    
    def _get_color(self, t):
        color = cm.tab10(int(t/10))
        return color
    
    def _get_marker(self, t):
        markers = ['.', 'o', 'v', '*', '^', '1']
        return markers[int(t%10)]
        
    def _generate_tasks(self):
        tasks = []
        for t in range(self.n_tasks):
            m_t = randint(3,5)
            tasks.extend([t*10+j for j in range(m_t)])
        ic(tasks)
        return tasks
        
    def _function_task(self, x, t=None):
        real_t = int(t/10)
        y = self.fun_tasks[real_t](x)
        # if t is not None:
        #     y += (1e-6 * self.fun_tasks[t](x))
        return y

    def _generate_task_functions(self):
        self.fun_tasks = {
            0: np.vectorize(lambda x: 10 * np.sin(x)),
            1: np.vectorize(lambda x: x**2),
            2: np.vectorize(lambda x: x**3)
        }

        task_functions = {}
        for t in self.tasks:
            task_functions[t] = (lambda x: self._function_task(x, t=t))

        return task_functions
    

class MTLIndependentRegression(MTLSyntheticRegressionDataset):
    """
    A class to generate a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    noise : float
        The standard deviation of the Gaussian noise to add to the target.
    random_state : int or None
        The seed to use for the random number generator.
    """

    def __init__(self, n_samples_per_task=25, noise=0.1, n_tasks=3,
                  random_state=42, start=-2., stop=2.):
        super().__init__(n_samples_per_task=n_samples_per_task, 
                         noise=noise, n_tasks=n_tasks, 
                         random_state=random_state, start=start, stop=stop)
        
    def get_colors(self):
        colors = {t: cm.tab10(int(t/10)) for t in self.tasks}
        ic(colors)
        # print(colors)
        return colors
    
    def _get_color(self, t):
        color = cm.tab10(int(t/10))
        return color
    
    def _get_marker(self, t):
        markers = ['.', 'o', 'v', '*', '^']
        return markers[int(t%10)]
        
    def _generate_tasks(self):
        tasks = []
        for t in range(self.n_tasks):
            m_t = 1
            tasks.extend([t*10+j for j in range(m_t)])
        ic(tasks)
        return tasks
        
    def _function_task(self, x, t=None):
        real_t = int(t/10)
        y = self.fun_tasks[real_t](x)
        # if t is not None:
        #     y += (1e-6 * self.fun_tasks[t](x))
        return y

    def _generate_task_functions(self):
        self.fun_tasks = {
            0: np.vectorize(lambda x: 10 * np.sin(x)),
            1: np.vectorize(lambda x: x**2),
            2: np.vectorize(lambda x: x**3)
        }

        task_functions = {}
        for t in self.tasks:
            task_functions[t] = (lambda x: self._function_task(x, t=t))

        return task_functions

# class MTLClustersARegression(MTLSyntheticRegressionDataset):
#     """
#     A class to generate a synthetic regression dataset.

#     Parameters
#     ----------
#     n_samples : int
#         The number of samples to generate.
#     noise : float
#         The standard deviation of the Gaussian noise to add to the target.
#     random_state : int or None
#         The seed to use for the random number generator.
#     """

#     def __init__(self, n_samples_per_task=25, noise=0.1, n_tasks=3,
#                   random_state=42, start=-2., stop=2.):
#         super().__init__(n_samples_per_task=n_samples_per_task, 
#                          noise=noise, n_tasks=n_tasks, 
#                          random_state=random_state, start=start, stop=stop)
        
#     def get_colors(self):
#         colors = {t: cm.tab10(int(t.split('_')[0])) for t in self.tasks}
#         ic(colors)
#         # print(colors)
#         return colors
    
#     def _get_color(self, t):
#         return self.colors[t]
    
#     def _get_marker(self, t):
#         markers = ['.', 'o', 'v', '*', '^']
#         return markers[int(t.split('_')[1])]
        
#     def _generate_tasks(self):
#         tasks = []
#         for t in range(self.n_tasks):
#             m_t = randint(1,5)
#             tasks.extend(['{}_{}'.format(t, j) for j in range(m_t)])
#         ic(tasks)
#         return tasks
        
#     def _function_task(self, x, t=None):
#         real_t = int(t.split('_')[0])
#         y = self.fun_tasks[real_t](x)
#         # if t is not None:
#         #     y += (1e-6 * self.fun_tasks[t](x))
#         return y

#     def _generate_task_functions(self):
#         self.fun_tasks = {
#             0: np.vectorize(lambda x: 10 * np.sin(x)),
#             1: np.vectorize(lambda x: x**2),
#             2: np.vectorize(lambda x: x**3)
#         }

#         task_functions = {}
#         for t in self.tasks:
#             task_functions[t] = (lambda x: self._function_task(x, t=t))

#         return task_functions
