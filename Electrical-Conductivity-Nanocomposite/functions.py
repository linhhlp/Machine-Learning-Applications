"""Some basic functions for model of electrical conductivity."""


from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler  # , MinMaxScaler

__author__ = "Linh Hoang (linhhlp)"
__copyright__ = "@2022 Project: Prediction of Electrical Conductivity"
__license__ = "MIT"
__version__ = "1.0.2"


def safe_log10(data):
    """Handle LOG safely in cases of too small values close or equal to 0.

    Parameters
    ----------
    data : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    data: np.Array[float]
        {array-like, sparse matrix} of shape (n_samples, n_features)
    """
    prob_tmp = np.where(data > 1.0e-10, data, 1.0e-10)
    result = np.where(data > 1.0e-10, np.log10(prob_tmp), -10)
    return result


class CustomizedScaler(ABC):
    """Base class for scalers."""

    scaler: StandardScaler

    @abstractmethod
    def fit_transform(self, data):
        """Fit then transform data."""

    @abstractmethod
    def transform(self, data):
        """Transform data."""

    @abstractmethod
    def inverse_transform(self, data):
        """Inverse transform data to get real value (before transformed)."""


class SuperHighVariationScaler(CustomizedScaler):
    """Create a wrapper for different scalers."""

    def __init__(self, scaler=None):
        """Initialize if scaler is given.

        Parameters
        ----------
        scaler : Scaler, optional
            Include StandardScaler, MinMaxScaler
            from sklearn.preprocessing, by default StandardScaler
        """
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

    def fit_transform(self, data):
        """Fit then transform data.

        Parameters
        ----------
        data : Array[float]
            Input data.
            {array-like, sparse matrix} of shape (n_samples, n_features)

        Returns
        -------
        data: Array[float]
            Transformed data.
            {ndarray, sparse matrix} of shape (n_samples, n_features)
        """
        data_loged = safe_log10(data)
        return self.scaler.fit_transform(data_loged)

    def transform(self, data):
        """Transform data."""
        data_loged = safe_log10(data)
        return self.scaler.transform(data_loged)

    def inverse_transform(self, data):
        """Inverse transform data to get real value (before transformed)."""
        data_unloged = self.scaler.inverse_transform(data)
        return np.float_power(10, data_unloged)

    def __str__(self):
        return f"SuperHighVariationScaler: {self.scaler}"


class SuperHighVariationScalerSimple(CustomizedScaler):
    """Simple Scaler with only LOG10."""

    def fit_transform(self, data):
        """Fit then transform data."""
        return safe_log10(data)

    def transform(self, data):
        """Transform data."""
        return safe_log10(data)

    def inverse_transform(self, data):
        """Inverse transform data to get real value (before transformed)."""
        return np.float_power(10, data)

    def __str__(self):
        return "SuperHighVariationScalerSimple"


class NoScaler(CustomizedScaler):
    """Just return the same data (without modifying)."""

    def fit_transform(self, data):
        """Fit then transform data."""
        return data

    def transform(self, data):
        """Transform data."""
        return data

    def inverse_transform(self, data):
        """Inverse transform data to get real value (before transformed)."""
        return data

    def __str__(self):
        return "NoScaler"


def map_string_to_num(data):
    """Map string to number: Change text ID to index ID of material.

    Parameters
    ----------
    data : _type_
        Input data.

    Returns
    -------
    _type_
        Change fields of data which are string to number.
    """
    polymer_map_list = {"HDPE": 0, "HDPEtreated": 1}
    filler_map_list = {"MWCNT": 0, "SWCNT": 1, "GNP": 2}
    data["polymer_1"] = data["polymer_1"].map(polymer_map_list)
    data["filler_1"] = data["filler_1"].map(filler_map_list)
    return data


def map_num_to_string(data):
    """Map number to string: Change index to text ID of material.

    Parameters
    ----------
    data : _type_
        Input data.

    Returns
    -------
    _type_
        Change fields of data which are number to string.
    """
    # inversed_map = {v: k for k, v in polymer_map_list.items()}
    polymer_map_list = {0: "HDPE", 1: "HDPEtreated"}
    filler_map_list = {0: "MWCNT", 1: "SWCNT", 2: "GNP"}
    data["polymer_1"] = data["polymer_1"].map(polymer_map_list)
    data["filler_1"] = data["filler_1"].map(filler_map_list)
    return data


class PlotRealTime(tf.keras.callbacks.Callback):
    """Plot Loss and Accuracy while training (graph real time fitting).

    References
    ----------
        https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
        https://www.tensorflow.org/guide/keras/custom_callback
    """

    def __init__(self, step=100):
        """Initialize.

        Parameters
        ----------
        step : int, optional
            interval of plotting based on epoch, by default 100
        """
        self.step = step
        self.epoch = 0
        self.num = 0
        self.metrics = {}

    def save_data(self, logs) -> None:
        """Save data to plot later."""
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

    def plot_data(self, logs) -> None:
        """Plot data."""
        metrics = [x for x in logs if "val" not in x]
        _, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        if self.epoch % self.step == 0:
            x_range = [self.step * x for x in range(0, self.num)]
        else:
            x_range = [self.step * x for x in range(0, self.num - 1)]
            x_range.append(self.epoch)

        for i, metric in enumerate(metrics):
            axs[i].plot(
                x_range,
                self.metrics[metric],
                label=metric,
                marker="+",
            )
            if logs["val_" + metric]:
                axs[i].plot(
                    x_range,
                    self.metrics["val_" + metric],
                    label="val_" + metric,
                    marker="o",
                )

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

    def on_train_begin(self, logs=None) -> None:
        """Initialize."""
        if logs:
            self.metrics = {metric: [] for metric in logs}
        else:
            self.metrics = {}

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Store metrics."""
        step = self.step
        self.epoch += 1
        if epoch % step != 0:
            return None
        self.num = epoch // step + 1
        self.save_data(logs)
        self.plot_data(logs)

    def on_train_end(self, logs=None) -> None:
        """Force to replot including last point of data."""
        self.save_data(logs)
        self.num += 1
        self.plot_data(logs)


def early_stopper(patience=100, monitor="val_loss", verbose=1):
    """Create a early stopper for training.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training will
        be stopped, by default 100.

    monitor:
        Quantity to be monitored.

    Returns
    -------
    tf.keras.callbacks.EarlyStopping
        Following keras.callbacks.EarlyStopping.

    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=verbose
    )


def reduce_lr(patience=100, min_lr=0.000001):
    """Reduce learning rate when no improvement.

    These parameters follow keras.callbacks.ReduceLROnPlateau.

    Parameters
    ----------
    patience : int, optional
        _description_, by default 100
    min_lr : float, optional
        _description_, by default 0.000001

    Returns
    -------
    tf.keras.callbacks.ReduceLROnPlateau
        Following keras.callbacks.ReduceLROnPlateau.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=patience, min_lr=min_lr
    )


def model_checkpoint_callback(checkpoint_filepath="tmp/checkpoint"):
    """Reduce learning rate when no improvement.

    These parameters follow tf.keras.callbacks.ModelCheckpoint.

    Parameters
    ----------
    checkpoint_filepath : str, optional
        by default at "tmp/checkpoint"

    Returns
    -------
    tf.keras.callbacks.ModelCheckpoint
        Following tf.keras.callbacks.ModelCheckpoint.
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )


def smooth_array(data, size: int = 10):
    """Smooth data in 1-D array.

    Data acts like signal with noise, fluctuates
    References: https://www.delftstack.com/howto/python/smooth-data-in-python/
    """
    new_arr = np.convolve(data, np.ones(size) / size, mode="same")
    new_arr[0:5] = np.average(data[0:5])
    new_arr[-5:] = np.average(data[-5])
    return new_arr


def sparse_array(data_frame, frac=0.95):
    """Reduce data rows to `frac` ratio (sparse data).

    When plotting too many points of data, cut down randomly.

    Parameters
    ----------
    dataFrame : Pandas.dataFrame
        Input data.
    frac : float, optional
        Fraction cut down. For example,  frac=0.95 remove 95%,
        only keep 5% leftover

    Notes:
    ------
        Equivalent Function df = df.sample(frac= (1-frac) ).
    """
    drop_indices = np.random.choice(
        data_frame.index,
        int(np.ceil(len(data_frame.index) * frac)),
        replace=False,
    )
    return data_frame.drop(drop_indices)


def generate_data_linear(sigma_not, pc, t, noise, n=1000, pmax=25):
    """Generate the data by using an linear equation with noise.

    Use linear data for testing purpose which is easier for linear regression.

    Parameters
    ----------
    sigma_not : float
        Fitting coefficients
    pc : float
        Fitting coefficients
    t : float
        Fitting coefficients
    noise : float
        Maximum in percentage flucation from real value
    n : int, optional
        Number of samples (number of data generated), by default 1000
    pmax : int, optional
        Maximum in percent, by default 25

    Returns
    -------
    p : ndarray
        Weight fraction
    sigma : ndarray
        The original electrical conductivity of the nanocomposite
    sigma_not : ndarray
        The electrical conductivity of the nanocomposite with noise
    """
    p = np.random.uniform(float(pc), float(pmax), n)
    p[0] = 1.00001 * p[0] * t  # force (p -pc) to not be 0
    sigma = sigma_not * (np.random.randint(1, 5) * p + pc)
    # add noise
    sigma_noise = sigma * (1 + (2 * np.random.random(n) - 1) * (noise / 100.0))

    return (p, sigma, sigma_noise)
