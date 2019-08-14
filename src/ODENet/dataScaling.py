import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class data_scaler(object):
    def __init__(self):
        self.norm = None
        self.norm_1 = None
        self.std = None
        self.case = None
        self.scale = 1
        self.bias = 1e-20
        #         self.bias = 1

        self.switcher = {
            "min_std": "min_std",
            "std2": "std2",
            "std_min": "std_min",
            "min": "min",
            "no": "no",
            "log": "log",
            "log_min": "log_min",
            "log_std": "log_std",
            "log2": "log2",
            "sqrt_std": "sqrt_std",
            "cbrt_std": "cbrt_std",
            "nrt_std": "nrt_std",
            "cb_std": "cb_std",
            "tan": "tan",
        }

    def fit_transform(self, input_data, case):
        self.case = case
        if self.switcher.get(self.case) == "min_std":
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.norm.fit_transform(input_data)
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "std2":
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)

        if self.switcher.get(self.case) == "std_min":
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == "min":
            self.norm = MinMaxScaler()
            out = self.norm.fit_transform(input_data)

        if self.switcher.get(self.case) == "no":
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = input_data

        if self.switcher.get(self.case) == "log_min":
            out = -np.log(np.asarray(input_data / self.scale) + self.bias)
            self.norm = MinMaxScaler()
            out = self.norm.fit_transform(out)

        if self.switcher.get(self.case) == "log_std":
            out = -np.log(np.asarray(input_data / self.scale) + self.bias)
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "log2":
            self.norm = MinMaxScaler()
            self.std = StandardScaler()
            out = self.norm.fit_transform(input_data)
            out = np.log(np.asarray(out) + self.bias)
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "sqrt_std":
            out = np.sqrt(np.asarray(input_data / self.scale))
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "cbrt_std":
            out = np.cbrt(np.asarray(input_data / self.scale))
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "nrt_std":
            out = np.power(np.asarray(input_data / self.scale), 1 / 4)
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "cb_std":
            out = np.power(np.asarray(input_data / self.scale), 3)
            self.std = StandardScaler()
            out = self.std.fit_transform(out)

        if self.switcher.get(self.case) == "tan":
            self.norm = MaxAbsScaler()
            self.std = StandardScaler()
            out = self.std.fit_transform(input_data)
            out = self.norm.fit_transform(out)
            out = np.tan(out / (2 * np.pi + self.bias))

        return out

    def transform(self, input_data):
        if self.switcher.get(self.case) == "min_std":
            out = self.norm.transform(input_data)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "std2":
            out = self.std.transform(input_data)

        if self.switcher.get(self.case) == "std_min":
            out = self.std.transform(input_data)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == "min":
            out = self.norm.transform(input_data)

        if self.switcher.get(self.case) == "no":
            out = input_data

        if self.switcher.get(self.case) == "log_min":
            out = -np.log(np.asarray(input_data / self.scale) + self.bias)
            out = self.norm.transform(out)

        if self.switcher.get(self.case) == "log_std":
            out = -np.log(np.asarray(input_data / self.scale) + self.bias)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "log2":
            out = self.norm.transform(input_data)
            out = np.log(np.asarray(out) + self.bias)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "sqrt_std":
            out = np.sqrt(np.asarray(input_data / self.scale))
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "cbrt_std":
            out = np.cbrt(np.asarray(input_data / self.scale))
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "nrt_std":
            out = np.power(np.asarray(input_data / self.scale), 1 / 4)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "cb_std":
            out = np.power(np.asarray(input_data / self.scale), 3)
            out = self.std.transform(out)

        if self.switcher.get(self.case) == "tan":
            out = self.std.transform(input_data)
            out = self.norm.transform(out)
            out = np.tan(out / (2 * np.pi + self.bias))

        return out

    def inverse_transform(self, input_data):

        if self.switcher.get(self.case) == "min_std":
            out = self.std.inverse_transform(input_data)
            out = self.norm.inverse_transform(out)

        if self.switcher.get(self.case) == "std2":
            out = self.std.inverse_transform(input_data)

        if self.switcher.get(self.case) == "std_min":
            out = self.norm.inverse_transform(input_data)
            out = self.std.inverse_transform(out)

        if self.switcher.get(self.case) == "min":
            out = self.norm.inverse_transform(input_data)

        if self.switcher.get(self.case) == "no":
            out = input_data

        if self.switcher.get(self.case) == "log_min":
            out = self.norm.inverse_transform(input_data)
            out = (np.exp(-out) - self.bias) * self.scale

        if self.switcher.get(self.case) == "log_std":
            out = self.std.inverse_transform(input_data)
            out = (np.exp(-out) - self.bias) * self.scale

        if self.switcher.get(self.case) == "log2":
            out = self.std.inverse_transform(input_data)
            out = np.exp(out) - self.bias
            out = self.norm.inverse_transform(out)

        if self.switcher.get(self.case) == "sqrt_std":
            out = self.std.inverse_transform(input_data)
            out = np.power(out, 2) * self.scale

        if self.switcher.get(self.case) == "cbrt_std":
            out = self.std.inverse_transform(input_data)
            out = np.power(out, 3) * self.scale

        if self.switcher.get(self.case) == "nrt_std":
            out = self.std.inverse_transform(input_data)
            out = np.power(out, 4) * self.scale

        if self.switcher.get(self.case) == "cb_std":
            out = self.std.inverse_transform(input_data)
            out = np.power(out, 1 / 3) * self.scale

        if self.switcher.get(self.case) == "tan":
            out = (2 * np.pi + self.bias) * np.arctan(input_data)
            out = self.norm.inverse_transform(out)
            out = self.std.inverse_transform(out)

        return out


class LogScaler(object):
    def fit_transform(self, input_data):
        out = np.log(input_data)
        return out

    def transform(self, input_data):
        out = np.log(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.exp(input_data)
        return out


class LogMirrorScaler(object):
    def fit_transform(self, input_data):
        out = np.log(input_data)
        return out

    def transform(self, input_data):
        out = np.log(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.exp(input_data)
        return out


class AtanScaler(object):
    def fit_transform(self, input_data):
        out = np.arctan(input_data)
        return out

    def transform(self, input_data):
        out = np.arctan(input_data)
        return out

    def inverse_transform(self, input_data):
        out = np.tan(input_data)
        return out


class NoScaler(object):
    def fit(self, input_data):
        out = input_data
        return out

    def fit_transform(self, input_data):
        out = input_data
        return out

    def transform(self, input_data):
        out = input_data
        return out

    def inverse_transform(self, input_data):
        out = input_data
        return out
