#!/usr/bin/python3
# -*- coding: utf-8 -*

# ML Imports
import time
import math
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# UI Imports
import sys, os
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QGraphicsDropShadowEffect, QErrorMessage, \
    QMessageBox
from PyQt5.QtGui import QDoubleValidator, QColor, QIcon


def resource_path(relative_path):
    try:
    
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



class MaxScaler(BaseEstimator, ClassifierMixin):

    def __init__(self, name="MaxScaler"):
        self.name = name

    def fit(self, X):
        self.max_elements = np.amax(X, axis=0)
        return self

    def transform(self, X):
        scaledX = X / self.max_elements
        return scaledX


class GeneralRegressionNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, name="GRNN", sigma=0.1):
        self.name = name
        self.sigma = sigma

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        # 2 multiplier?
        self.pow_sigma = np.power(self.sigma, 2)

    def predict_single(self, x):
        # start_time = time.time()
        gaussian_distances = np.exp(-np.power(np.sqrt((np.square(self.train_X - x).sum(axis=1))), 2) \
                                    / (2 * self.pow_sigma))
        gaussian_distances_sum = gaussian_distances.sum()
        if gaussian_distances_sum < math.pow(10, -7): gaussian_distances_sum = math.pow(10, -7)
        result = np.dot(gaussian_distances, self.train_y) / gaussian_distances_sum
        # print("--- %s seconds ---" % (time.time() - start_time))
        return result

    def predict(self, X):
        predictions = np.apply_along_axis(self.predict_single, axis=1, arr=X)
        return predictions


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def display_vector(vector):
    vector = [str(x) for x in vector]
    result = "("
    result += ",  ".join(vector)
    result += ")"
    return result


# UI part
class App(QMainWindow):

    # Ініціалізація GUI
    def __init__(self):
        super().__init__()
        self.data = {}
        self.params = ["sigma_1", "sigma_2", "df_train", "df_test"]
        self.init_ui()

    # Ініціалізація компонентів GUI
    def init_ui(self):
        self.ui = uic.loadUi("ml.ui")

        self.setWindowIcon(QIcon('favicon.ico'))

        self.run_shadow = QGraphicsDropShadowEffect()
        self.run_shadow.setBlurRadius(35)
        self.run_shadow.setYOffset(0)
        self.run_shadow.setXOffset(0)
        self.run_shadow.setColor(QColor(46, 193, 172))

        self.run_shadow_1 = QGraphicsDropShadowEffect()
        self.run_shadow_1.setBlurRadius(35)
        self.run_shadow_1.setYOffset(0)
        self.run_shadow_1.setXOffset(0)
        self.run_shadow_1.setColor(QColor(46, 193, 172))

        self.run_shadow_2 = QGraphicsDropShadowEffect()
        self.run_shadow_2.setBlurRadius(35)
        self.run_shadow_2.setYOffset(0)
        self.run_shadow_2.setXOffset(0)
        self.run_shadow_2.setColor(QColor(46, 193, 172))

        self.upload_shadow = QGraphicsDropShadowEffect()
        self.upload_shadow.setBlurRadius(35)
        self.upload_shadow.setYOffset(0)
        self.upload_shadow.setXOffset(0)
        self.upload_shadow.setColor(QColor(40, 171, 185))

        self.upload_shadow_2 = QGraphicsDropShadowEffect()
        self.upload_shadow_2.setBlurRadius(35)
        self.upload_shadow_2.setYOffset(0)
        self.upload_shadow_2.setXOffset(0)
        self.upload_shadow_2.setColor(QColor(40, 171, 185))

        self.ui.s1_run_button.setGraphicsEffect(self.run_shadow)
        self.ui.s2_run_button.setGraphicsEffect(self.run_shadow_1)
        self.ui.s4_run_button.setGraphicsEffect(self.run_shadow_2)

        self.ui.s1_upload_button.setGraphicsEffect(self.upload_shadow)
        self.ui.s2_upload_button.setGraphicsEffect(self.upload_shadow_2)

        self.ui.s1_upload_button.clicked.connect(self.upload_train_data)
        self.ui.s2_upload_button.clicked.connect(self.upload_test_data)
        self.ui.s1_run_button.clicked.connect(self.get_sigmas)
        self.ui.s2_run_button.clicked.connect(self.get_parameters)
        self.ui.s4_run_button.clicked.connect(self.make_prediction_for_vector)

        self.onlyInt = QDoubleValidator()
        self.ui.s1_sigma_1_input.setValidator(self.onlyInt)
        self.ui.s1_sigma_2_input.setValidator(self.onlyInt)
        self.ui.sno2_input.setValidator(self.onlyInt)
        self.ui.C6H6_input.setValidator(self.onlyInt)
        self.ui.Ti_input.setValidator(self.onlyInt)
        self.ui.WO_input.setValidator(self.onlyInt)
        self.ui.WO2_input.setValidator(self.onlyInt)
        self.ui.InO_input.setValidator(self.onlyInt)
        self.ui.T_input.setValidator(self.onlyInt)
        self.ui.RH_input.setValidator(self.onlyInt)
        self.ui.AH_input.setValidator(self.onlyInt)
        self.ui.NO_input.setValidator(self.onlyInt)
        self.ui.NO2_input.setValidator(self.onlyInt)

        self.ui.show()

    # Завантаження підготовчої вибірки
    def upload_train_data(self):
        train_data_file_name = QFileDialog.getOpenFileName(self, 'Завантажити підготовчу вибірку')[0]
        if train_data_file_name:
            try:
                df_train = pd.read_csv(train_data_file_name, header=None)
                self.data["df_train"] = df_train
                self.ui.s1_upload_status.setText(train_data_file_name.split("/")[-1])
                self.ui.s1_upload_status.setStyleSheet("color: rgb(230, 254, 240);")
            except:
                self.ui.s1_upload_status.setText("Помилка завантаження")
                self.ui.s1_upload_status.setStyleSheet("color: #F31431;")
        else:
            self.ui.s1_upload_status.setText("Підготовча вибірка не завантажена!")
            self.ui.s1_upload_status.setStyleSheet("color: #F31431;")

    # Завантаження тестової вибірки
    def upload_test_data(self):
        test_data_file_name = QFileDialog.getOpenFileName(self, 'Завантажити тестову вибірку')[0]
        if test_data_file_name:
            try:
                df_test = pd.read_csv(test_data_file_name, header=None)
                self.data["df_test"] = df_test
                self.ui.s2_upload_status.setText(test_data_file_name.split("/")[-1])
                self.ui.s2_upload_status.setStyleSheet("color: rgb(230, 254, 240);")
            except:
                self.ui.s2_upload_status.setText("Помилка завантаження")
                self.ui.s2_upload_status.setStyleSheet("color: #F31431;")
        else:
            self.ui.s2_upload_status.setText("Тестова вибірка не завантажена!")
            self.ui.s2_upload_status.setStyleSheet("color: #F31431;")

    # Завантаження сигма- параметрів
    def get_sigmas(self):

        sigma_1 = self.str_to_float(self.ui.s1_sigma_1_input.text())
        sigma_2 = self.str_to_float(self.ui.s1_sigma_2_input.text())

        if not sigma_1 or not sigma_2:
            self.ui.sigma_label.setText("Введіть параметри")
            self.ui.sigma_label.setStyleSheet("color: #F31431;")
        else:
            self.ui.sigma_label.setText("σ₁ = {0},  σ₂ = {1}".format(round(sigma_1, 3), round(sigma_2, 3)))
            self.ui.sigma_label.setStyleSheet("color: rgb(230, 254, 240);")

        self.data["sigma_1"] = sigma_1
        self.data["sigma_2"] = sigma_2

        if "df_train" not in self.data:
            self.ui.s1_upload_status.setText("Завантажте вибірку")
            self.ui.s1_upload_status.setStyleSheet("color: #F31431;")

    # !!!
    def get_parameters(self):
        if "df_test" not in self.data:
            self.ui.s2_upload_status.setText("Завантажте вибірку")
            self.ui.s2_upload_status.setStyleSheet("color: #F31431;")

        errors = []

        for element in self.params:
            if element in self.data:
                pass
            else:
                errors.append(element)

        if errors:
            message = "Недопустимі значення: " + ", ".join(errors)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Помилка")
            msg.setInformativeText(message)
            msg.setWindowTitle("Набір даних не коректний")
            msg.exec_()
        else:
            self.calc_prediction()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Готово!")
            msg.setInformativeText("Тестування ансамблю завершено")
            msg.setWindowTitle("Звіт режиму тестування")
            msg.exec_()





    # Обробка вектора і побудова передбачення для нього
    def make_prediction_for_vector(self):

        sno2 = self.str_to_float(self.ui.sno2_input.text())
        c6h6 = self.str_to_float(self.ui.C6H6_input.text())
        ti = self.str_to_float(self.ui.Ti_input.text())
        wo = self.str_to_float(self.ui.WO_input.text())
        wo2 = self.str_to_float(self.ui.WO2_input.text())
        ino = self.str_to_float(self.ui.InO_input.text())
        t = self.str_to_float(self.ui.T_input.text())
        rh = self.str_to_float(self.ui.RH_input.text())
        ah = self.str_to_float(self.ui.AH_input.text())
        no = self.str_to_float(self.ui.NO_input.text())
        no2 = self.str_to_float(self.ui.NO2_input.text())

        vector = [sno2, c6h6, ti, wo, wo2, ino, t, rh, ah, no, no2]

        self.data['sno2'] = sno2
        self.data['c6h6'] = c6h6
        self.data['ti'] = ti
        self.data['wo'] = wo
        self.data['wo2'] = wo2
        self.data['ino'] = ino
        self.data['t'] = t
        self.data['rh'] = rh
        self.data['ah'] = ah
        self.data['no'] = no
        self.data['no2'] = no2

        if None in vector:
            self.ui.vector_label.setText("Змінні введено некоректно")
            self.ui.vector_label.setStyleSheet("color: #F31431;")
        else:
            self.ui.vector_label.setText(display_vector(vector))
            self.ui.vector_label.setStyleSheet("color: rgb(230, 254, 240);")
            self.data["vector"] = vector

            if "grnn1" in self.data and "grnn2" in self.data:
                y = self.calc_prediction_for_vector()
                result = "CO — {0}".format(round(y, 2))
                self.ui.y_label.setText(result)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Помилка")
                msg.setInformativeText("Не проведено тестування ансамблю")
                msg.setWindowTitle("Помилка")
                msg.exec_()

    def calc_prediction(self):
        df_train = self.data["df_train"]
        df_test = self.data["df_test"]
        sigma_1 = self.data["sigma_1"]
        sigma_2 = self.data["sigma_2"]

        train_X = df_train.iloc[:, :-1].reset_index(drop=True)
        train_y = df_train.iloc[:, -1].reset_index(drop=True)
        test_X = df_test.iloc[:, : -1].reset_index(drop=True)
        test_y = df_test.iloc[:, -1].reset_index(drop=True)

        scaler = MaxScaler()
        scaler.fit(train_X)

        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        start_time_g1 = time.time()
        grnn1 = GeneralRegressionNeuralNetwork(sigma=sigma_1)
        deltas = []
        for i in range(train_X.shape[0]):
            train_X_except_i = train_X.drop([i])
            train_y_except_i = train_y.drop([i])

            grnn1.fit(train_X_except_i, train_y_except_i)

            deltas.extend(train_y[i] - grnn1.predict((pd.DataFrame(train_X.iloc[[i]]))))

        time_g1 = time.time() - start_time_g1
        pred_y = grnn1.predict(test_X)

        mape = mean_absolute_percentage_error(test_y, pred_y)
        rmse = root_mean_squared_error(test_y, pred_y)

        self.data["mape"] = mape
        self.data["rmse"] = rmse

        self.ui.s3_mape_value_label.setText(str(round(mape, 2)) + "%")
        self.ui.s3_rmse_value_label.setText(str(round(rmse, 2)) + "%")

        start_time_g2 = time.time()
        grnn2 = GeneralRegressionNeuralNetwork(sigma=sigma_2)
        grnn2.fit(train_X, pd.Series(deltas))
        pred_deltas = grnn2.predict(test_X)
        time_g2 = time.time() - start_time_g2 + time_g1
        final_pred = pred_y + pred_deltas

        self.data["grnn2"] = grnn1
        self.data["grnn2"] = grnn2
        self.ui.s3_time_value_label.setText(str(round(time_g2, 2)) + " сек.")
        np.savetxt("twoGRNN.csv", final_pred, delimiter=",", fmt="%1.5f")

    def calc_prediction_for_vector(self):
        vector = self.data["vector"]
        grnn2 = self.data["grnn2"]

        return grnn2.predict_single(vector)

    def display_data(self):
        for key in self.data:
            self.data_table.add_row([key, str(self.data[key]), type(self.data[key])])

        print("Отримано такі вхідні дані:")
        print(self.data_table)

    @staticmethod
    def str_to_float(num):
        if len(num) != 0:
            return float(str(num.replace(",", ".")))
        else:
            return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
