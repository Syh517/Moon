import shap
import math
from matplotlib import pyplot as plt
from IPython.display import (display, display_html, display_png, display_svg)
import numpy as np
shap.initjs()


def SelectAbnormal(X_train,y_train):
    X=[]
    for i in range(len(y_train)):
        if y_train[i]==1:
            X.append(X_train[i])
    X=np.array(X)
    return X
def Kexplainer(model,X_train, y_train,X_test, i):
    X=SelectAbnormal(X_train, y_train)
    feature_names = ['Feature 0', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5',
                     'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11',
                     'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15', 'Feature 16',
                     'Feature 17', 'Feature 18', 'Feature 19', 'Feature 20', 'Feature 21',
                     'Feature 22', 'Feature 23', 'Feature 24', 'Feature 25', 'Feature 26',
                     'Feature 27', 'Feature 28', 'Feature 29', 'Feature 30', 'Feature 31',
                     'Feature 32', 'Feature 33', 'Feature 34', 'Feature 35', 'Feature 36', 'Feature 37']

    explainer = shap.KernelExplainer(model.predict, X, link="logit",feature_names=feature_names)
    shap_values = explainer.shap_values(X_test[i:i + 1, ])
    print(explainer.expected_value)

    # shap.initjs()  # 用来显示的
    # shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=feature_names,features=X_test[i])
    # plt.show()

    # shap.plots.force(explainer.expected_value[0], shap_values[0], X_test[i], link="logit", show=False)
    # plt.show()

    # shap.summary_plot(shap_values, X_test[i].reshape(1, X_test[0].shape[0]), max_display=38,
    #                   feature_names=feature_names, plot_type="bar")


