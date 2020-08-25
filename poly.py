# Externals
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def predict_poly(country, X, y, X_pred, x_days, degree: int):
    # Step: Create true polynom
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, y)

    lin2 = linear_model.LinearRegression()
    lin2.fit(X_poly, y)
    y_approximate = lin2.predict(X_poly)
    new_X, new_y = zip(*sorted(zip(X, y_approximate)))
    coef = lin2.coef_

    # Step: Predict
    poly_pred = PolynomialFeatures(degree=degree)
    X_pred_poly = poly_pred.fit_transform(X_pred)
    y_pred = lin2.predict(X_pred_poly)
    # print(X_pred_poly)
    # print(y_pred)
    new_X_pred, new_y_pred = zip(*sorted(zip(X_pred, y_pred)))

    # Step: Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(X, y, color='black')
    ax.plot(new_X, new_y, color='green', linewidth=3)

    ax.scatter(new_X_pred, new_y_pred, color='red')
    ax.plot(new_X_pred, new_y_pred, color='blue', linewidth=3)

    # Step: Metrics
    r2 = r2_score(y, new_y)
    rmse = mean_squared_error(y, new_y, squared=False)

    # Step: Notes
    nums = "Predict: " + ", ".join([str(round(x, 1)) for x in y_pred])
    xlabels = [x.strftime('%m.%d') for x in x_days]
    end = (-1) * len(X_pred)
    dates = ", ".join(xlabels[end:])
    title = f'{country}, {nums}\n@ {dates}\nR2 = {r2}\nRMSE = {rmse}'
    ax.set_title(title)
    plt.xticks(ticks=range(1, len(x_days) + 1), labels=xlabels)

    # Step: Show
    fig.tight_layout()
    plt.show()
