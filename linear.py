# Externals
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def predict_lr(country, X, y, X_pred, x_days):
    # Step: Predict
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    y_pred = regr.predict(X_pred)
    y_approximate = regr.predict(X)
    coef = regr.coef_

    # Step: Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(X, y, color='black')
    ax.plot(X, y_approximate, color='green', linewidth=3)

    ax.scatter(X_pred, y_pred, color='red')
    ax.plot(X_pred, y_pred, color='blue', linewidth=3)

    # Step: Notes
    nums = "Predict: " + ", ".join([str(round(x, 1)) for x in y_pred])
    xlabels = [x.strftime('%m.%d') for x in x_days]
    end = (-1) * len(X_pred)
    dates = ", ".join(xlabels[end:])
    title = f'{country}, {nums}\n@ {dates}\nK = {coef}'
    ax.set_title(title)
    plt.xticks(ticks=range(1, len(x_days) + 1), labels=xlabels)

    # Step: R2 score
    r2 = r2_score(y, y_approximate)
    rmse = mean_squared_error(y, y_approximate, squared=False)
    print(f'R2 score = {r2}')
    print(f'RMSE = {rmse}')

    # Step: Show
    fig.tight_layout()
    plt.show()
