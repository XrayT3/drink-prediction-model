import sys
import requests
import pandas as pd
import seaborn as sns
from termcolor import colored
import pycountry_convert as pc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def download_dataset(url, fname):
    response = requests.get(url)
    if response.status_code == 200:
        with open(fname, "wb") as f:
            f.write(response.content)
        print("File has been downloaded\n")
    else:
        print("File has not been downloaded", file=sys.stderr)
        print("Status code:", response.status_code, file=sys.stderr)


if __name__ == '__main__':
    color_output = "green"
    path = "https://raw.githubusercontent.com/fivethirtyeight/data/master/alcohol-consumption/drinks.csv"
    filename = "drinks.csv"
    download_dataset(path, filename)
    df = pd.read_csv(filename)

    print(colored("Display the data types of each column for understanding what we have", color_output))
    print(df.dtypes)

    # clean data
    # convert name of countries to wiki format
    df["country"] = df["country"].replace("&", "and", regex=True)
    df["country"] = df["country"].replace("-", " and ", regex=True)

    continent_codes = []
    continents = []
    # find for each country its continent
    for country in df["country"].values.tolist():
        try:
            country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
            continent_code = pc.country_alpha2_to_continent_code(country_code)
            continent_codes.append(continent_code)
            continents.append(pc.convert_continent_code_to_continent_name(continent_code))
        except KeyError:
            df = df[df["country"] != country]  # remove countries with incorrect names
    # add missing columns
    df["continent_code"] = continent_codes
    df["continent"] = continents
    df.to_csv("drinks_clean.csv")
    print(colored("\nAdd missing columns: continent and its code", color_output))
    print(df.head())

    print(colored("\nGet the number of wine servings per continent:", color_output))
    print(df.groupby(["continent"]).sum()["wine_servings"])
    # now we know that Europeans like wine much more than others
    # and beer as well (:
    # btw we can look at this
    sns.boxplot(x="continent", y="beer_servings", data=df)
    plt.show()

    # let's look if there is correlation between wine and beer
    sns.regplot(x="wine_servings", y="beer_servings", data=df)
    plt.show()
    plt.close()
    # doesn't look like

    # now we will try to predict total litres of alcohol
    print(colored("\nCorrelation between litres of alcohol and each drink", color_output))
    print(df.corr()["total_litres_of_pure_alcohol"])

    # prepare data for models
    data_x = df._get_numeric_data().drop(columns="total_litres_of_pure_alcohol")
    data_y = df[["total_litres_of_pure_alcohol"]]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.10, random_state=0)

    # the first model uses only the number of "wine_servings"
    model = LinearRegression()
    model.fit(x_train[["wine_servings"]], y_train)
    print("\nThe R-square of the first model:", model.score(x_test[["wine_servings"]], y_test))

    # the second model uses list of features
    model2 = LinearRegression()
    model2.fit(x_train, y_train)
    print("The R-square of the second model:", model2.score(x_test, y_test))

    # the third model uses polynomial list of features
    pipeline_input = [("scale", StandardScaler()),
                      ("polynomial", PolynomialFeatures(degree=4, include_bias=False)),
                      ("model", LinearRegression())]
    model3 = Pipeline(pipeline_input)
    model3.fit(x_train, y_train)
    print("The R-square of the third model:", model3.score(x_test, y_test))

    # the last model uses polynomial list of features and alpha
    pipeline_input = [("scale", StandardScaler()),
                      ("polynomial", PolynomialFeatures(degree=4, include_bias=False)),
                      ("model", Ridge(alpha=0.1))]
    model4 = Pipeline(pipeline_input)
    model4.fit(x_train, y_train)
    print("The R-square of the fourth model:", model4.score(x_test, y_test))
