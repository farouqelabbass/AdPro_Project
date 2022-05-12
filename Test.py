from energy import Energy
import warnings
warnings.filterwarnings("ignore")

test = Energy()
df = test.read_data(True)
test.arima_forecast("Germany", 10)
# test.consumption_country(["Czechia", "Morocco", "Bulgaria"])
# test.allcountries_scatter(2016)
# ch = df[df["country"] == "France"]
# ch.plot(y="emissions")
# test.consumption_area_plot("France", True)
# test.gdp_over_years('sebhgj')
