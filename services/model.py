import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

#Path to the CSV file with data to train the model
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "HistoricalSalesData.csv")

def train_model():
    # Loading the CSV file with past sales info
    df = pd.read_csv(CSV_PATH)

    # Turning item names into numbers so the model can understand them
    le = LabelEncoder()
    df['ItemEncoded'] = le.fit_transform(df['Item Name'])

    # Converting sale dates into numbers so the model can use them
    df['SaleDate_Ordinal'] = pd.to_datetime(df['Sale Date']).map(pd.Timestamp.toordinal)

    # Preparing a dictionary with the last 5 records per item variant to see a more recent snapshot of the item's value
    item_avg_dict = {}
    for item in df['Item Name'].unique():
        # Gets the last 5 sales for this item
        item_df = df[df['Item Name'] == item].sort_values('SaleDate_Ordinal').tail(5)
        X = item_df[['ItemEncoded', 'SaleDate_Ordinal']]
        y = item_df['Total Cost']
        # Making a simple model that predicts total cost from item and date
        model = LinearRegression()
        model.fit(X, y)
        # Saving the model, encoded value, and average of these 5 sales.
        item_avg_dict[item] = {
            'model': model,
            'le_value': le.transform([item])[0],
            'historical_avg': y.mean()
        }

    return item_avg_dict, le

def predict_total_cost(item_name, sale_date, item_avg_dict, le):
    # Returns nothing if data for the item doesn't exist
    if item_name not in item_avg_dict:
        return None, None

    model_info = item_avg_dict[item_name]
    model = model_info['model']
    le_value = model_info['le_value']
    historical_avg = model_info['historical_avg']

    # Converts the new sale date to a number
    sale_date_ord = pd.to_datetime(sale_date).toordinal()
    X_new = pd.DataFrame([[le_value, sale_date_ord]], columns=['ItemEncoded', 'SaleDate_Ordinal'])

    # Predicts the total cost for this item on this date
    predicted_total = model.predict(X_new)[0]

    return historical_avg, predicted_total



