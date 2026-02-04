from flask import Flask, request, jsonify
from services.model import train_model, predict_total_cost
from flask_cors import CORS
import pandas as pd
import os

# Create the Flask app
app = Flask(__name__)
CORS(app)

# Load CSV in app.py for frontend data
from services.model import CSV_PATH   # use same path as model.py

print("APP CSV_PATH:", CSV_PATH)
print("Exists:", os.path.exists(CSV_PATH))

df = pd.read_csv(CSV_PATH)

# Makes sure that columns have the right data types
df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
for col in ['Sale Price', 'Shipping Cost', 'Total Cost']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# List of all items for dropdown
items = df['Item Name'].dropna().unique().tolist()

# Trains models at startup
item_avg_dict, le = train_model()

@app.route("/")
def health_check():
    return {"status": "ok"}

# Defines the route for predicting item sale value
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    required = ['item_name', 'sale_date', 'shipping_price', 'asking_price']
    for feature in required:
        if feature not in data:
            return jsonify({"error": f"A feature is missing: '{feature}'"}), 400

    item_name = data['item_name']
    sale_date = data['sale_date']
    asking_price = float(data['asking_price'])
    shipping_price = float(data['shipping_price'])

    # Predict total cost
    historical_avg, predicted_total = predict_total_cost(item_name, sale_date, item_avg_dict, le)

    if predicted_total is None:
        return jsonify({"error": f"We are not able to evaluate '{item_name}'"}), 404

    # Compute ROI
    roi = ((predicted_total - (asking_price + shipping_price)) / (asking_price + shipping_price)) * 100
    recommendation = "Buy" if roi > 10 else "Do Not Buy"

    # Check if prediction is within ±15% of historical average
    tolerance = 0.15
    within_range = abs(predicted_total - historical_avg) / historical_avg <= tolerance

    # Log the evaluation
    log_entry = {
        "item_name": item_name,
        "sale_date": sale_date,
        "asking_price": asking_price,
        "shipping_price": shipping_price,
        "predicted_total": predicted_total,
        "historical_avg": historical_avg,
        "within_range": within_range
    }

    LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "prediction_log.csv")
    import csv
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    # Return the results
    return jsonify({
        "historical_avg_sale_value": f"${historical_avg:.2f}",
        "predicted_true_sale_value": f"${predicted_total:.2f}",
        "recommendation": recommendation,
        "roi": f"{roi:.2f}%",
        "prediction_ok": bool(within_range) 
    })

# Provides the frontend with a list of available item names
@app.route("/api/items", methods=["GET"])
def get_items():
    return jsonify(items)

# Route that returns the sales data for the selected item
@app.route("/api/sales/<path:item_name>", methods=["GET"])
def get_sales_for_item(item_name):
    try:
        # Filters the data for the chosen item
        sub = df[df['Item Name'] == item_name].copy()
        if sub.empty:
            return jsonify([])

        # Selects important columns and sorts by date
        sub = sub[['Item Name', 'Sale Price', 'Shipping Cost', 'Total Cost', 'Sale Date']].sort_values('Sale Date')
        sub['Sale Date'] = sub['Sale Date'].dt.strftime('%Y-%m-%d')

        # Converts the sales data to a list of dictionaries and returns it as JSON for frontend
        return jsonify(sub.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



