import kiteconnect

kite = kiteconnect.KiteConnect(api_key="your_api_key")
kite.set_access_token("access_token_from_login_response")

# Get the list of instruments
instruments = kite.instruments()

# Get the instrument token for Nifty 50
instrument = [i for i in instruments if i['tradingsymbol'] == "NIFTY 50"][0]
instrument_token = instrument['instrument_token']

# Get historical data for the last 1 month
from_date = "2022-01-01"
to_date = "2022-02-01"
interval = "daily"

data = kite.historical_data(instrument_token, from_date, to_date, interval)

# Print the data
print(data)
