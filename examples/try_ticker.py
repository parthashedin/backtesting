import logging
from kiteconnect import KiteTicker

logging.basicConfig(level=logging.DEBUG)


api_key = "p0zhrqp68wf4qd9j"#userdata.loc[i, "api_key"]
api_secret = "hc13yo68q7bnvivchixnjvud0q5y5fp4" #userdata.loc[i, "api_secret"]
request_token = "q5PiOaOKLAR3O6R3BaYsa7gZt3yvo7vS" #userdata.loc[i, "request_token"]
access_token = "HLpd5AkkDcSF6Ac6wMq3CAVaT6fQCgxS" #userdata.loc[i, "access_token"]
public_token = "SLhnTyHDSRytgQ3TKXgb3QreniIlAdaE" #userdata.loc[i, "public_token"]
# Initialise
kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):  # noqa
    # Callback to receive ticks.
    logging.info("Ticks: {}".format(ticks))

def on_connect(ws, response):  # noqa
    # Callback on successful connect.
    # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
    #list all tickers you want to trade
    # tickerlist = ["ICICIBANK"]
    # tokenlist = [1270529]
    ws.subscribe([1270529])

    # Set RELIANCE to tick in `full` mode.
    ws.set_mode(ws.MODE_FULL, [1270529])

def on_order_update(ws, data):
    logging.debug("Order update : {}".format(data))

# Assign the callbacks.
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_order_update = on_order_update

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
kws.connect()