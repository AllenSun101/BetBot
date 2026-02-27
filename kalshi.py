import requests

market_ticker = "KXNCAAMBGAME-26JAN31SAMWCU-WCU"
orderbook_url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/orderbook"

orderbook_response = requests.get(orderbook_url)
orderbook_data = orderbook_response.json()

print(f"\nOrderbook for {market_ticker}:")
print("YES BIDS:")
for bid in orderbook_data['orderbook']['yes']:
    print(f"  Price: {bid[0]}¢, Quantity: {bid[1]}")

print("\nNO BIDS:")
for bid in orderbook_data['orderbook']['no'][:]:
    print(f"  Price: {bid[0]}¢, Quantity: {bid[1]}")

event_ticker = "KXNCAAMBGAME-26JAN31SAMWCU"
event_url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
event_response = requests.get(event_url)
event_data = event_response.json()

print(f"Event Details:")
print(f"Title: {event_data['event']['title']}")
print(f"Category: {event_data['event']['category']}")