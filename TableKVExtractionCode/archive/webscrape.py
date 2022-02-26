import requests
from bs4 import BeautifulSoup

URL = 'https://www.digikey.com/en/products/detail/ATMEGA164P-20AU/ATMEGA164P-20AU-ND/1245833?utm_campaign=buynow&utm_medium=aggregator&curr=usd&utm_source=octopart'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id="__NEXT_DATA__")
print(results.prettify())
