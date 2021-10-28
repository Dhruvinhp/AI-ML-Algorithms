import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/Law'
fr = [] 
wanted = ['legal']    
a = requests.get(url).text
soup = BeautifulSoup(a, 'html.parser')
for word in wanted:
    freq = soup.get_text().lower().count(word)
    # print(freq)
    dic = {'phrase': word, 'frequency': freq}          
    fr.append(dic)  
    print('Frequency of', word, 'is:', freq)
