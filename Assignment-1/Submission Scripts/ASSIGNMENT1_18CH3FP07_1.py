# scrape.py
# scrapes earning call data from Seeking Alpha

import requests
import time
import random
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from itertools import cycle

def get_proxies():
    ua=UserAgent()
    headers = {'User-Agent':ua.random}
    url='https://free-proxy-list.net/'

    r=requests.get(url,headers=headers)
    page = BeautifulSoup(r.text, 'html.parser')

    proxies=[]

    for proxy in page.find_all('tr'):
        i=ip=port=0

    for data in proxy.find_all('td'):
        if i==0:
            ip=data.get_text()
        if i==1:
            port=data.get_text()
        i+=1

    if ip!=0 and port!=0:
        proxies+=[{'http':'http://'+ip+':'+port}]

    return proxies

def get_date(c):
    end = c.find('|')
    return c[0:end-1]

def get_ticker(c):
    beg = c.find('(')
    end = c.find(')')
    return c[beg+1:end]

def grab_page(url,proxy):
    print("attempting to grab page: " + url)
    ua = UserAgent()
    headers = {'User-Agent':ua.random}
    proxies={"http":proxy,"https":proxy}
    page = requests.get(url,headers=headers,proxies=proxies)
 
    page_html = page.text
    soup = BeautifulSoup(page_html, 'html.parser')

    meta = soup.find("div",{'class':'a-info get-alerts'})
    content = soup.find(id="a-body")

    if type(meta) or type(content) == "NoneType":
        print("skipping this link, no content here")
        return
    else:
        text = content.text
        mtext = meta.text

        filename = get_ticker(mtext) + "_" + get_date(mtext)
        file = open(filename.lower() + ".txt", 'w')
        file.write(text)
        file.close
        print(filename.lower()+ " sucessfully saved")

def process_list_page(i,proxy):
    origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + "/" + str(i)
    print("getting page " + origin_page)
    page = requests.get(origin_page)
    page_html = page.text
    #print(page_html)
    soup = BeautifulSoup(page_html, 'html.parser')
    alist = soup.find_all("li",{'class':'list-group-item article'})
    for i in range(0,len(alist)):
        url_ending = alist[i].find_all("a")[0].attrs['href']
        url = "https://seekingalpha.com" + url_ending
        grab_page(url,proxy)
        time.sleep(random.uniform(0.5,3.0))  # CHANGE: Variable delay for better outcome

for i in range(1,10): #choose what pages of earnings to scrape
    proxies=get_proxies()
    proxy_cycle = cycle(proxies)
    process_list_page(i,next(proxy_cycle))