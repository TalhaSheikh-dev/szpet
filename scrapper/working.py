
import requests 
from bs4 import BeautifulSoup
import json

def id_scrapper():
    data = {
        '_token': '',
        'email': 'matt@salezilla.io',
        'password': 'Skillzilla69!?',
        'button': ''
    }

    url = "https://app.wavo.co/login" 
    main_req_url = "https://app.wavo.co/campaigns/{}/inbox?keywords=&interest=&status=&account_id=0&currentPage=1&bottom=false&get_count=false&page={}&display="
    url_campaing = "https://app.wavo.co/campaigns/list?team_id=0&teammate_id=0&account_id=0&status=ALL&page={}&per_page=10&keywords=&agency_id=v0ml2n35100n769eypwo"
    all_urls = []

    with requests.Session() as s:
        response = s.get(url)
        soup = BeautifulSoup(response.text)
        token = soup.find('input', {'name': "_token"})['value']
        data['_token'] = token
        response = s.post(url, data=data)
        counter = 1
        while True:
            response = s.get(url_campaing.format(counter))
            data = json.loads(response.text)
            data_list = data["campaigns"]["data"]
            if len(data_list) == 0:
                break
            for item in data_list:
                all_urls.append(item["hashid"])
            counter = counter+1

        message = []
        status = []
        for url in all_urls:
            camp_id_single = url.split("/")[-1]
            i = 1
            while True:
                url_to_hit = main_req_url.format(camp_id_single,i)
                response = s.get(url_to_hit)

                json_data = json.loads(response.text)
                main_data_list = json_data["prospects_thread"]["data"]
                if len(main_data_list) == 0:
                    break
                
                for single in main_data_list:
                    status.append(single["status"])
                    message.append(single["email_threads"][0]["snippet"])

                i = i+1
            
        
        import pandas as pd
        df = pd.DataFrame({"text":message,"status":status})        
        #df.to_csv("testing.csv")
    return df        
        




            
        

