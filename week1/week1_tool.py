import http.client
import json
from math import sqrt

def catch_products(cateid, max_pages=None):
    base_url = "ecshweb.pchome.com.tw"
    endpoint = "/search/v4.3/all/results"
    all_products = []
    page = 1
    
    while True:
        params = f"cateid={cateid}&attr=&pageCount=40&page={page}"
        url = f"{endpoint}?{params}"
        
        connection = http.client.HTTPSConnection(base_url)

        connection.request("GET", url)
        response = connection.getresponse()
        
        if response.status != 200:
            print(f"Failed to catch page {page}. Status code: {response.status}")
            break
        
        data = json.loads(response.read().decode("utf-8"))
        products = data.get("Prods", [])
        all_products.extend(products)
        
        print(f"catch page {page} with {len(products)} products.")

        # 檢查是否到最後一頁
        total_pages = data.get("TotalPage", 1)
        if max_pages and page >= max_pages:
            break
        if page >= total_pages:
            break        
        page += 1

    return all_products

def calculate_z_score(data, value_col):
    values = []
    for item in data:
        if item.get(value_col) is not None:
            values.append(item[value_col])
    mean_value = sum(values) / len(values)
    var = sum((x-mean_value)**2 for x in values) / len(values)
    std = sqrt(var)
    
    for item in data:
        value = item.get(value_col)
        if value is not None:
            item["ZScore"] = (value - mean_value) / std
    
    return data
    