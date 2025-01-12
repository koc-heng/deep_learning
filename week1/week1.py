import csv
from  week1_tool import catch_products, calculate_z_score

cateid = "DSAA31"
products = catch_products(cateid)

#task1 得到所有產品的id
with open("product.txt", "w", encoding="utf-8") as file:
    for product in products:
        product_id = product.get("Id")  
        if product_id:
            file.write(f"{product_id}\n")  

print(f"Saved {len(products)} product IDs to product.txt.")
print("-----Finished task 1-----")

#task2 抓取評論數大於等於1且分數高於4.9分的產品
best_product_ids = []
for product in products:
    review_count = product.get("reviewCount") or 0  
    rating_value = product.get("ratingValue") or 0 
    if review_count >= 1 and rating_value > 4.9:
        best_product_ids.append(product.get("Id"))  

with open("best-products.txt", "w", encoding="utf-8") as file:
    for product_id in best_product_ids:
        file.write(f"{product_id}\n")

print(f"Saved {len(best_product_ids)} best product IDs to best-products.txt.")
print("-----Finished task 2-----")

#task 3 抓取i5處理器並計算它的平均價格
i5 = []
total_price = 0
count = 0
for product in products:
    name = product.get("Name", "")
    describe = product.get("Describe", "")
    price = product.get("Price", "")
    
    if "i5" in name or "i5" in describe:
        i5.append(product)  
        total_price += price  
        count += 1  
        
i5_average_price = total_price / count 
print(f"i5 average price is: {i5_average_price}.") 
print("-----Finished task 3-----")

#task4 拿取所有產品並列出它們的價格和z分數
product_price_data = []
for product in products:
    product_id = product.get("Id")  
    price = product.get("Price")   
    if product_id and price is not None:  
        product_price_data.append({"ProductID": product_id, "Price": price})
        
standardization = calculate_z_score(product_price_data, "Price")

with open("standardization.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["ProductID", "Price", "ZScore"])

    for item in standardization:
        writer.writerow([item["ProductID"], item["Price"], item["ZScore"]])

print(f"Saved {len(standardization)} products with Z scores to standardization.csv.")
print("-----Finished task 4-----")
