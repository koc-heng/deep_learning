# Description: This is a tool for catching PTT titles.  

import requests
from bs4 import BeautifulSoup
import csv
import time

def catch_ptt_title(board_url, board_name, max_titles=200000, output_csv='ptt.csv'):
    
    current_url = board_url
    count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['board', 'nrec', 'title', 'link', 'author', 'date']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            try:
                resp = requests.get(current_url, timeout=45)
                resp.raise_for_status() 
            except Exception as e:
                print(f"error: {current_url}\n{e}")
                break

            soup = BeautifulSoup(resp.text, 'html.parser')
            entries = soup.find_all('div', class_='r-ent')
            if not entries:
                print(f"we can't find r-ent {current_url}")
                break

            for ent in entries:
                nrec_div = ent.find('div', class_='nrec')
                nrec_str = nrec_div.get_text(strip=True) if nrec_div else ''

                title_div = ent.find('div', class_='title')
                if title_div:
                    a_tag = title_div.find('a')
                    if a_tag:
                        title_text = a_tag.text.strip()
                        link = a_tag['href']
                    else:
                        title_text = '(no title)'
                        link = ''
                else:
                    title_text = '(title_div missing)'
                    link = ''

                author_div = ent.find('div', class_='author')
                author = author_div.get_text(strip=True) if author_div else ''

                date_div = ent.find('div', class_='date')
                date_str = date_div.get_text(strip=True) if date_div else ''

                writer.writerow({
                    'board' : board_name,
                    'nrec'  : nrec_str,  
                    'title' : title_text,
                    'link'  : link,
                    'author': author,
                    'date'  : date_str
                })
                count += 1

                if count >= max_titles:
                    print(f"[ok] catch {max_titles} :D ")
                    return 

            paging_div = soup.find('div', class_='btn-group btn-group-paging')
            if not paging_div:
                print("....no paging_div")
                break

            a_list = paging_div.find_all('a')
            if len(a_list) < 2:
                print("....no a_list")
                break

            prev_link = a_list[1].get('href', None)
            if not prev_link:
                print("....no prev_link")
                break

            current_url = "https://www.ptt.cc" + prev_link

            time.sleep(1)

    print(f"[end] catch {count} titles ")