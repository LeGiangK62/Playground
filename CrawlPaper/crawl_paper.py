import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import pandas as pd

# Settings
years = range(2018, 2025)
base_url = "https://dblp.org/db/conf/aaai/aaai{}.html"
keyw = 'quantum'

# Output filenames
file_name_txt = f"{keyw}_papers.txt"
file_name_csv = f"{keyw}_papers.csv"
file_name_json = f"{keyw}_papers.json"
file_name_excel = f"{keyw}_papers.xlsx"

# Storage
quantum_papers = []  # List of (year, title)
filtered_papers = {}  # Dict: year -> [titles]

print(f"\nCrawling '{keyw}' papers from AAAI-{years[0]} to AAAI-{years[-1]}")

for year in years:
    url = base_url.format(year)
    print(f"Fetching {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    entries = soup.find_all('li', class_='entry inproceedings')
    matched_titles = []

    for entry in entries:
        title_tag = entry.find('cite')
        if title_tag:
            title = title_tag.text.strip()
            if keyw.lower() in title.lower():
                quantum_papers.append((year, title))
                matched_titles.append(title)

    if matched_titles:
        filtered_papers[year] = matched_titles
        print(f"  Found {len(matched_titles)} '{keyw}' papers in {year}")
    time.sleep(1)

# Save to text
with open(file_name_txt, "w", encoding="utf-8") as txt_file:
    for year, title in quantum_papers:
        txt_file.write(f"[{year}] {title}\n")

# Save to CSV
with open(file_name_csv, "w", encoding="utf-8", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Year", "Title"])
    writer.writerows(quantum_papers)

# Save to JSON
with open(file_name_json, "w", encoding="utf-8") as json_file:
    json.dump(filtered_papers, json_file, indent=2, ensure_ascii=False)

# Save to Excel
excel_data = [{"Year": year, "Title": title} for year, titles in filtered_papers.items() for title in titles]
df = pd.DataFrame(excel_data)
df.to_excel(file_name_excel, index=False)

print(f"\n✅ Saved {len(quantum_papers)} '{keyw}' papers to:")
print(f"  → {file_name_txt}")
print(f"  → {file_name_csv}")
print(f"  → {file_name_json}")
print(f"  → {file_name_excel}")
