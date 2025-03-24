import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import pandas as pd  # for Excel export

# Settings
years = range(2018, 2025)
base_url = "https://dblp.org/db/conf/aaai/aaai{}.html"
keyw = 'quantum'

# Output filenames
file_name_txt = f"{keyw}_papers.txt"
file_name_csv = f"{keyw}_papers.csv"
file_name_json = "all_papers.json"
file_name_excel = "all_papers.xlsx"

# Storage
quantum_papers = []
all_papers = {}

print(f"\nCrawling {keyw}-related papers in AAAI-{years[0]} to AAAI-{years[-1]}")

for year in years:
    url = base_url.format(year)
    print(f"Fetching {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    entries = soup.find_all('li', class_='entry inproceedings')
    year_titles = []

    for entry in entries:
        title_tag = entry.find('cite')
        if title_tag:
            title = title_tag.text.strip()
            year_titles.append(title)
            if keyw.lower() in title.lower():
                quantum_papers.append((year, title))

    all_papers[year] = year_titles
    print(
        f"  Found {len(year_titles)} total papers, {sum(1 for t in year_titles if keyw.lower() in t.lower())} related to '{keyw}'")
    time.sleep(1)

# Save quantum papers to text
with open(file_name_txt, "w", encoding="utf-8") as txt_file:
    for year, title in quantum_papers:
        txt_file.write(f"[{year}] {title}\n")

# Save quantum papers to CSV
with open(file_name_csv, "w", encoding="utf-8", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Year", "Title"])
    writer.writerows(quantum_papers)

# Save all papers to JSON
with open(file_name_json, "w", encoding="utf-8") as json_file:
    json.dump(all_papers, json_file, indent=2, ensure_ascii=False)

# Save all papers to Excel
excel_data = []
for year, titles in all_papers.items():
    for title in titles:
        excel_data.append({"Year": year, "Title": title})

df = pd.DataFrame(excel_data)
df.to_excel(file_name_excel, index=False)

print(f"\nSaved {len(quantum_papers)} '{keyw}' papers to {file_name_txt} and {file_name_csv}")
print(f"Saved all {sum(len(t) for t in all_papers.values())} papers to {file_name_json} and {file_name_excel}")
