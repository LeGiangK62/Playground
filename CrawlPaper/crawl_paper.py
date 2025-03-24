import requests
from bs4 import BeautifulSoup
import csv

# Set the base URL for AAAI conference years
years = range(2018, 2025)  # Adjust range as needed
base_url = "https://dblp.org/db/conf/aaai/aaai{}.html"

keyw = 'quantum'
file_name_txt = keyw + '_papers.txt'
file_name_csv = keyw + '_papers.csv'
all_papers = []
print(f"\n Crawling {keyw}-related papers in AAAI-{years[0]} to AAAI-{years[-1]}")
for year in years:
    url = base_url.format(year)
    print(f"Fetching {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # All titles are in <cite> under <li class="entry ...">
    entries = soup.find_all('li', class_='entry inproceedings')

    for entry in entries:
        title_tag = entry.find('cite')
        if title_tag:
            title = title_tag.text.strip()
            if keyw.lower() in title.lower():
                all_papers.append((year, title))

# Save to text file
with open(file_name_txt, "w", encoding="utf-8") as txt_file:
    for year, title in all_papers:
        txt_file.write(f"[{year}] {title}\n")

# Save to CSV file
with open(file_name_csv, "w", encoding="utf-8", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Year", "Title"])
    writer.writerows(all_papers)

print(f"\nSaved {len(all_papers)} {keyw}-related papers to {file_name_txt} and '{file_name_csv}'")

