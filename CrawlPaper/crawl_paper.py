import requests
from bs4 import BeautifulSoup

# Set the base URL for AAAI conference years
years = range(2018, 2024)  # Adjust range as needed
base_url = "https://dblp.org/db/conf/aaai/aaai{}.html"

quantum_papers = []

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
            if "quantum" in title.lower():
                quantum_papers.append((year, title))

# Save to text file
with open("quantum_papers.txt", "w", encoding="utf-8") as txt_file:
    for year, title in quantum_papers:
        txt_file.write(f"[{year}] {title}\n")

# Save to CSV file
with open("quantum_papers.csv", "w", encoding="utf-8", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Year", "Title"])
    writer.writerows(quantum_papers)

print(f"\nSaved {len(quantum_papers)} quantum-related papers to 'quantum_papers.txt' and 'quantum_papers.csv'")

