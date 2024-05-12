from bs4 import BeautifulSoup

content = '/home/jake/Programming/Llama3-Chatbot/URLList/parsers/Family Weekend _ Towson University.html'
with open(content, 'r', encoding='utf-8') as f:
    html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    selectors = ['div#skip-to-main', 'div.row', 'div.utility', 'div.main', 'div.mobile', 'div.links', 'div.secondary', 'div.bottom', 'div.sidebar', 'nav.subnavigation', 'div#subnavigation', 'div.subnavigation', 'div.sidebar']
    for div in soup.select(' , '.join(selectors)):
        div.decompose()
    for noscript_tag in soup.find_all('noscript'):
        noscript_tag.decompose()
    souped_text = soup.get_text(strip=True, separator=" ")
    print(str(souped_text))