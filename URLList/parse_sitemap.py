from parsel import Selector
import httpx
import os

def parse_sitemap(output_dir):
    sitemap_url_path = "https://www.towson.edu/sitemap.xml"
    response = httpx.get(sitemap_url_path)
    selector = Selector(response.text)
    urls = []
    pdfs = []
    for url in selector.xpath('//url'):
        location = url.xpath('loc/text()').get()
        modified = url.xpath('lastmod/text()').get()
        if ".html" in location:
            urls.append(location)
        else:
            pdfs.append(location)
    write_list_to_file(output_dir=output_dir, file_name="urls.txt", data=urls)
    write_list_to_file(output_dir=output_dir, file_name="pdfs.txt", data=pdfs)


def write_list_to_file(output_dir, file_name, data):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    parse_sitemap("URLList")
