from googleapiclient.discovery import build
import requests
import os


key = 'AIzaSyAuJsWeELieQPqODi0Hx8mA1BxUs3grjLE'
engine_id = '003868579984038687904:rlufnvo9uu4'
images_dir = 'images'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686)'
                         ' AppleWebKit/537.17 (KHTML, like Gecko)'
                         ' Chrome/24.0.1312.27 Safari/537.17'}


def download(search_term, pages):
    """Download images

    Args:
        search_term(str): image search term
        pages(int): number of pages of results to return, each page contains 10 images

    Returns:
        None
    """
    def search(search_term, start=None, api_key=key, cse_id=engine_id):
        """Search the web for images using Google custom search

        Args:
            start: The index of the first result to return
            search_term(str): string image search term
            api_key(str): custom search engine api key
            cse_id(str): custom search engine id

        Returns:
            list of stings, a list of links to images
        """
        service = build('customsearch', 'v1', developerKey=api_key)
        results = service.cse().list(q=search_term,
                                     cx=cse_id,
                                     start=start,
                                     searchType='image',
                                     imgType='photo',
                                     imgSize='medium',
                                     imgColorType='color').execute()
        return [item['link'] for item in results['items']]

    links = []
    for i in range(pages):
        links += search(search_term, start=i*10)

    for img_link in links:
        req = requests.get(img_link, stream=True, headers=headers)
        with open(os.path.join(images_dir, img_link.split('/')[-1]), 'wb') as f:
            f.write(req.content)
