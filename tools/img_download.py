from googleapiclient.discovery import build


key = 'AIzaSyAuJsWeELieQPqODi0Hx8mA1BxUs3grjLE'
engine_id = '003868579984038687904:rlufnvo9uu4'


def search(search_term, num=None, api_key=key, cse_id=engine_id):
    """Search the web for images using Google custom search

    Args:
        search_term(str): string image search term
        num(int): number of search results to return
        api_key(str): custom search engine api key
        cse_id(str): custom search engine id

    Returns:
        A list of dictionaries, each represents a search result
    """
    service = build('customsearch', 'v1', developerKey=api_key)
    results = service.cse().list(q=search_term,
                                 cx=cse_id,
                                 num=num,
                                 searchType='image',
                                 imgType='photo',
                                 imgSize='huge',
                                 imgColorType='color').execute()
    return results['items']
