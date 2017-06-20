from googleapiclient.discovery import build


key = 'AIzaSyAuJsWeELieQPqODi0Hx8mA1BxUs3grjLE'
engine_id = '003868579984038687904:rlufnvo9uu4'


def search(search_term, api_key=key, cse_id=engine_id):
    service = build('customsearch', 'v1', developerKey=api_key)
    results = service.cse().list(q=search_term,
                                 cx=cse_id,
                                 searchType='image',
                                 imgType='photo',
                                 imgSize='medium',
                                 imgColorType='color').execute()
    return results['items']
