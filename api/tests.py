from .search_model import SearchModel


model = SearchModel()
output_get = model.update_query('vitamin C cures COVID-19')
print(output_get)