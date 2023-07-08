import json
from rich import inspect

with open('wiki.json') as wiki_file:
    wiki = json.load(wiki_file)
    inspect(wiki)