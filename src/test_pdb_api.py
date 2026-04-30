import requests
import json

# Test with correct attribute name
query = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
            "operator": "exact_match",
            "value": "353153"
        }
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 10
        }
    }
}

url = "https://search.rcsb.org/rcsbsearch/v2/query"
response = requests.post(url, json=query, timeout=30)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:1000]}")