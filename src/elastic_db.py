from elasticsearch import Elasticsearch

class ElasticDB:
    """Abstraction for performing operations with Elastic DataBase"""
    def __init__(self, api_key: str, elastic_uri="http://localhost:9200", index_name="visual-search"):
        """Init

        Args:
            api_key (str): api key for elastic db
            elastic_uri (str, optional): Address to elastic db. Defaults to "http://localhost:9200".
            index_name (str, optional): Main index name. Defaults to "visual-search".
        """
        self.client = Elasticsearch(
            elastic_uri,
            api_key=api_key,
        )
        self.index_name = index_name
        
    def create_index(self):
        """Function for creating main index"""
        self.client.indices.create(
            index=self.index_name,
            body={
                "mappings": {
                    "properties": {
                        "image_vector": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine",
                        },
                        "text_vector": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine"
                        },
                        "logo_vector": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine"
                        },
                        "vector_combination": {
                            "type": "dense_vector",
                            "dims": 512,
                            "similarity": "cosine"
                        },
                        "filepath": {
                            "type": "keyword",
                        },
                        "category": {
                            "type": "keyword"
                        },
                        "bbox": {
                            "type": "keyword"
                        },
                        "is_single_product": {
                            "type": "boolean"
                        }
                    },
                },
            }
        )
    
    def insert_index(self, **kwargs):
        """Inserts index

        Args:
            filepath (str): filepath to the image. Used for displaying image results
            vector (list): Vector generated from encoding model
            
            {
                "vector": vector,
                "filepath": filepath,
                "category": category,
                "bbox": bbox,
                "is_single_product": is_single_product
            }
        """
        _body = kwargs
        self.client.index(
            index=self.index_name,
            body=_body
        )
    
    def search(self, query_vector: list, label: str, k: int = 5, single_product_only: bool = True):
        """Performs k-nearest-neighbors search

        Args:
            vector (list): embedding vector
            k (int, optional): k-nearest-search parameter. Defaults to 5.
        """
        _must = []
        if label is not None:
            labels_mapper = {
                "Musical instruments": ["Musical instruments"],
                "Vehicles": ["Vehicles"],
                "Furniture": ["Furniture"],
                "Food": ["Food"],
                "Instruments": ["Instruments", "Electronics"],
                "Clothing": ["Clothing", "Sports"],
                "Sports": ["Sports", "Clothing"],
                "Accessories": ["Accessories"],
                "Books": ["Books"],
                "Electronics": ["Electronics", "Instruments"],
                "Cosmetics": ["Cosmetics", "Food", "Accessories"],
                "Footwear": ["Footwear"]
            }
            _must.append({
                "terms": {
                    "category": labels_mapper[label]
                }
            })
        
        if single_product_only:
            _must.append({
            "term": {
                "is_single_product": single_product_only
            }})
        
        response = self.client.search(
                    index=self.index_name,
                    body={
                        "query": {
                            "knn": {
                                "field": "vector_combination",
                                "query_vector": query_vector,
                                "k": k,
                                "filter": {
                                    "bool": {
                                        "must":_must
                                    }
                                }
                            }
                        }
                    }
                )
        return response["hits"]["hits"]