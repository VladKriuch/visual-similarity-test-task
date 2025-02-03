import os

from elasticsearch import Elasticsearch

class ElasticDB:
    """Abstraction for performing operations with Elastic DataBase"""
    
    INSERT_QUERY_REQUIRED_PARAMS = {
        "image_vector": lambda vector: isinstance(vector, list),
        "filepath": lambda filepath: isinstance(filepath, str) and os.path.exists(filepath),
        "vector_combination": lambda vector: isinstance(vector, list)
    }
    
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
                        "logo_vector": { # Check if required at all
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
                        "is_single_product": { # TODO: Check if required at all
                            "type": "boolean"
                        }
                    },
                },
            }
        )
    
    def validate_insert_query(self, insert_query_kwargs: dict):
        """Validation for key-value pairs in insert index params

        Args:
            insert_query_kwargs (dict): Params given to insert_index function
        """
        for key, validation_func in self.INSERT_QUERY_REQUIRED_PARAMS.items():
            if not key in insert_query_kwargs:
                raise KeyError(f"Parameter {key} is required to insert object into DB")
            if not validation_func(insert_query_kwargs[key]):
                raise ValueError(f"Parameter {key} failed to pass validation")
        
    def insert_index(self, **kwargs):
        """Inserts index

        Args:
            kwargs (dict): Parameters to be inserted
            
            Required params
            {
                "image_vector": vector,
                "filepath": filepath,
                "combination_vector": vector,
            }
            Non-required params
            {
                "bbox": bbox,
                "is_single_product": is_single_product,
                "text_vector": vector,
                "category": string
            }
        """
        
        
        _body = kwargs
        self.client.index(
            index=self.index_name,
            body=_body
        )
    
    def search(self, 
               query_vector: list, 
               label: str = None, 
               k: int = 5, 
               single_product_only: bool = False,
               full_image_search: bool = True):
        """Multi-staged search system with filters

        Args:
            query_vector (list): query vector obtained from querring model
            label (str, optional): label for category, use None to not include categories into filters
            k (int, optional): _description_. Defaults to 5. K parameter of nearest neighbors search
            single_product_only (bool, optional): _description_. Defaults to True. Whether to look onto images where single product was detected
            full_image_search (bool, optional): _description_. Defaults to True. Whether to perform full image search only or include detected images

        Returns:
            Results from elasticdb search query
        """
        # Search is multi-staged thing
        _must = []
        
        # Include labels into filters
        if label is not None:
            # Some of the labels may overlap, thus there's a bit of hardcoded mapping involved 
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
        
        if full_image_search:
            _must.append({
                "term": {
                    "bbox": "None"
                }
            })
        
        # Get response from elasticdb
        response = self.client.search(
                    index=self.index_name,
                    body={
                        "query": {
                            "knn": {
                                "field": "image_vector",
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