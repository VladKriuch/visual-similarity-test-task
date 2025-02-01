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
                        "vector": {
                            "type": "dense_vector",
                            "dims": 512,
                        "similarity": "cosine",
                        },
                        "filepath": {
                            "type": "keyword",
                        },
                    },
                },
            }
        )
    
    def insert_index(self, filepath: str, vector: list):
        """Inserts index

        Args:
            filepath (str): filepath to the image. Used for displaying image results
            vector (list): Vector generated from encoding model
        """
        self.client.index(
            index=self.index_name,
            body={
                "vector": vector,
                "filepath": filepath,
            }
        )
    
    def search(self, query_vector: list, k: int = 5):
        """Performs k-nearest-neighbors search

        Args:
            vector (list): embedding vector
            k (int, optional): k-nearest-search parameter. Defaults to 5.
        """
        response = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_vector,  # The input vector
                        "k": k
                    }
                }
            }
        )
        return response["hits"]["hits"]