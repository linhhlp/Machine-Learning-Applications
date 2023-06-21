"""This is a demo of credentials 
Rename to cred.py
"""
import os

# CO.HERE AI API
# get free Trial API Key at https://cohere.ai/
API_key = "D******g"

# CASSANDRA CONNEECTION API (DataStax in this case)
SECURE_CONNECT_BUNDLE_PATH = os.path.join(
    os.path.abspath(""),
    "PATH_TO_ZIP_FILE_SECURE_CONNECT_BUNDLE",
    # for example
    # "secure-connect-movies-vector-search.zip"
)
ASTRA_CLIENT_ID = "USER_ID"
ASTRA_CLIENT_SECRET = "PASSWORD"
