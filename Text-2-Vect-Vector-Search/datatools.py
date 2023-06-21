"""All needed classes for this module.

    Includes:
    * Database Connectors: Postgre and Cassandra
    * LoadDataFromCSV
    * LoadDataToCassandra
    * LoadDataToPostgre
"""
import ast
import gc
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from cassandra import OperationTimedOut  # , ConsistencyLevel,
from cassandra.cluster import Cluster as CassandraCluster
from cassandra.cluster import ConnectionShutdown, NoHostAvailable
from cassandra.cluster import Session as CassandraSession
from cassandra.policies import (HostFilterPolicy, RoundRobinPolicy,
                                WhiteListRoundRobinPolicy)
from cassandra.query import BatchStatement

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ConnectDB(ABC):
    """Connect from different DBs as ABSTRACT."""

    table_name = None  # could be string or dict

    def __init__(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def execute(self, command):
        pass

    @abstractmethod
    def disconnect(self):
        pass


class CassandraConnector(ConnectDB):
    """Connector for Cassandra.

    This is the simplified connector for the Cassandra in a cluster.
    There are other configuration options available such as:
        * auth_provider
        * get_load_balancing_policy
    """

    session: CassandraSession
    cluster: CassandraCluster

    def __init__(self, cluster, keyspace, table_name):
        self.cluster = cluster
        self.keyspace = keyspace
        self.table_name = table_name
        self.session = None

    def connect(self):
        if self.session is not None:
            log.debug("session is not None, try to shut down first.")
            self.disconnect()

        try:
            self.session = self.cluster.connect()
            # self.create_keyspace(self.keyspace)
            self.set_keyspace(self.keyspace)

        except (OperationTimedOut, NoHostAvailable, ConnectionShutdown) as e:
            print(e)
            err = e

        if self.session is None:
            raise err

    def execute(self, command):
        result = None
        if self.session is None:
            self.connect()

        try:
            result = self.session.execute(command)
        except ConnectionShutdown:
            self.connect()
            result = self.session.execute(command)
        return result

    def prepare(self, command):
        result = None
        try:
            result = self.session.prepare(command)
        except ConnectionShutdown:
            self.connect()
            result = self.session.prepare(command)
        return result

    def get_load_balancing_policy(self):
        return WhiteListRoundRobinPolicy([self.ips[0]])

        # below is the use of whitelist
        def whitelist_address(host):
            return host.address != self.ips[0]

        return HostFilterPolicy(
            child_policy=RoundRobinPolicy(), predicate=whitelist_address
        )

    def create_keyspace(self, keyspace, replication=None):
        if replication is None:
            replication = """WITH replication={
                                'class': 'SimpleStrategy',
                                'replication_factor' : 3
                             }"""
        q = f"CREATE KEYSPACE IF NOT EXISTS {keyspace} " + replication + ";"
        return self.execute(q)

    def set_keyspace(self, keyspace):
        return self.execute(f"USE {keyspace};")

    def disconnect(self):
        try:
            self.cluster.shutdown()
        except Exception:
            log.debug("Can not shut down.")


class LoadDataFromCSV:
    """Load big size of dataset from CSV file in `pandas` DataFrame format."""

    def load_all(self, file_path, dtype=None, chunksize=10**7):
        """Load all at once."""
        if dtype is None:
            dtype = {}
        return pd.concat(
            pd.read_csv(file_path, dtype=dtype, chunksize=chunksize)
        )

    def load_by_chunk(
        self,
        file_path,
        dtype=None,
        chunksize=10**6,
        encoding="utf-8",
        skiprows: int = None,
        fillna=None,
    ):
        """Generate data by chunk as an Iterator/Generator.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Full path to the CSV file
        dtype : dict, optional
            {col:data_type}, by default None
        chunksize : int, optional
            Size (number of row) for each reading iteration, by default 10**6
        encoding : str, optional
            The encode of CSV file, by default "utf-8"
        skiprows : int, optional
            If you want to skip number of row, set with a int, by default None.
            Note, in the program, skiprows become a range which igore 1st row
            as a header
        fillna : dict, optional
            Depending on different col, fill with different value. For example:
            fillna = {
            "date_type": "event_time",  # pd.to_datetime(chunk[col])
            "category_code": "NO_CODE",  # text
            "price": 0.0,  # number / float
            "user_session": "uuid()",  # str(uuid.uuid4())
            }
            By default None

        Yields
        ------
        pd.DataFrame
            Generate dataframe from CSV file based on chunksize
        """
        if skiprows is not None:
            skiprows = range(1, skiprows)  # convert number to range

        if chunksize == -1:
            return pd.read_csv(
                file_path, dtype=dtype, encoding=encoding, skiprows=skiprows
            )

        if dtype is None:
            dtype = {}

        with pd.read_csv(
            file_path,
            dtype=dtype,
            chunksize=chunksize,
            encoding=encoding,
            skiprows=skiprows,
        ) as reader:
            for chunk in reader:
                # deal with missing values
                if fillna is not None:
                    for col, value in fillna.items():
                        if col == "date_type":
                            chunk[value] = pd.to_datetime(chunk[value])
                        else:
                            if value == "uuid()":
                                value = str(uuid.uuid4())
                            chunk[col].fillna(value, inplace=True)

                yield chunk

                del chunk
                gc.collect()  # cleaning memory


class LoadDataToCassandra(LoadDataFromCSV):
    """Load data from CSV files for eCommerce Project."""

    prepared_quey: str

    def prepare_cassandra(self, conn: ConnectDB):
        """Handle Database connection, create if not exist.

        Handle Cassandra connection, create if not exist keyspace and
         table queries
        """
        table_create_query = f"""
CREATE TABLE IF NOT EXISTS {conn.table_name} (
year int,
rank int,
movie_id text,
title text,
imbd_rating float,
imbd_votes int,
plot text,
wiki_link text,
item_vector_1024 VECTOR<FLOAT, 1024>, 
item_vector_4096 VECTOR<FLOAT, 4096>, 
PRIMARY KEY (year, imbd_rating, title)
);
"""
        conn.execute(table_create_query)

        create_index_query = f"""
CREATE CUSTOM INDEX IF NOT EXISTS ann_movie_index_1024 ON 
{conn.table_name}(item_vector_1024) USING 'StorageAttachedIndex';
"""
        conn.execute(create_index_query)

        create_index_query = f"""
CREATE CUSTOM INDEX IF NOT EXISTS ann_movie_index_4096 ON 
{conn.table_name}(item_vector_4096) USING 'StorageAttachedIndex';
"""
        conn.execute(create_index_query)

        self.prepared_quey = f"""
INSERT INTO {conn.table_name} (year, rank, movie_id, title, imbd_rating,
imbd_votes, plot, wiki_link, item_vector_1024, item_vector_4096) 
VALUES (?,?,?,?,?,?,?,?,?,?)
"""

    def save_to_cassandra(
        self,
        conn,
        file_path,
        dtype=None,
        BATCH_SIZE=100,
        CHUNK_SIZE=10**6,
        SKIP_ROWS=None,
    ):
        """Save extracted data to Cassandra.

        First, we iterate data and batch inserts by using
            load_by_chunk to load Data from CSV
        Last, we insert each grouped data into the coresponding table.

        Parameters
        ----------
        conn : Database connection
            Cassandra connection by
            cassandra_conn = DBWrapper("cassandra", info) or CassandraConnector
        file_path : str or pathlib.Path
            Full path to the CSV file
        dtype : dict, optional
            {col:data_type}, by default None
        BATCH_SIZE : int, optional
            The size of a batch when inserting into Cassandra, by default 100
        CHUNK_SIZE : int, optional
            Size (number of row) for each reading iteration, by default 10**6
        SKIP_ROWS : int, optional
            If you want to skip number of row, set with a int, by default None.
            Note, in the program, skiprows become a range which igore 1st row
            as a header
        """
        # Extract information from dataframe
        # Start connecting to Cassandra
        conn.connect()
        # Handle Cassandra connection, create if not exist
        self.prepare_cassandra(conn)

        # Tracking number of rows already processed (inserted to DB)
        rows_number = 0 if SKIP_ROWS is None else SKIP_ROWS
        # Handle missing data
        for df in self.load_by_chunk(
            file_path,
            dtype=dtype,
            chunksize=CHUNK_SIZE,
            skiprows=SKIP_ROWS,
        ):
            log.debug("%s Current # of rows = %s", file_path, rows_number)

            rows_number += CHUNK_SIZE
            # save all to Cassandra
            # log.debug(f" Inserting into Cassandra")
            df.item_vector_1024 = df.item_vector_1024.apply(ast.literal_eval)
            df.item_vector_4096 = df.item_vector_4096.apply(ast.literal_eval)
            # df["movie_id"] = df["movie_id"].str.encode("utf-8")
            # df["plot"] = df["plot"].str.encode("utf-8")
            # df["title"] = df["title"].str.encode("utf-8")
            # df["wiki_link"] = df["wiki_link"].str.encode("utf-8")
            rows = tuple(df.itertuples(index=False, name=None))
            for row in rows:
                conn.session.execute(
                    f"""
INSERT INTO {conn.table_name} (year, rank, movie_id, title, imbd_rating,
imbd_votes, plot, wiki_link, item_vector_1024, item_vector_4096) 
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
                    row,
                )
            # Clean up to release resources (memory)
            del df
            gc.collect()

        # Clean up
        # conn.disconnect()  # disconnect externally
        log.debug(" Finished Inserting into Cassandra")
        gc.collect()
