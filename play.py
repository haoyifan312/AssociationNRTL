import sqlite3
import pandas as pd


def create_table():
    conn = sqlite3.connect('association_nrtlsac.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE mol_par 
    (name text, 
    X real,
    Y real,
    delta_A real,
    delta_D real,
    nu_A integer,
    nu_D integer,
    r real,
    nu_additional1 integer,
    nu_additional2 integer,
    nu_additional3 integer,
    delta_additional1 real,
    delta_additional2 real,
    delta_additional3 real)''')


def write_table_from_csv():
    conn = sqlite3.connect('association_nrtlsac.db')
    data = pd.read_csv('molecular_parameters.csv')  # load to DataFrame
    # write to sqlite table
    data.to_sql('mol_par', conn, if_exists='append', index=False)


def delete_table():
    conn = sqlite3.connect('association_nrtlsac.db')
    c = conn.cursor()
    c.execute('DROP TABLE mol_par')


