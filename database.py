import sqlite3
import pandas as pd

from AssociationNRTLSAC import AssociationNRTLSACMolecule


def show_all_data():
    """
    print whole database table of all data
    :return:
    """
    conn = sqlite3.connect('association_nrtlsac.db')
    df = pd.read_sql('''SELECT name text, 
    X,
    Y,
    delta_A,
    delta_D,
    nu_A,
    nu_D,
    r 
    FROM mol_par ''', conn)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)


def get_molecular_data(name: str):
    """
    query Association NRTL-SAC molecular parameters from database
    :param name: str, molecule name as in database, case insensitive
    :return: AssociationNRTLSAC object if molecule in the databse, otherwise None
    """
    name = name.lower()
    conn = sqlite3.connect('association_nrtlsac.db')
    c = conn.cursor()
    t = (name,)
    c.execute('SELECT * FROM mol_par where name=?', t)
    data = c.fetchone()
    if data:
        _, X, Y, delta_A, delta_D, nu_A, nu_D, r, *additional_sites_data = data
        ret = AssociationNRTLSACMolecule(name=name,
                                         r=r,
                                         X=X,
                                         Y=Y,
                                         nu_D=nu_D,
                                         delta_D=delta_D,
                                         nu_A=nu_A,
                                         delta_A=delta_A)
        if additional_sites_data[0] is not None:
            nu_additional = [additional_sites_data[i] for i in range(3) if additional_sites_data[i] is not None]
            delta_additional = [additional_sites_data[i] for i in range(3, 6) if additional_sites_data[i] is not None]
            if len(nu_additional) != len(delta_additional):
                raise Exception(f'additional association site data incorrect for molecule {name} in the database')
            ret.nu_additional_sites = nu_additional
            ret.delta_additional_sites = delta_additional
        return ret


def add_to_database(molecule: AssociationNRTLSACMolecule):
    nu_more = molecule.nu_additional_sites
    delta_more = molecule.delta_additional_sites
    nu1 = nu2 = nu3 = del1 = del2 = del3 = None
    if nu_more and nu_more:
        if len(nu_more) != len(delta_more):
            raise Exception(f'additional association sites data incorrect for molecule {molecule.name}')
        elif len(nu_more) > 3:
            raise Exception(f'additional association sites more than 3 cannot be added to database '
                            f'for molecule {molecule.name}')
        make_up = [None] * (3 - len(molecule.nu_additional_sites))
        nu_more.extend(make_up)
        delta_more.extend(make_up)
        nu1, nu2, nu3, del1, del2, del3 = (nu_more, delta_more)

    row = (molecule.name.lower(),
           molecule.X,
           molecule.Y,
           molecule.delta_A,
           molecule.delta_D,
           molecule.nu_A,
           molecule.nu_D,
           molecule.r,
           nu1,
           nu2,
           nu3,
           del1,
           del2,
           del3)
    conn = sqlite3.connect('association_nrtlsac.db')
    c = conn.cursor()
    c.execute('INSERT INTO mol_par VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', row)
    conn.commit()


def delete_molecule_from_db(name: str):
    """
    delete molecule from databse
    :param name: str, molecule name as in database
    :return:
    """
    conn = sqlite3.connect('association_nrtlsac.db')
    c = conn.cursor()
    c.execute('DELETE FROM mol_par WHERE NAME=?', (name.lower(),))
    conn.commit()

