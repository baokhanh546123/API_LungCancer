from enum import Enum
from typing import Optional
from psycopg2 import Error
from pathlib import Path
from dotenv import load_dotenv
import logging , psycopg2 as pysql , os 


current_dir = Path(__file__).parent
env_file = current_dir / 'connection.env'
load_dotenv(env_file)

logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')

class Role(Enum):
    MASTER = 'Master'
    MEMBER = 'Member'

def connect_server():
    try:
        connection = pysql.connect(
            dbname=os.getenv('DBNAME_SQL'),
            user=os.getenv('USER_SQL'),
            password=os.getenv('PASSWORD_SQL'),
            host=os.getenv('HOST_SQL'),
            port=os.getenv('PORT_SQL')
        )
        cursor = connection.cursor()
        if cursor:
            logging.info("Database connection established successfully.")
        return connection
    except (Exception, Error) as error:
        logging.error("Error while connecting to PostgreSQL: %s", error)
        return None

def get_members_table(member_id: Optional[int] = None , role: Optional[Role] = None , full_name: Optional[str] = None , url_image: Optional[str] = None ,
                      email : Optional[str] = None , query : str = 'all'):
    connection = None
    cursor = None
    try:
        connection = connect_server()
        if connection is None:
            logging.error("Failed to connect to the database.")
            return None
        cursor = connection.cursor()
        if query == 'all':
            cursor.execute("SELECT * FROM members")
            members = cursor.fetchall()
            logging.info("All members retrieved successfully.")
            return members
        else:
            cursor.execute(f"SELECT {role}, {full_name},  {email} , {url_image} FROM members WHERE member_id = %s", (member_id,))
            member = cursor.fetchone()
            logging.info("Member retrieved successfully.")
            return member
    except (Exception, Error) as error:
        logging.error("Error while retrieving member: %s", error)
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logging.info("Database connection closed.")
            

def length_member_id():
    connection = None
    cursor = None
    try:
        connection = connect_server()
        if connection is None:
            logging.error("Failed to connect to the database.")
            return None
        cursor = connection.cursor()
        cursor.execute("SELECT member_id FROM members")
        member_id = cursor.fetchall()
        len_member_id = len(member_id)
        logging.info("Member retrieved successfully.")
        return len_member_id
    except (Exception, Error) as error:
        logging.error("Error while retrieving member: %s", error)
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logging.info("Database connection closed.")


def produre_add_members(role : Role, full_name  : str , email : str , url_img : Optional[str] = None):
    connection = None
    cursor = None
    try:
        connection = connect_server()
        if connection is None:
            logging.error("Failed to connect to the database.")
            return 
        cursor = connection.cursor()
        role_value = role.value if isinstance(role, Role) else str(role)
        cursor.execute("CALL add_members(%s, %s, %s , %s)", (role_value, full_name, email , url_img))
        connection.commit()
        logging.info("Member added successfully.")
    except (Exception, Error) as error:
        logging.error("Error while executing stored procedure: %s", error)
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logging.info("Database connection closed.")

def produre_update_members(member_id: int, role: Role, full_name: str, email: str , url_img: Optional[str] = None):
    connection = None
    cursor = None
    try:
        connection = connect_server()
        if connection is None:
            logging.error("Failed to connect to the database.")
            return 
        cursor = connection.cursor()
        role_value = role.value if isinstance(role, Role) else str(role)
        cursor.execute("CALL update_member(%s, %s, %s, %s , %s)", (member_id, role_value, full_name, email , url_img))
        connection.commit()
        logging.info("Member updated successfully.")
    except (Exception, Error) as error:
        logging.error("Error while executing stored procedure: %s", error)
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logging.info("Database connection closed.")

def produre_delete_members(member_id:int):
    connection = None
    cursor = None
    try:
        connection = connect_server()
        if connection is None:
            logging.error("Failed to connect to the database.")
            return 
        cursor = connection.cursor()
        cursor.execute("CALL delete_member(%s)", (member_id,))
        connection.commit()
        logging.info("Member deleted successfully.")
    except (Exception, Error) as error:
        logging.error("Error while executing stored procedure: %s", error)
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logging.info("Database connection closed.")


#if __name__ == "__main__":
    #connect_server()
    #get_members_table()
    #produre_add_members(Role.MASTER, "Nguyen Van A",'3@gmail.com')
    #produre_update_members(3, Role.MEMBER, "Nguyen Van B", '3@gmail.com')
    #produre_delete_members(4)