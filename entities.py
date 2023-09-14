from pony.orm import *
from datetime import datetime

db = Database()

class Detection(db.Entity):
    id = PrimaryKey(int, auto=True)
    id_tracking = Required(str)
    clase = Required(str)
    fecha = Required(datetime)
    
class Update(db.Entity):
    id = PrimaryKey(int, auto=True)
    last_update = Required(datetime)
    
db.bind(provider='sqlite', filename='database.sqlite', create_db=True)
db.generate_mapping(create_tables=True)