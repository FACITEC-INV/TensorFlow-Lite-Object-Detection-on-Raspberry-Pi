import orm_sqlite

db = orm_sqlite.Database('detections.db')

class Detection(orm_sqlite.Model):
    id = orm_sqlite.IntegerField(primary_key=True)
    label = orm_sqlite.StringField()
    date = orm_sqlite.StringField()
    
Detection.objects.backend = db