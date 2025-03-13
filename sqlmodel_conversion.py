from sqlmodel import SQLModel, create_engine, Session, Field, select, Session


class Person(SQLModel, table=True):
    # Infers table name from class name

    ssn : int = Field(primary_key=True, index=True)
    firstname: str
    lastname: str
    gender: str
    age: int

    # Init no longer necessary, handled by SQLModel

    def __repr__(self):
        return f"({self.ssn}) {self.firstname} {self.lastname} ({self.gender}, {self.age})"

class Thing(SQLModel, table=True):

    tid: int = Field(primary_key=True, index=True)
    description: str
    owner: int = Field(foreign_key="person.ssn")

    def __repr__(self):
        return f"({self.tid}) {self.description} owned by {self.owner}"

database_url = "sqlite:///mydb.db"
engine = create_engine(database_url, echo = True)
SQLModel.metadata.create_all(bind=engine)


def get_session():
    return Session(engine)

session = get_session()

person = Person(ssn=12312, firstname="Mike", lastname="Smith", gender="m", age=35)
session.add(person) 
session.commit() # apply changes to the database

p1 = Person(ssn=12363, firstname="Anna", lastname="Blue", gender="f", age=30)
p2 = Person(ssn=12314, firstname="Bob", lastname="Blue", gender="m", age=45)
p3 = Person(ssn=15256, firstname="Angela", lastname="Cold", gender="f", age=22)

session.add_all([p1, p2, p3])
session.commit()

t1 = Thing(tid=1, description="Car", owner=p1.ssn)
session.add(t1)
session.commit()

people = session.exec(select(Person)).all()
for person in people:
    print(person)

things = session.exec(select(Thing)).all()
for thing in things:
    print(thing)

# Close the session after use
session.close()