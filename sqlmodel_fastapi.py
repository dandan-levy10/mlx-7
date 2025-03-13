from sqlmodel import SQLModel, create_engine, Session, Field, select, Session
from fastapi import FastAPI, Depends, HTTPException
from typing import Optional

app = FastAPI()

class Person(SQLModel, table=True): # Database model
    # Infers table name from class name

    ssn : int = Field(primary_key=True, index=True)
    firstname: str
    lastname: str
    gender: str
    age: int

    # Init no longer necessary, handled by SQLModel

    def __repr__(self):
        return f"({self.ssn}) {self.firstname} {self.lastname} ({self.gender}, {self.age})"

class PersonCreate(SQLModel): # Request model- note no table=True

    firstname: str
    lastname: str
    gender: str
    age: int

class Thing(SQLModel, table=True):

    tid: int = Field(primary_key=True, index=True)
    description: str
    owner: int = Field(foreign_key="person.ssn")

    def __repr__(self):
        return f"({self.tid}) {self.description} owned by {self.owner}"

database_url = "sqlite:///mydb.db"
engine = create_engine(database_url, echo = True)
SQLModel.metadata.drop_all(engine) # NOTE: deletes all existing tables in db before recreating- delete in production
SQLModel.metadata.create_all(bind=engine)


def get_session():
    with Session(engine) as session:
        yield session

session = Session(engine) # Directly create a session- for non-FastAPI use

person = Person(ssn=12312, firstname="Mike", lastname="Smith", gender="m", age=35)
session.add(person) 

p1 = Person(ssn=12363, firstname="Anna", lastname="Blue", gender="f", age=30)
p2 = Person(ssn=12314, firstname="Bob", lastname="Blue", gender="m", age=45)
p3 = Person(ssn=15256, firstname="Angela", lastname="Cold", gender="f", age=22)

session.add_all([p1, p2, p3])

t1 = Thing(tid=1, description="Car", owner=p1.ssn)
session.add(t1)

people = session.exec(select(Person)).all()
for person in people:
    print(person)

things = session.exec(select(Thing)).all()
for thing in things:
    print(thing)

# Apply changes to the database
session.commit() 
# Close the session after use
session.close()

# Check if person exists
def check_person_exists(session: Session, firstname: str, lastname:str):
    statement = select(Person).where(Person.firstname == firstname, Person.lastname == lastname)
    result = session.exec(statement).first() # Returns first match or None
    return result is not None # Returns True if person exists, False otherwise

# Create a new person
@app.post("/people/", response_model=Person)
def create_person(person:PersonCreate, session: Session = Depends(get_session)):
    db_person = Person(**person.model_dump()) # .model_dump() is SQLModel's version of object.dict()
    
    if not check_person_exists(session= session, firstname=db_person.firstname, lastname=db_person.lastname):    
        session.add(db_person)
        session.commit()
        session.refresh(db_person)
        return db_person
    else:
        raise HTTPException(status_code=400, detail="Person already exists")

# Retrieve people
@app.get("/people/", response_model=list[Person])
def get_people(session: Session = Depends(get_session)):
    statement = select(Person)
    results = session.exec(statement).all()
    return results

