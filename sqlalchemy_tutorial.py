from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class Person(Base):
    __tablename__ = "people"

    ssn = Column("ssn", Integer, primary_key = True)
    firstname = Column("firstname", String)
    lastname = Column("lastname", String)
    gender = Column("gender", CHAR)
    age = Column("age", Integer)


    def __init__(self, ssn, first, last, gender, age):
        self.ssn = ssn
        self.firstname = first
        self.lastname = last
        self.gender = gender
        self.age = age

    def __repr__(self):
        return f"({self.ssn}) {self.firstname} {self.lastname} ({self.gender}, {self.age})"

class Thing(Base):
    __tablename__ = "things"

    tid = Column("tid", Integer, primary_key=True)
    description = Column("description", String)
    owner = Column(Integer, ForeignKey("people.ssn"))

    def __init__(self, tid, description, owner):
        self.tid = tid
        self.description = description
        self.owner = owner

    def __repr__(self):
        return f"({self.tid}) {self.description} owned by {self.owner}"

engine = create_engine("sqlite:///mydb.db", echo = True)
Base.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

person = Person(12312, "Mike", "Smith", "m", 35)
session.add(person) 
session.commit() # apply changes to the database

p1 = Person(12363, "Anna", "Blue", "f", 30)
p2 = Person(12314, "Bob", "Blue", "m", 45)
p3 = Person(15256, "Angela", "Cold", "f", 22)

session.add(p1)
session.add(p2)
session.add(p3)
session.commit()

t1 = Thing(1, "Car", p1.ssn)
session.add(t1)
session.commit()

# results = session.query(Person).filter(Person.age >25)
# for r in results:
#     print(r)