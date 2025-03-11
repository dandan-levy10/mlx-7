from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel

app = FastAPI()


# students = {
#     1: {
#         "name": "John",
#         "age": 20,
#         "city": "New York"
#     },
#     2: {
#         "name": "Jane",
#         "age": 21,
#         "city": "Los Angeles"}
# }



class Student(BaseModel):
    name: str
    age: int
    city: str

class UpdateStudent(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None

students = {
    1: Student(name="John", age=20, city="New York"),
    2: Student(name="Jane", age=21, city="Los Angeles")
    }

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.get("/get-student/{student_id}")
def get_student(student_id: int = Path(..., description="The ID of the student you want to view", gt=0)):
    if student_id not in students:
        return {"Error": "Student ID does not exist"}
    return students[student_id]

@app.get("/get-by-name/{student_id}")
def get_student_by_name(*, student_id: int, name: Optional[str] = None):
    for id in students:
        if students[id]["name"] == name:
            return students[id]
    return {"Error": "Student name not found"}

# Create a new student using POST method and Student class
@app.post("/create-student/{student_id}")
def create_student(student_id: int, student: Student):
    if student_id in students:
        return {"Error": "Student ID already exists"}
    students[student_id] = student
    return students[student_id]

# Update a student using PUT method and Student class
@app.put("/update-student/{student_id}")
def update_student(student_id: int, student: UpdateStudent):
    if student_id not in students:
        return {"Error": "Student ID does not exist"}
    
    if student.name != None:
        students[student_id].name = student.name

    if student.age != None:
        students[student_id].age = student.age

    if student.city != None:
        students[student_id].city = student.city

    return students[student_id]

# Delete a student using DELETE method
@app.delete("/delete-student/{student_id}")
def delete_student(student_id: int):
    if student_id not in students: 
        return {"Error": "Student ID does not exist"}
    del students[student_id]
    return students