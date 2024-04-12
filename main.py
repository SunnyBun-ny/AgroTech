import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



@app.get("/")
def index():
    return {'message' : 'Hello World'}

@app.post("/greetings")
def greetings(number):
    match number:
        case 1:
            return {'greetings' : 'Have a good day!!'}
        case 2: 
            return {'greetings' : 'Good Morning!!'}
        case _:
            return {'greetings' : 'Happy Cristmas!!'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)

    
#python -m uvicorn main:app --reload
# web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app