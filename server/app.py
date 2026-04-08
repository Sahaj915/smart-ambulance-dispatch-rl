from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Server is running"}

def main():
    return app

if __name__ == "__main__":
    main()