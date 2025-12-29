
# Lung Cancer Detection Web Application

This project is a **web-based system** for detecting lung cancer using
chest X-ray images.\
It integrates a modern web stack with lightweight animations to provide
a **clinician-friendly interface**.

------------------------------------------------------------------------

## Features
-   Upload and process lung X-ray images
-   Interactive UI with smooth animations (GSAP)
-   Backend powered by **FastAPI**
-   Data stored in **PostgreSQL**   
-   Frontend built with **HTML & CSS**
-   Public web temporary **ngrok**
------------------------------------------------------------------------

## Tools & Libraries
-   **FastAPI and JS** (Python backend)
-   **HTML5 & CSS** (frontend design)
-   **PostgreSQL** (database)
-   **GSAP** (animation library)
-   **Uvicorn** (server runner)
-   **Tensorflow && Pytorch** (Deep Learning library)

------------------------------------------------------------------------

## Installation & Setup

### 1.  Prerequisites

Ensure the following are installed on your system:
- Python 3.9 or higher (if not it , you can dowload [Here](https://www.python.org/downloads/))
- PostgreSQL 
- Git (That would be great.)


Verify Python installation:

#### Linux / macOS

``` bash 
python3 --version
```
<b><i>Or</b></i>

#### Windows

``` bash 
python --version
```

If show <i>Python 3.xx.x<i> , your computer have python else access to link and dowload it.

- You can dowload PostgreSQL , you can access to ChatGPT and entry to prompt
``` prompt
    You are a PostgreSQL expert, please provide detailed instructions on how to install PostgreSQL on [your platform] and create a schema named CNN with a default port of 5432. The schema should return a text or table format, presented in an easy-to-understand manner.
```
<h3>Please specify your platform as Windows, Linux, or macOS.</h3>

### 2. Dowload project 

#### Git 
``` bash
git clone https://github.com/baokhanh546123/API_LungCancer.git
```

### Github
- Access <a href = 'https://github.com/baokhanh546123/API_LungCancer'>Link</a>
- Click to code green button and dowload zip 
- Unzip


### 3. Create Virtual Environment & Install Dependencies

#### Windows

``` bash
cd API_LungCancer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS

``` bash
cd API_LungCancer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Application
- Before you run the project, you have to change in .env
#### Linux / macOS
``` bash
nano connect/connection.env 
```
- Change username and password approriate then press <strong> <u> Ctrl + O , Enter and Ctrl + X  </u> </strong>
### Windows
``` bash
notepad connect/connection.env
```
- Change username and password approriate then press <strong> <u> Ctrl + S </u> </strong>


### Develop Mode
``` bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production
#### Linux / macOS
``` bash 
    python3 main.py
```
#### Windows
``` bash 
    python main.py
```

- The app will be available at **http://127.0.0.1:8000/**



------------------------------------------------------------------------

## Feedback

If you try this project, please share your feedback, suggestions, or
improvements.\
Your input helps make the system better!

--------------------------------------------

## Author
Developed with using FastAPI, PostgreSQL, and GSAP animations.
