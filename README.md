
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

### 1. Install Python

- Make sure you have **Python 3.9+** installed.\
If not, download it here: [Python
Downloads](https://www.python.org/downloads/)

- You can check your computer have python , open terminate 
``` bash 
python3 --version
```
<b><i>Or</b></i>
``` bash 
python --version
```

If show <i>Python 3.13.7<i> , your computer have python else access to link and dowload it.

### 2. Dowload project 

#### Git 
``` bash
git clone https://github.com/baokhanh546123/API_LungCancer.git
```

### Github
- Access <a href = 'https://github.com/baokhanh546123/API_LungCancer'>Link</a>
- Click to code and dowload zip 
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

- For develop
``` bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- For run
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
