
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

- You need dowload <a href = 'https://ngrok.com/'>ngrok</a> which allow everyone can access to web.

### Dowload ngrok

#### Windows 
You can access <a href = 'https://dashboard.ngrok.com/get-started/setup/windows'>Link</a>

####  macOS
``` bash
    brew install ngrok
    ngrok config add-authtoken yourtoken
```

#### Linux
``` bash
    snap install ngrok
    ngrok config add-authtoken yourtoken
```

- Other case , you can install via docker or RaspberryPi for embedding device , ... you can access to <a href = 'https://dashboard.ngrok.com'>ngrok</a>
- After , run ngrok by command below
```
    ngrok htpp 8000
```
### 2. Dowload model 
- You can dowload model <a href = 'https://drive.google.com/drive/u/1/folders/127wtoC5b6TeUsBNkpuhIZEiIO0KeHbrc'>Here</a>

### 3. Create Virtual Environment & Install Dependencies

#### Windows

``` bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS

``` bash
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
