# General Laboratory Interface for Design, Experimentation, and Recording

####  Documentation Coming Soon...

### Glider Does Not Work On Python 3.14+ Currently

## Prerequisites

1. Python 3.11, 3.12, or 3.13

## Getting Started

### Windows
1. Create a Virtual Environment
```python -m venv venv```
2. Activate Virtual Environment
```venv/scripts/Activate```
3. Install Dependencies
```pip install ".[pc]"```
4. Launch GLIDER
```glider```
### Linux
1. Create a Virtual Environment
```python -m venv venv```
2. Activate Virtual Environment
```source venv/bin/activate```
3. Install Dependencies
```pip install ".[pc]"```
4. Launch GLIDER
```glider```
### MacOS
1. Create a Virtual Environment
```python -m venv venv```
2. Activate Virtual Environment
```source venv/bin/activate```
3. Install Dependencies
```pip install ".[pc]"```
4. Launch GLIDER
```glider```
### Raspberry Pi
1. Install PyQt6 Through apt
```sudo apt install python3-pyqt6```
1. Create a Virtual Environment with System Package
```python -m venv --system-site-packages venv```
2. Activate Virtual Environment
```source venv/bin/activate```
3. Install Dependencies
```pip install -e .```
4. Launch GLIDER
```glider```

# To Dos
1. Switch to uv package manager