# CausalBox: A Causal Inference Toolkit 🔬

## Table of Contents

1. [About CausalBox](#1-about-causalbox)
2. [Features](#2-features)
3. [How It Works (End-to-End Overview)](#3-how-it-works-end-to-end-overview)
   - [Core Components](#core-components)
   - [Data Flow](#data-flow)
4. [Setup & Installation](#4-setup--installation)
   - [Prerequisites](#prerequisites)
   - [Clone the Repository](#clone-the-repository)
   - [Install Dependencies](#install-dependencies)
   - [Run the Backend (FastAPI)](#run-the-backend-fastapi)
   - [Run the Frontend (Flask)](#run-the-frontend-flask)
5. [Usage Guide](#5-usage-guide)
6. [Project Structure](#6-project-structure)
7. [Key Technologies Used](#7-key-technologies-used)
8. [Future Enhancements](#8-future-enhancements)
9. [Contributing](#9-contributing)
10. [License](#10-license)

---

## 1. About CausalBox

**CausalBox** is a production-ready, modular, and interactive causal inference toolkit that provides:

- Causal graph discovery from tabular data
- Simulated interventions (`do`-calculus)
- Average Treatment Effect (ATE) estimation
- Real-time visualization and editing of causal graphs
- A powerful API backend and user-friendly frontend

This project is designed for data scientists, researchers, and AI engineers to explore causal structures in data using algorithms like PC, FCI, and LiNGAM, while enabling dynamic editing and analysis via a web interface.

## 2. Features

- 📊 **Causal Graph Discovery**: Use PC, FCI, and LiNGAM to generate graphs from observational data.
- ⚛️ **Simulated Interventions**: Perform `do(X=x)` and observe changes in dependent variables.
- 📊 **ATE Estimation**: Calculate Average Treatment Effect between treatment and outcome variables.
- 📈 **Interactive Graph Editing**: Add, remove, or change edges/nodes dynamically and rerun inference.
- 🌐 **Web-based Interface**: Built with Flask + Cytoscape.js for visualization; FastAPI backend for heavy computation.
- 📱 **Export Graphs**: Save causal graphs as PNG or SVG.

## 3. How It Works (End-to-End Overview)

### Core Components

#### Frontend (Flask + Cytoscape.js)

- HTML/JS/CSS with a Flask server
- Renders interactive graphs with Cytoscape.js
- Sends JSON to backend for computation

#### Backend (FastAPI + Uvicorn)

- FastAPI handles API routes:
  - `/discover` - run causal graph discovery
  - `/intervene` - simulate interventions
  - `/ate` - compute average treatment effects
- Uses CausalLearn for algorithms (PC, FCI, LiNGAM)
- Uses NetworkX for graph structure

#### Core Python Modules

- `causal_engine.py`: Heart of the logic; handles preprocessing, discovery, intervention, ATE
- `config.py`: Centralized config for parameters and constants
- `routes.py`: API endpoints for FastAPI
- `flask_app.py`: Frontend Flask server (renders `index.html`)
- `index.html`: UI with dropdowns, buttons, and embedded Cytoscape.js
- `main.py`: Entrypoint that runs both backend (via Uvicorn) and optionally Flask

### Data Flow

```
User Upload CSV → Flask UI → FastAPI API Call → Preprocessing & Discovery → Graph JSON → Flask UI Render
```

1. User uploads CSV via Flask frontend
2. Flask sends data to FastAPI via `/discover`
3. Backend processes, runs PC/FCI/LiNGAM, returns causal graph
4. Graph rendered using Cytoscape.js
5. User can edit graph (add/remove edge)
6. User runs intervention or ATE
7. Backend recalculates and sends results

## 4. Setup & Installation

### Prerequisites

- Python 3.9 - 3.11 (recommended)

### Clone the Repository

```bash
git clone https://github.com/yourusername/CausalBox.git
cd CausalBox
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

If `causallearn` fails, try:

```bash
pip install git+https://github.com/py-why/causal-learn.git
```

### Run the Backend (FastAPI)

```bash
uvicorn fastapi_app:app --reload
```

### Run the Frontend (Flask)

```bash
python flask_app.py
```

## 5. Usage Guide

1. Open `http://localhost:5000` in your browser
2. Upload a CSV dataset
3. Select a causal discovery algorithm (PC, FCI, LiNGAM)
4. View the causal graph
5. Click on nodes/edges to edit graph
6. Simulate an intervention using dropdown
7. Estimate ATE between two variables
8. Export graph to PNG or SVG

## 6. Project Structure

```
causalbox/
├── app/
│   ├── causal_engine.py         # Core causal inference logic
│   ├── config.py                # Configuration settings
│   ├── routes.py                # FastAPI endpoints
│   └── templates/
│       └── index.html           # Frontend UI
├── fastapi_app.py               # FastAPI server
├── flask_app.py                 # Flask frontend server
├── main.py                      # Entrypoint to run backend/frontend
├── requirements.txt             # All dependencies
├── README.md                    # This file
```

## 7. Key Technologies Used

- **Backend**:

  - FastAPI
  - Uvicorn
  - Pandas, NumPy
  - NetworkX
  - CausalLearn (PC, FCI, LiNGAM)
  - Scikit-learn

- **Frontend**:

  - Flask
  - Cytoscape.js
  - HTML/CSS/JS

## 8. Future Enhancements

- Add counterfactual estimations
- Integrate Granger causality for time-series
- Docker support
- Causal model comparison metrics
- Live-editable causal weight inference

## 9. Contributing

We welcome contributions! Feel free to submit issues or pull requests.

## 10. License

This project is licensed under the MIT License.

---

Built with ❤️ by the CausalBox Team.

