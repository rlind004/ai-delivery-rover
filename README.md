# AI Term Project: Autonomous Delivery Rover

This project implements an autonomous delivery rover that uses a **Convolutional Neural Network (CNN)** for terrain classification and the **A* Search Algorithm** for optimal pathfinding based on satellite imagery.

## 🚀 Setup & Installation

To run this project on a local machine (VS Code, Jupyter, or PyCharm), please follow these steps:

### 1. Install Dependencies
Ensure you have Python installed. Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
```

### 2. Map Image
Place your satellite map image in the project root directory and name it `map.png`. The rover's navigation logic will automatically scan this file.

### 3. Execution
Open `Arlind_Disha.ipynb` and run the cells in order:
1.  **Section 1:** Trains the CNN on the EuroSAT dataset (Automated download - Internet required for first run).
2.  **Section 2:** Evaluates the model with a Confusion Matrix.
3.  **Section 3:** Scans `map.png` and calculates the A* path.

---

## 🛠 Project Components
- **Arlind_Disha.ipynb:** Core logic (CNN + A*).
- **project_report.md:** Detailed technical report (PEAS, algorithm analysis, ethics).
- **requirements.txt:** List of required Python libraries.

## 📘 Note to Professor
*   **Internet Connection:** Required only for the first run to download the ~89MB EuroSAT dataset.
*   **Performance:** Code is optimized for GPU if available, but will run on CPU (approx. 1.5 - 2 mins per epoch on standard laptops).
