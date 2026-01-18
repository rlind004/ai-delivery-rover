import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import heapq
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Delivery Rover", layout="wide")

st.title("🛰️ AI Autonomous Delivery Rover")
st.markdown("""
This application uses a **Convolutional Neural Network (CNN)** to classify terrain from satellite imagery 
and the **A* Search Algorithm** to find the optimal path for a delivery rover.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

# 1. Model Selection
st.sidebar.subheader("1. AI Model")
model_source = st.sidebar.radio("Select Model Source", ["Upload .pth file", "Use Default (eurosat_cnn.pth)"])

model_file = None
if model_source == "Upload .pth file":
    model_file = st.sidebar.file_uploader("Upload trained model (eurosat_cnn.pth)", type=["pth"])
else:
    if os.path.exists("eurosat_cnn.pth"):
        model_file = "eurosat_cnn.pth"
    else:
        st.sidebar.warning("Default model 'eurosat_cnn.pth' not found in directory.")

# 2. Map Selection
st.sidebar.subheader("2. Map Image")
map_source = st.sidebar.radio("Select Map Source", ["Upload Image", "Use Sample (map.png)"])

map_file = None
if map_source == "Upload Image":
    map_file = st.sidebar.file_uploader("Upload Satellite Map", type=["png", "jpg", "jpeg"])
else:
    if os.path.exists("map.png"):
        map_file = "map.png"
    else:
        st.sidebar.warning("Default map 'map.png' not found.")

# 3. Pathfinding Parameters
st.sidebar.subheader("3. Navigation")
st.sidebar.markdown("**Start Position**")
start_x = st.sidebar.number_input("Start X", min_value=0, value=2, key="sx")
start_y = st.sidebar.number_input("Start Y", min_value=0, value=2, key="sy")

st.sidebar.markdown("**Goal Position**")
end_x = st.sidebar.number_input("Goal X", min_value=0, value=0, key="ex")
end_y = st.sidebar.number_input("Goal Y", min_value=0, value=0, key="ey")

run_btn = st.sidebar.button("🚀 Launch Rover")

# 4. Evaluation
st.sidebar.subheader("4. Model Evaluation")
eval_btn = st.sidebar.button("📊 Evaluate Model")

# --- MODEL DEFINITION ---
class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SatelliteCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
           'River', 'SeaLake']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(path_or_file):
    if path_or_file is None:
        return None
    
    try:
        model = SatelliteCNN(num_classes=10).to(device)
        
        # Handle both file path and uploaded file object
        if isinstance(path_or_file, str):
            checkpoint = torch.load(path_or_file, map_location=device)
        else:
            checkpoint = torch.load(path_or_file, map_location=device)
            
        sd = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- PHYSICS & CLEANING ---
def get_surface_cost(terrain_type):
    if terrain_type in ['River', 'SeaLake']: return 999
    elif terrain_type == 'Highway': return 0.1
    elif terrain_type in ['Industrial', 'Residential']: return 35
    elif terrain_type in ['Pasture', 'AnnualCrop', 'HerbaceousVegetation']: return 20
    elif terrain_type == 'Forest': return 100
    else: return 50

def clean_cost_grid(costs):
    rows, cols = costs.shape
    cleaned = costs.copy()
    corrections = 0
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if costs[r, c] >= 500:
                river_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        if costs[r+dr, c+dc] >= 500:
                            river_neighbors += 1
                if river_neighbors < 4:
                    cleaned[r, c] = 20
                    corrections += 1
    return cleaned

def scan_terrain(model, image_file, scan_step=16):
    image = Image.open(image_file).convert('RGB')
    w, h = image.size
    cols, rows = w // scan_step, h // scan_step

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cost_grid = np.zeros((rows, cols))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with torch.no_grad():
        for r in range(rows):
            # Update progress every row
            progress_bar.progress(int((r / rows) * 100))
            status_text.text(f"Scanning row {r+1}/{rows}...")
            
            for c in range(cols):
                cy, cx = r * scan_step + scan_step//2, c * scan_step + scan_step//2
                crop_size = 32
                left = max(0, cx - crop_size)
                top = max(0, cy - crop_size)
                right = min(w, cx + crop_size)
                bottom = min(h, cy + crop_size)

                tile = image.crop((left, top, right, bottom))
                input_t = transform(tile).unsqueeze(0).to(device)

                outputs = model(input_t)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top2_prob, top2_idx = torch.topk(probs, 2)

                best_class = CLASSES[top2_idx[0][0].item()]
                best_conf = top2_prob[0][0].item()
                second_class = CLASSES[top2_idx[0][1].item()]

                if best_class in ['River', 'SeaLake'] and best_conf < 0.90:
                    terrain = second_class
                else:
                    terrain = best_class

                cost_grid[r, c] = get_surface_cost(terrain)
    
    progress_bar.empty()
    status_text.empty()
    final_costs = clean_cost_grid(cost_grid)
    return image, final_costs, rows, cols

def a_star(costs, start, goal):
    rows, cols = costs.shape
    start = (min(start[0], rows-1), min(start[1], cols-1))
    goal = (min(goal[0], rows-1), min(goal[1], cols-1))

    costs[start] = 1
    costs[goal] = 1

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal: break

        for dx, dy, dist in [(-1,0,1), (1,0,1), (0,-1,1), (0,1,1),
                             (-1,-1,1.4), (-1,1,1.4), (1,-1,1.4), (1,1,1.4)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                surface_cost = costs[nx, ny]
                if surface_cost >= 500: continue
                new_cost = cost_so_far[current] + (surface_cost * dist)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + (abs(goal[0]-nx) + abs(goal[1]-ny))
                    heapq.heappush(frontier, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    if goal not in came_from: return None
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    return path

# --- EVALUATION LOGIC ---
def evaluate_model(model):
    st.info("Downloading EuroSAT dataset for evaluation (this may take a moment)...")
    data_path = './eurosat_data'
    
    # Use same transform as training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Download/Load
        dataset = datasets.EuroSAT(root=data_path, download=True, transform=transform)
        
        # Split (Same seed/logic as training ideally, but random split is okay for generic eval)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_data = random_split(dataset, [train_size, val_size])
        
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        y_true = []
        y_pred = []
        
        progress_bar = st.progress(0)
        status = st.empty()
        total_batches = len(val_loader)
        
        status.text("Running inference on validation set...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                progress_bar.progress(min((i + 1) / total_batches, 1.0))
                
        progress_bar.empty()
        status.empty()
        
        st.success(f"Evaluation Complete! Tested on {len(y_true)} images.")
        
        # 1. Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
        
        # 2. Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        
        # 3. F1-Score Bar Chart
        f1_scores = [report[cls]['f1-score'] for cls in CLASSES]
        fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=f1_scores, y=CLASSES, palette="viridis", ax=ax_f1)
        ax_f1.set_xlabel("F1 Score")
        ax_f1.set_title("Per-Class Performance (F1 Score)")
        ax_f1.set_xlim(0, 1.0)
        
        return fig_cm, fig_f1, acc, report
        
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return None, None, 0, None

# --- MAIN LOGIC ---
if run_btn:
    if not model_file:
        st.error("Please select or upload a model file.")
    elif not map_file:
        st.error("Please select or upload a map image.")
    else:
        model = load_model(model_file)
        if model:
            image, costs, rows, cols = scan_terrain(model, map_file)
            
            st.success("Map Scanned Successfully!")
            
            # Determine Coordinates
            start_coord = (int(start_y), int(start_x))
            
            if end_x == 0 and end_y == 0:
                goal_coord = (rows-1, cols-1)
                st.toast(f"Goal defaulting to bottom-right: {goal_coord}")
            else:
                goal_coord = (int(end_y), int(end_x))
            
            if not (0 <= goal_coord[0] < rows and 0 <= goal_coord[1] < cols):
                st.error(f"Goal coordinate {goal_coord} is out of bounds (Max: {rows-1}, {cols-1}).")
                path = None
            else:
                path = a_star(costs, start_coord, goal_coord)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Rover Navigation")
                fig1, ax1 = plt.subplots(figsize=(10, 10))
                ax1.imshow(image)
                ax1.axis('off')
                
                if path:
                    # Convert grid coords to pixel coords
                    SCAN_STEP = 16
                    ys = [r * SCAN_STEP + SCAN_STEP//2 for r, c in path]
                    xs = [c * SCAN_STEP + SCAN_STEP//2 for r, c in path]
                    ax1.plot(xs, ys, color='yellow', linewidth=3)
                    ax1.scatter(xs[0], ys[0], c='green', s=150, label="Start")
                    ax1.scatter(xs[-1], ys[-1], c='red', s=150, label="Goal")
                    ax1.legend()
                
                st.pyplot(fig1)

            with col2:
                st.subheader("Cost Map (Terrain Analysis)")
                fig2, ax2 = plt.subplots(figsize=(10, 10))
                heatmap = ax2.imshow(costs, cmap='jet', interpolation='nearest')
                plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
                ax2.axis('off')
                
                if path:
                    path_ys = [r for r, c in path]
                    path_xs = [c for r, c in path]
                    ax2.plot(path_xs, path_ys, color='white', linewidth=2, linestyle='--')
                
                st.pyplot(fig2)
                
            if path:
                st.info(f"Path found with length: {len(path)} steps.")
            else:
                st.warning("No path found to the destination.")

if eval_btn:
    if not model_file:
         st.error("Please select or upload a model file (Section 1) first.")
    else:
        model = load_model(model_file)
        if model:
            st.markdown("---")
            st.header("📊 Model Evaluation & Metrics")
            
            fig_cm, fig_f1, acc, report = evaluate_model(model)
            
            if fig_cm:
                # Display Top-Level Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Accuracy", f"{acc*100:.2f}%")
                c2.metric("Total Test Images", "5400 (Approx)") # 20% of EuroSAT
                c3.metric("Model Status", "Ready")
                
                st.subheader("1. Where is the model making mistakes?")
                st.pyplot(fig_cm)
                
                st.subheader("2. Class Performance (F1-Score)")
                st.write("F1-Score attempts to find the balance between Precision (not categorizing blindly) and Recall (not missing real items).")
                st.pyplot(fig_f1)
                
                st.subheader("3. Detailed Classification Report")
                # Convert report to handy dataframe
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.highlight_max(axis=0))
