# Project Proposal: Autonomous Delivery Rover

**Student Name:** [Your Name]  
**Course:** CEN352 - Artificial Intelligence  

---

## 1. Problem Definition
The "last-mile" delivery problem involves navigating complex, non-linear environments where traditional road maps are unavailable. Our objective is to design an autonomous rover capable of traversing diverse terrains (forests, pastures, residential areas) while avoiding hazards (water bodies). The agent must make rational decisions to balance path energy-efficiency with safety, ensuring the lowest-cost traversal across discretized satellite maps.

## 2. Proposed AI Techniques
This project integrates two core pillars of Artificial Intelligence to achieve autonomous navigation:

### 2.1 Computer Vision: Convolutional Neural Networks (CNN)
To perceive the environment, we utilize a **CNN** trained on the **EuroSAT dataset**. 
- **Function:** The model classifies 64x64 pixel satellite image tiles into 10 distinct terrain categories.
- **Integration:** These classifications are used to dynamically generate a **Cost Grid**, where each terrain type is assigned a numerical traversal weight (e.g., Highway = 1, Forest = 15, Water = $\infty$).

### 2.2 Intelligent Search: A* Algorithm
Once the cost map is generated, the **A* Search Algorithm** is employed for global path planning.
- **Function:** A* finds the mathematically optimal path from the starting point to the destination.
- **Optimality:** By using an admissible **Euclidean distance heuristic**, the algorithm guarantees the most energy-efficient route while successfully circumventing all impassable terrain identified by the CNN.

## 3. Expected Outcomes
The final system will demonstrate a complete AI pipeline: transforming raw visual data into a cost-aware navigation strategy. Performance will be measured by the classification accuracy of the CNN and the optimality/safety of the paths produced by the A* search.
