**1. Data Loading & Preprocessing (`NBADataset`)**
* It loads tensor files (`.pt`) containing sequences of size `[Time, Entities, Features]`.
* The features include `(x, y)` coordinates, `isplayer` (boolean), and `team` ID.
* It splits the sequences into **Context** (the past 8 frames the model can see) and **Horizon** (the future 12 frames the model must predict).
* It normalizes the spatial coordinates `(x, y)` to have a mean of 0 and a standard deviation of 1, which helps the neural network learn faster and more stably.

**2. Graph Construction (`build_hetero_edges`)**
* Since basketball is highly interactive, the model uses a graph to represent the scene. Entities (10 players + 1 ball) are **nodes**.
* It dynamically generates **heterogeneous edges** (different types of relationships) between nodes based on team IDs. 
* There are 5 relations: Teammate to Teammate (0), Opponent to Opponent (1), Player to Ball (2), Ball to Player (3), and Self-loops (4). 

**3. Spatio-Temporal Graph Neural Network (`NBAModel_HeteroSTGCN` & `HeteroSTGCNCell`)**
* **Spatial Component:** It uses a Relational Graph Attention Network (`RGATConv`). This allows the model to aggregate information from neighboring nodes differently depending on their relationship (e.g., players pay more attention to teammates moving towards the basket, or to the ball).
* **Temporal Component:** It uses a Recurrent Neural Network (`GRUCell`). The GRU maintains a hidden state for each node over time, updating it as new frames are processed.
* **Autoregressive Prediction:** For the context phase (frames 0 to 7), the model simply absorbs the known data to build up its hidden state. For the horizon phase (frames 8 to 19), it predicts the *delta* (the change in position), adds it to the last known position, and feeds its own prediction back into itself to predict the next step.

**4. Training & Kaggle Submission (`NBATrainer`)**
* Trains the model using Mean Squared Error (`MultiStepMSE`) over the 12-step horizon.
* Includes a function to automatically format the predictions back to real-world coordinates and export them to a `solution.csv` file for the EPFL Kaggle competition.

---
