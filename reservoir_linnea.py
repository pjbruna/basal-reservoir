import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter 
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
    


## MODEL ##

class ResonanceNetwork:
    def __init__(self, input_nnodes=None, nnodes=None, input_connectivity=None, p_link=None, 
             leak=None, lrate_targ=None, lrate_wmat=None, targ_min=None, 
             network_type='grid', seed=None):
        self.nnodes = nnodes
        self.p_link = p_link
        self.leak = leak
        self.lrate_targ = lrate_targ
        self.lrate_wmat = lrate_wmat
        self.targ_min = targ_min
        self.network_type = network_type
        self.seed = seed
        self.weight_history = []
        self.spike_history = []
        self.act_history = []

        if input_nnodes is not None:
            self.input_nnodes = input_nnodes
            self._create_network()
            self.input_connectivity = input_connectivity
            self.input_wmat = np.random.choice([0, 5], size=(input_nnodes, self.nnodes), 
                                   p=[1 - input_connectivity, input_connectivity])
            self._initialize_weights()

            self.spikes = np.zeros(self.nnodes)
            self.targets = np.repeat(targ_min, self.nnodes)
            self.acts = np.zeros(self.nnodes)
            self.prespike_acts = np.zeros(self.nnodes)
            self._save_acts() # safe first 0 activation levels
            self._save_spikes() # save first 0 spikes

    def _create_network(self):
        if self.network_type == 'grid':
            side = int(np.round(np.sqrt(self.nnodes)))
            G = nx.grid_2d_graph(side, side)
            pos = {i: coord for i, coord in enumerate(G.nodes())}

        elif self.network_type == 'hexagonal':
            side = int(np.round(np.sqrt(self.nnodes / 2)))
            G = nx.hexagonal_lattice_graph(side, side, with_positions=True)
            pos_raw = nx.get_node_attributes(G, 'pos')
            pos = {i: pos_raw[node] for i, node in enumerate(G.nodes())}

        elif self.network_type == 'triangular':
            side = int(np.round(np.sqrt(self.nnodes / 1.5)))
            G = nx.triangular_lattice_graph(side, side, with_positions=True)
            pos_raw = nx.get_node_attributes(G, 'pos')
            pos = {i: pos_raw[node] for i, node in enumerate(G.nodes())}

        elif self.network_type == 'random':
            G = nx.erdos_renyi_graph(self.nnodes, self.p_link, seed=self.seed)
            G = nx.convert_node_labels_to_integers(G)
            pos = nx.spring_layout(G, seed=self.seed)

        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")

        G = nx.convert_node_labels_to_integers(G)
        self.G = G
        self.pos = pos
        self.nnodes = G.number_of_nodes()
        self.link_mat = nx.to_numpy_array(G, dtype=int)

    def _initialize_weights(self):
        self.wmat = np.where(self.link_mat == 1, np.random.normal(0, 1, size=(self.nnodes, self.nnodes)), 0)
        self._save_weights()

    def _save_weights(self):
        self.weight_history.append(self.wmat.copy())
   
    def _save_acts(self):
        self.act_history.append(self.acts.copy())
    
    def _save_spikes(self):
        self.spike_history.append(self.spikes.copy())


    def get_acts(self, input):
        self.acts = self.acts * self.leak + np.dot(input, self.input_wmat) + np.dot(self.spikes, self.wmat)
        self.prespike_acts = self.acts

        thresholds = self.targets * 2
        self.spikes[self.acts >= thresholds] = 1
        self.spikes[self.acts < thresholds] = 0

        self.acts[self.spikes == 1] -= thresholds[self.spikes == 1]
        self.acts[self.acts < 0] = 0

        errors = self.acts - self.targets
        return errors

    def learning(self, prev_spikes, errors):
        prev_active = np.argwhere(prev_spikes > 0)[:, 0]
        prev_inactive = np.argwhere(prev_spikes <= 0)[:, 0]

        active_neighbors = self.link_mat.copy()
        active_neighbors[prev_inactive, :] = 0
        active_neighbors = np.sum(active_neighbors, axis=0)

        d_wmat = np.zeros((self.nnodes, self.nnodes))
        d_wmat[:, :] = errors * self.lrate_wmat
        d_wmat[self.link_mat == 0] = 0
        d_wmat[prev_inactive, :] = 0

        d_wmat = np.where(active_neighbors != 0, d_wmat / active_neighbors.astype(np.float64), 0)
        self.wmat -= d_wmat

        self.targets += errors * self.lrate_targ
        self.targets[self.targets < self.targ_min] = self.targ_min

    def run(self, train_data, learn_on=True):
        log_spikes = pd.DataFrame()
        log_acts = pd.DataFrame()
        log_wmat = pd.DataFrame()

        for row in range(len(train_data)):
            prev_spikes = self.spikes.copy()
            input = train_data[row]

            errors = self.get_acts(input)

            if learn_on:
                self.learning(prev_spikes, errors)
                self._save_weights()
                self._save_acts()
                self._save_spikes()

            log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
            log_acts = pd.concat([log_acts, pd.Series(self.prespike_acts)], ignore_index=True, axis=1)
            log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

        return log_spikes, log_acts, log_wmat

    def echo(self, cue):
        end_spikes = self.spikes.copy()
        end_targets = self.targets.copy()
        end_acts = self.acts.copy()

        log_spikes = pd.DataFrame()
        log_acts = pd.DataFrame()
        log_wmat = pd.DataFrame()

        for row in range(len(cue)):
            prev_spikes = self.spikes.copy()
            input = cue[row]

            errors = self.get_acts(input)
            self.learning(prev_spikes, errors)
            self._save_weights()
            self._save_acts()
            self._save_spikes()

            log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
            log_acts = pd.concat([log_acts, pd.Series(self.prespike_acts)], ignore_index=True, axis=1)
            log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

        self.spikes = end_spikes
        self.targets = end_targets
        self.acts = end_acts

        return log_spikes, log_acts, log_wmat

    def plot_initial_network(self):
        plt.figure(figsize=(10, 7))
        edge_widths = [abs(self.wmat[u, v]) * 5 for u, v in self.G.edges()]
        nx.draw(
            self.G,
            self.pos,
            with_labels=False,
            node_color='lightgray',
            edge_color='gray',
            width=edge_widths,
            node_size=100,
            edgecolors='black'
        )
        plt.axis('equal')
        plt.title(f"Initial {self.network_type.capitalize()} Network")
        plt.show()
       
                
    def animate_weight_evolution(self, interval=300, save_path=None):
        fig, ax = plt.subplots()
        pos = self.pos
    
        norm = Normalize(vmin=0, vmax=1.5)  # You can set dynamic max or keep fixed
        cmap = cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Activation level")
    
        def update(frame):
            ax.clear()
            snapshot = self.weight_history[frame]
            activations = self.act_history[frame]
            spikes = self.spike_history[frame]
        
            # === Rescale edge weights for visibility ===
            weights = [snapshot[u, v] for u, v in self.G.edges()]
            abs_weights = np.array([abs(w) for w in weights])
            scaling_factor = 5  # tweak this until it looks good
            widths = [abs(w) * scaling_factor for w in weights]
        
            nx.draw_networkx_edges(
                self.G,
                pos,
                ax=ax,
                edge_color='gray',
                width=widths
            )
        
            # === Node color = activation ===
            node_colors = cmap(norm(activations))
            nodes = list(self.G.nodes())
            node_collection = nx.draw_networkx_nodes(
                self.G,
                pos,
                ax=ax,
                node_color=node_colors,
                node_size=100
            )
        
            # === Red outline for spikes ===
            edgecolors = ['red' if spikes[n] > 0 else 'black' for n in nodes]
            node_collection.set_edgecolor(edgecolors)
        
            ax.set_title(f"Time Step {frame}")
            ax.set_aspect('equal')
            ax.axis('off')
    
        ani = animation.FuncAnimation(fig, update, frames=len(self.weight_history), interval=interval)
    
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            if save_path.endswith(".mp4"):
                ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
            elif save_path.endswith(".gif"):
                ani.save(save_path, writer='pillow', fps=1000 // interval)
            else:
                raise ValueError("Unsupported file extension. Use .mp4 or .gif")
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
            plt.pause(0.001)

    def animate_weight_evolution2(self, interval=300, save_path=None):
    
        fig, ax = plt.subplots()
        pos = self.pos  # use consistent layout
    
        def update(frame):
            ax.clear()
            snapshot = self.weight_history[frame]
            weights = [snapshot[u, v] for u, v in self.G.edges()]
            widths = [max(abs(w), 0.8) for w in weights]  # ensure visibility
    
            nx.draw(
                self.G,
                pos,
                ax=ax,
                with_labels=False,
                width=widths,
                edge_color='gray',
                node_color='lightgray',
                node_size=100,
                edgecolors='black'
            )
            ax.set_title(f"Time Step {frame}")
            ax.set_aspect('equal')
    
        ani = animation.FuncAnimation(fig, update, frames=len(self.weight_history), interval=interval)
    
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
            # Choose writer based on file extension
            if save_path.endswith(".mp4"):
                ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
            elif save_path.endswith(".gif"):
                ani.save(save_path, writer='pillow', fps=1000 // interval)
            else:
                raise ValueError("Unsupported file extension. Use .mp4 or .gif")
    
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
            plt.pause(0.001)
    

                
## NON SPIKING RESERVOIR ##

class SlimeMoldReservoir:
  def __init__(self, input_nnodes=None, nnodes=None, input_connectivity=None, p_link=None, leak=None, lrate_targ=None, lrate_wmat=None, targ_min=None, network_type='grid', seed=None, decay_rate = None):
    # Model hyperparameters
    self.nnodes = nnodes
    self.p_link = p_link
    self.leak = leak
    self.lrate_targ = lrate_targ
    self.lrate_wmat = lrate_wmat
    self.targ_min = targ_min
    self.network_type = network_type
    self.seed = seed
    self.decay_rate = decay_rate
    self.weight_history = []
    self.act_history = []

    if input_nnodes is not None:
      self.input_nnodes = input_nnodes
      self.input_connectivity = input_connectivity

      # Create internal network using networkx
      self._create_network()

      # Create input weight matrix using final self.nnodes
      self.input_wmat = np.random.choice([0,1], size=(input_nnodes, self.nnodes), p=[1-input_connectivity, input_connectivity])

      # Initialize weights and state
      self._initialize_weights()
      self.targets = np.repeat(targ_min, self.nnodes)
      self.acts = np.zeros(self.nnodes)
      self._save_acts() # safe first 0 activation levels

  def _create_network(self):
    size = int(np.sqrt(self.nnodes))

    if self.network_type == 'grid':
      side = int(np.round(np.sqrt(self.nnodes)))
      G = nx.grid_2d_graph(size, size)
      G = nx.grid_2d_graph(side, side)
      pos = {i: coord for i, coord in enumerate(G.nodes())}

    elif self.network_type == 'hexagonal':
        side = int(np.round(np.sqrt(self.nnodes / 2)))  # adjust for denser node packing
        G = nx.hexagonal_lattice_graph(side, side, with_positions=True)
        pos_raw = nx.get_node_attributes(G, 'pos')
        pos = {i: pos_raw[node] for i, node in enumerate(G.nodes())}
      
    elif self.network_type == 'triangular':
        side = int(np.round(np.sqrt(self.nnodes / 1.5)))  # same logic
        G = nx.triangular_lattice_graph(side, side, with_positions=True)
        pos_raw = nx.get_node_attributes(G, 'pos')
        pos = {i: pos_raw[node] for i, node in enumerate(G.nodes())}
      
    elif self.network_type == 'random':
      G = nx.erdos_renyi_graph(self.nnodes, self.p_link, seed=self.seed)
      G = nx.convert_node_labels_to_integers(G)
      pos = nx.spring_layout(G, seed=self.seed) 
      
    else:
      raise ValueError(f"Unsupported network type: {self.network_type}")
        
    G = nx.convert_node_labels_to_integers(G)
    self.G = G
    self.pos = pos
    self.nnodes = G.number_of_nodes()  # Update to match actual node count
    self.link_mat = nx.to_numpy_array(G, dtype=int)

  def _initialize_weights(self):
    self.wmat = np.where(self.link_mat == 1, np.random.normal(0, 1, size=(self.nnodes, self.nnodes)), 0)
    self._save_weights()

  def _save_weights(self):
    self.weight_history.append(self.wmat.copy())
  
    
  def _save_acts(self):
    self.act_history.append(self.acts.copy())

  def learning(self, input):
    active_neighbors = (self.acts * self.leak) * self.wmat
    self.acts = np.dot(input, self.input_wmat) + np.dot(self.acts * self.leak, self.wmat)
    errors = self.acts - self.targets  #flipped errors

    d_wmat = np.zeros((self.nnodes, self.nnodes))
    d_wmat[:, :] = errors * self.lrate_wmat
    d_wmat[self.link_mat == 0] = 0

    active_neighbors = np.where(active_neighbors != 0, active_neighbors / np.sum(active_neighbors, axis=0), 0)
    d_wmat = d_wmat * active_neighbors
    self.wmat -= d_wmat

    self.targets = self.targets + (errors * self.lrate_targ)
    self.targets[self.targets < self.targ_min] = self.targ_min
    self.wmat -= self.decay_rate * self.wmat * self.link_mat

    return errors

  def run(self, train_data, learn_on=True):
    log_acts = pd.DataFrame()
    log_wmat = pd.DataFrame()
    log_errors = pd.DataFrame()

    for row in range(len(train_data)):
      input = train_data[row]
      if learn_on:
        errors = self.learning(input)
        self._save_weights() # store the weights for each time Step
        self._save_acts()

      log_acts = pd.concat([log_acts, pd.Series(self.acts)], ignore_index=True, axis=1)
      log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)
      log_errors = pd.concat([log_errors, pd.Series(errors)], ignore_index=True, axis=1)

      print(self.targets)

       #print out info about system
      print(f"\n--- Time Step {row} ---")
      print("Mean error:", np.mean(errors))
      print("Max error:", np.max(errors))
      print("Min error:", np.min(errors))
      print("Mean activation:", np.mean(self.acts))
      print("Mean target:", np.mean(self.targets))
      print("Mean weight:", np.mean(self.wmat))

    return log_acts, log_wmat, log_errors

  def echo(self, cue):
    end_targets = self.targets.copy()
    end_acts = self.acts.copy()
    end_wmat = self.wmat.copy()

    log_acts = pd.DataFrame()
    log_wmat = pd.DataFrame()
    log_errors = pd.DataFrame()

    for row in range(len(cue)):
      input = cue[row]
      errors = self.learning(input)
      self._save_weights()
      self._save_acts()

      log_acts = pd.concat([log_acts, pd.Series(self.acts)], ignore_index=True, axis=1)
      log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)
      log_errors = pd.concat([log_errors, pd.Series(errors)], ignore_index=True, axis=1)

    self.targets = end_targets
    self.acts = end_acts
    self.wmat = end_wmat

    return log_acts, log_wmat, log_errors

  def plot_initial_network(self, show_inputs=True):

    # === Start from reservoir graph ===
    if show_inputs:
        full_G = self.G.copy()
        input_ids = [f"I{i}" for i in range(self.input_nnodes)]

        # Add input nodes
        for label in input_ids:
            full_G.add_node(label)

        # Add edges from input nodes to reservoir nodes
        for i_idx, label in enumerate(input_ids):
            for j in range(self.nnodes):
                if self.input_wmat[i_idx, j] == 1:
                    full_G.add_edge(label, j)
    else:
        full_G = self.G

    # === Layout ===
    pos_reservoir = self.pos  # saved position layout

    if show_inputs:    #choose whether input nodes are shown
        min_x = min(x for x, y in pos_reservoir.values())
        x_offset = min_x - 1.5
        input_spacing = 1.0
        pos_inputs = {
            f"I{i}": (x_offset, i * input_spacing) for i in range(self.input_nnodes)
        }
        pos_full = {**pos_reservoir, **pos_inputs}
    else:
        pos_full = pos_reservoir

    # === Edge widths and colors ===
    edge_widths = []
    edge_colors = []

    for u, v in full_G.edges():
        if show_inputs and (isinstance(u, str) or isinstance(v, str)):
            edge_widths.append(1.5)
            edge_colors.append('royalblue')
        else:
            edge_widths.append(abs(self.wmat[u, v]) * 5)
            edge_colors.append('gray')

    # === Node colors and sizes ===
    node_colors = []
    node_sizes = []

    for node in full_G.nodes():
        if show_inputs and isinstance(node, str):
            node_colors.append('skyblue')
            node_sizes.append(300)
        else:
            node_colors.append('lightgray')
            node_sizes.append(500)

    # === Draw ===
    plt.figure(figsize=(10, 7))
    nx.draw(
        full_G,
        pos=pos_full,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=node_sizes,
        edgecolors='black'
    )
    plt.axis('equal')
    plt.title("Initial Network" + (" with Input Nodes" if show_inputs else " (Reservoir Only)"))
    plt.show()


  def animate_weight_and_activation(self, interval=300, save_path=None):
    fig, ax = plt.subplots()
    pos = self.pos
    norm = Normalize(vmin=0, vmax=np.max(self.act_history))
    cmap = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Activation level")

    def update(frame):
        ax.clear()
        snapshot = self.weight_history[frame]
        activations = self.act_history[frame]

        weights = [snapshot[u, v] for u, v in self.G.edges()]
        scaling_factor = 5  # tweak this until it looks good
        widths = [abs(w) * scaling_factor for w in weights]

        nx.draw_networkx_edges(
            self.G,
            pos,
            ax=ax,
            edge_color='gray',
            width=widths
        )

        node_colors = cmap(norm(activations))
        nodes = list(self.G.nodes())
        node_collection = nx.draw_networkx_nodes(
            self.G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=100
        )
        node_collection.set_edgecolor('black')

        ax.set_title(f"Time Step {frame}")
        ax.set_aspect('equal')
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(self.weight_history), interval=interval)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if save_path.endswith(".mp4"):
            ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
        elif save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow', fps=1000 // interval)
        else:
            raise ValueError("Unsupported file extension. Use .mp4 or .gif")
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
        plt.pause(0.001)


  def animate_weight_evolution(self, interval=300, save_path=None):

    fig, ax = plt.subplots()
    pos = self.pos  # use consistent layout

    def update(frame):
        ax.clear()
        snapshot = self.weight_history[frame]
        weights = [snapshot[u, v] for u, v in self.G.edges()]
        widths = [max(abs(w), 0.8) for w in weights]  # ensure visibility

        nx.draw(
            self.G,
            pos,
            ax=ax,
            with_labels=False,
            width=widths,
            edge_color='gray',
            node_color='lightgray',
            node_size=100,
            edgecolors='black'
        )
        ax.set_title(f"Time Step {frame}")
        ax.set_aspect('equal')

    ani = animation.FuncAnimation(fig, update, frames=len(self.weight_history), interval=interval)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Choose writer based on file extension
        if save_path.endswith(".mp4"):
            ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
        elif save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow', fps=1000 // interval)
        else:
            raise ValueError("Unsupported file extension. Use .mp4 or .gif")

        print(f"Animation saved to {save_path}")
    else:
        plt.show()
        plt.pause(0.001)

      
  def print_network_info(self):
    print(f"Network type: {self.network_type}")
    print(f"Number of reservoir nodes: {self.nnodes}")
    print(f"Number of input nodes: {self.input_nnodes}")
    print(f"Total nodes (input + reservoir): {self.nnodes + self.input_nnodes}")
    print(f"Number of edges in reservoir: {self.G.number_of_edges()}")

