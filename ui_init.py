import os
import tkinter as tk
from tkinter import ttk


class UIInit:
    """Base UI initializer class. NNBuilderApp subclasses this to get the
    UI layout and initial widgets.
    """
    def __init__(self, root):
        self.root = root
        root.title("NN Builder")

        # Top config frame
        cfg_frame = ttk.LabelFrame(root, text="Training Configuration")
        cfg_frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(cfg_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.epochs_var = tk.IntVar(value=10)
        self.epochs_spin = ttk.Spinbox(cfg_frame, from_=1, to=10000, textvariable=self.epochs_var, width=8)
        self.epochs_spin.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(cfg_frame, text="Learning rate:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.lr_var = tk.StringVar(value="0.001")
        self.lr_entry = ttk.Entry(cfg_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(cfg_frame, text="Optimizer:").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        self.opt_var = tk.StringVar(value="adam")
        self.opt_menu = ttk.OptionMenu(cfg_frame, self.opt_var, "adam", "adam", "sgd", "rmsprop")
        self.opt_menu.grid(row=0, column=5, padx=4, pady=4)

        # Fix: use Spinbox for batch size (OptionMenu was used incorrectly)
        ttk.Label(cfg_frame, text="Batch size:").grid(row=0, column=6, sticky="w", padx=4, pady=4)
        self.batch_size_var = tk.IntVar(value=10)
        self.batch_size_spin = ttk.Spinbox(cfg_frame, from_=1, to=1000000, textvariable=self.batch_size_var, width=10)
        self.batch_size_spin.grid(row=0, column=7, padx=4, pady=4)

        # Dataset selection frame
        dataset_frame = ttk.Frame(root)
        dataset_frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(dataset_frame, text="Dataset:").pack(side="left", padx=4)
        # presets list
        # include "Custom..." so other code can select it without mismatch
        self.dataset_presets = ["None", "MNIST", "CIFAR-10", "CIFAR-100", "Custom..."]
        self.dataset_preset_var = tk.StringVar(value=self.dataset_presets[0])
        self.dataset_preset_menu = ttk.OptionMenu(dataset_frame, self.dataset_preset_var, self.dataset_presets[0], *self.dataset_presets, command=self._on_dataset_preset_change)
        self.dataset_preset_menu.pack(side="left", padx=4)

        self.dataset_var = tk.StringVar(value="")
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_var, width=40)
        self.dataset_entry.pack(side="left", padx=4, expand=True, fill="x")
        self.dataset_browse_btn = ttk.Button(dataset_frame, text="Browse...", command=self.browse_dataset)
        self.dataset_browse_btn.pack(side="left", padx=4)
        # (do not initialize preset state yet; wait until log_text exists)

        # Layers frame
        layers_frame = ttk.LabelFrame(root, text="Layers")
        layers_frame.pack(fill="both", expand=False, padx=8, pady=6)
        self.layers_container = ttk.Frame(layers_frame)
        self.layers_container.pack(fill="x", padx=4, pady=4)

        # Controls to add layer
        add_btn = ttk.Button(layers_frame, text="Add Layer", command=self.add_layer)
        add_btn.pack(side="left", padx=8, pady=6)
        clear_btn = ttk.Button(layers_frame, text="Clear Layers", command=self.clear_layers)
        clear_btn.pack(side="left", padx=8, pady=6)
        save_btn = ttk.Button(layers_frame, text="Save Config", command=self.save_config)
        save_btn.pack(side="left", padx=8, pady=6)
        load_btn = ttk.Button(layers_frame, text="Load Config", command=self.load_config)
        load_btn.pack(side="left", padx=8, pady=6)

        # Status & progress
        status_frame = ttk.Frame(root)
        status_frame.pack(fill="x", padx=8, pady=6)
        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(side="left", padx=4, pady=4)
        self.status_label = ttk.Label(status_frame, text="Idle")
        self.status_label.pack(side="left", padx=8)

        # Action buttons
        action_frame = ttk.Frame(root)
        action_frame.pack(fill="x", padx=8, pady=6)
        self.play_btn = ttk.Button(action_frame, text="Play (Train)", command=self.on_play)
        self.play_btn.pack(side="left", padx=4)
        self.quit_btn = ttk.Button(action_frame, text="Quit", command=root.quit)
        self.quit_btn.pack(side="left", padx=4)

        # Log
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        # initialize entry/browse state now that log_text exists
        self._on_dataset_preset_change(self.dataset_preset_var.get())

        # internal
        self.layer_rows = []
        # start with one layer
        self.add_layer()
        # persistent configs file
        self.config_file = os.path.join(os.path.dirname(__file__), "configs.txt")
        # store saved configs as name -> dict (persisted to self.config_file)
        self.saved_configs = {}
        self._load_saved_configs()

