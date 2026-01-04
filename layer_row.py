import tkinter as tk
from tkinter import ttk

class LayerRow:
    def __init__(self, parent, remove_callback):
        self.frame = ttk.Frame(parent)
        # Type of layer
        self.type_var = tk.StringVar(value="Layer")
        self.type_menu = ttk.OptionMenu(
            self.frame,
            self.type_var,
            "Layer",
            "Layer",
            "Conv1D",
            "Conv2D",
            "Dense",
            "Dropout",
            "Maxpool1D",
            "Maxpool2D",
            "Input",
            "LocallyConnected",
            "Reshape",
            command=self._on_type_change
        )
        self.type_menu.pack(side="left", padx=4)

        # ---- Parameter widgets (created once; shown/hidden as needed) ----
        # Neurons / filters / units
        self.neurons_var = tk.IntVar(value=32)
        self.neurons_spin = ttk.Spinbox(self.frame, from_=1, to=65536, textvariable=self.neurons_var, width=6)
        self.neurons_label = ttk.Label(self.frame, text="neurons:")

        # Activation
        self.activation_var = tk.StringVar(value="Activation")
        self.activation_menu = ttk.OptionMenu(self.frame, self.activation_var, "Activation", "linear", "relu", "sigmoid", "softmax", "tanh")
        self.activation_label = ttk.Label(self.frame, text="act:")

        # Conv kernel dims (conv2d)
        self.kernel_width_var = tk.IntVar(value=3)
        self.kernel_width_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.kernel_width_var, width=4)
        self.kernel_width_label = ttk.Label(self.frame, text="kW:")
        self.kernel_height_var = tk.IntVar(value=3)
        self.kernel_height_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.kernel_height_var, width=4)
        self.kernel_height_label = ttk.Label(self.frame, text="kH:")

        # Conv1D / LocallyConnected kernel_size
        self.kernel_size_var = tk.IntVar(value=3)
        self.kernel_size_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.kernel_size_var, width=4)
        self.kernel_size_label = ttk.Label(self.frame, text="kSize:")

        # Dropout
        self.dropout_var = tk.DoubleVar(value=0.5)
        self.dropout_entry = ttk.Entry(self.frame, textvariable=self.dropout_var, width=6)
        self.dropout_label = ttk.Label(self.frame, text="rate:")

        # Reshape (comma-separated dims)
        self.reshape_var = tk.StringVar(value="")
        self.reshape_entry = ttk.Entry(self.frame, textvariable=self.reshape_var, width=12)
        self.reshape_label = ttk.Label(self.frame, text="shape:")

        # Pooling
        self.pool_size_var = tk.IntVar(value=2)
        self.pool_size_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.pool_size_var, width=4)
        self.pool_size_label = ttk.Label(self.frame, text="pool:")
        self.pool_width_var = tk.IntVar(value=2)
        self.pool_width_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.pool_width_var, width=4)
        self.pool_width_label = ttk.Label(self.frame, text="pW:")
        self.pool_height_var = tk.IntVar(value=2)
        self.pool_height_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.pool_height_var, width=4)
        self.pool_height_label = ttk.Label(self.frame, text="pH:")
        self.stride_var = tk.IntVar(value=1)
        self.stride_spin = ttk.Spinbox(self.frame, from_=1, to=32, textvariable=self.stride_var, width=4)
        self.stride_label = ttk.Label(self.frame, text="stride:")

        # Remove button
        self.remove_btn = ttk.Button(self.frame, text="Remove", command=lambda: remove_callback(self))
        self.remove_btn.pack(side="left", padx=4)

        # initialize visibility
        self._current_type = None
        self._hide_all_params()

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def destroy(self):
        self.frame.destroy()

    def _on_type_change(self, _=None):
        # OptionMenu passes the selection as argument; ignore and call update
        self.update_fields()

    def _hide_all_params(self):
        # Hide all optional widgets (include labels)
        widgets = [
            self.neurons_label, self.neurons_spin,
            self.activation_label, self.activation_menu,
            self.kernel_width_label, self.kernel_width_spin,
            self.kernel_height_label, self.kernel_height_spin,
            self.kernel_size_label, self.kernel_size_spin,
            self.dropout_label, self.dropout_entry,
            self.reshape_label, self.reshape_entry,
            self.pool_size_label, self.pool_size_spin,
            self.pool_width_label, self.pool_width_spin,
            self.pool_height_label, self.pool_height_spin,
            self.stride_label, self.stride_spin
        ]
        for w in widgets:
            try:
                w.pack_forget()
            except Exception:
                pass

    def update_fields(self):
        # show only the widgets relevant to the selected layer type
        layer_type = self.type_var.get()
        if layer_type == self._current_type:
            return
        self._current_type = layer_type

        self._hide_all_params()

        # pack parameters in a sensible left-to-right order
        if layer_type == "Dense":
            # number of neurons, activation
            self.neurons_label.pack(side="left", padx=2)
            self.neurons_spin.pack(side="left", padx=4)
            self.activation_label.pack(side="left", padx=2)
            self.activation_menu.pack(side="left", padx=4)
        elif layer_type == "Conv2D":
            # filters, kernel_width, kernel_height, activation
            self.neurons_label.pack(side="left", padx=2)  # filters
            self.neurons_spin.pack(side="left", padx=4)
            self.kernel_width_label.pack(side="left", padx=1)
            self.kernel_width_spin.pack(side="left", padx=2)
            self.kernel_height_label.pack(side="left", padx=1)
            self.kernel_height_spin.pack(side="left", padx=2)
            self.activation_label.pack(side="left", padx=2)
            self.activation_menu.pack(side="left", padx=4)
        elif layer_type == "Conv1D":
            # filters, kernel_size, activation
            self.neurons_label.pack(side="left", padx=2)  # filters
            self.neurons_spin.pack(side="left", padx=4)
            self.kernel_size_label.pack(side="left", padx=2)
            self.kernel_size_spin.pack(side="left", padx=4)
            self.activation_label.pack(side="left", padx=2)
            self.activation_menu.pack(side="left", padx=4)
        elif layer_type == "Input":
            # number of neurons only
            self.neurons_label.pack(side="left", padx=2)
            self.neurons_spin.pack(side="left", padx=4)
        elif layer_type == "Dropout":
            # rate between 0 and 1
            self.dropout_label.pack(side="left", padx=2)
            self.dropout_entry.pack(side="left", padx=4)
        elif layer_type == "Reshape":
            # dimensions string, e.g. "28,28"
            self.reshape_label.pack(side="left", padx=2)
            self.reshape_entry.pack(side="left", padx=4)
        elif layer_type == "LocallyConnected":
            # like Conv1D: filters, kernel_size, activation
            self.neurons_label.pack(side="left", padx=2)
            self.neurons_spin.pack(side="left", padx=4)
            self.kernel_size_label.pack(side="left", padx=2)
            self.kernel_size_spin.pack(side="left", padx=4)
            self.activation_label.pack(side="left", padx=2)
            self.activation_menu.pack(side="left", padx=4)
        elif layer_type == "Maxpool1D":
            # pool_size, stride
            self.pool_size_label.pack(side="left", padx=2)
            self.pool_size_spin.pack(side="left", padx=4)
            self.stride_label.pack(side="left", padx=2)
            self.stride_spin.pack(side="left", padx=4)
        elif layer_type == "Maxpool2D":
            # pool_width, pool_height, stride
            self.pool_width_label.pack(side="left", padx=1)
            self.pool_width_spin.pack(side="left", padx=2)
            self.pool_height_label.pack(side="left", padx=1)
            self.pool_height_spin.pack(side="left", padx=2)
            self.stride_label.pack(side="left", padx=2)
            self.stride_spin.pack(side="left", padx=4)
        else:
            # unknown or placeholder -> show nothing
            pass

    def get_config(self):
        layer_type = self.type_var.get()
        cfg = {"type": layer_type}
        if layer_type == "Dense":
            cfg.update({"neurons": int(self.neurons_var.get()), "activation": self.activation_var.get()})
        elif layer_type == "Conv2D":
            cfg.update({
                "filters": int(self.neurons_var.get()),
                "kernel_width": int(self.kernel_width_var.get()),
                "kernel_height": int(self.kernel_height_var.get()),
                "activation": self.activation_var.get()
            })
        elif layer_type == "Conv1D":
            cfg.update({
                "filters": int(self.neurons_var.get()),
                "kernel_size": int(self.kernel_size_var.get()),
                "activation": self.activation_var.get()
            })
        elif layer_type == "Input":
            cfg.update({"neurons": int(self.neurons_var.get())})
        elif layer_type == "Dropout":
            # ensure float between 0 and 1; caller should validate further
            cfg.update({"rate": float(self.dropout_var.get())})
        elif layer_type == "Reshape":
            cfg.update({"shape": self.reshape_var.get()})
        elif layer_type == "LocallyConnected":
            cfg.update({
                "filters": int(self.neurons_var.get()),
                "kernel_size": int(self.kernel_size_var.get()),
                "activation": self.activation_var.get()
            })
        elif layer_type == "Maxpool1D":
            cfg.update({
                "pool_size": int(self.pool_size_var.get()),
                "stride": int(self.stride_var.get())
            })
        elif layer_type == "Maxpool2D":
            cfg.update({
                "pool_width": int(self.pool_width_var.get()),
                "pool_height": int(self.pool_height_var.get()),
                "stride": int(self.stride_var.get())
            })
        # For unknown layer types, return only the type
        return cfg