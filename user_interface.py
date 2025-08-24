import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, scrolledtext
import json
import subprocess
import threading
import os
import tempfile
import time
from typing import List, Dict
from fortran_generator import FortranNeuralGenerator

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Designer - Advanced")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.layers = []
        self.layer_types = {
            'linear': {'neurons': True, 'activation': True},
            'conv1d': {'filters': True, 'kernel_size': True, 'stride': True, 'padding': True, 'activation': True},
            'conv2d': {'filters': True, 'kernel_size': True, 'stride': True, 'padding': True, 'activation': True},
            'self_attention': {'d_model': True, 'num_heads': True, 'dropout': True},
            'cross_attention': {'d_model': True, 'num_heads': True, 'dropout': True},
            'multihead_attention': {'d_model': True, 'num_heads': True, 'dropout': True},
            'dropout': {'dropout_rate': True},
            'embedding': {'vocab_size': True, 'embed_dim': True},
            'flatten': {},
            'maxpool1d': {'pool_size': True, 'stride': True},
            'maxpool2d': {'pool_size': True, 'stride': True}
        }
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel - Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Network Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Training Parameters
        train_frame = ttk.LabelFrame(config_frame, text="Training Parameters", padding="5")
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(train_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(train_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.01")
        ttk.Entry(train_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(train_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W)
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(train_frame, textvariable=self.batch_var, width=10).grid(row=2, column=1, padx=5)
        
        # Layer Management
        layer_frame = ttk.LabelFrame(config_frame, text="Layer Management", padding="5")
        layer_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Layer type selection
        type_frame = ttk.Frame(layer_frame)
        type_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(type_frame, text="Layer Type:").grid(row=0, column=0, sticky=tk.W)
        self.layer_type_var = tk.StringVar(value="linear")
        type_combo = ttk.Combobox(type_frame, textvariable=self.layer_type_var, 
                                values=list(self.layer_types.keys()), width=15, state="readonly")
        type_combo.grid(row=0, column=1, padx=5)
        type_combo.bind('<<ComboboxSelected>>', self.on_layer_type_change)
        
        # Dynamic parameter frame
        self.param_frame = ttk.LabelFrame(layer_frame, text="Layer Parameters", padding="5")
        self.param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Initialize parameter widgets
        self.param_vars = {}
        self.param_widgets = {}
        self.setup_parameter_widgets()
        
        # Add layer button
        ttk.Button(layer_frame, text="Add Layer", command=self.add_layer).grid(row=2, column=0, pady=10)
        
        # Layer list
        self.layer_listbox = tk.Listbox(layer_frame, height=8)
        self.layer_listbox.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Layer control buttons
        button_frame = ttk.Frame(layer_frame)
        button_frame.grid(row=4, column=0, pady=5)
        
        ttk.Button(button_frame, text="Remove Layer", command=self.remove_layer).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Move Up", command=self.move_layer_up).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Move Down", command=self.move_layer_down).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Edit Layer", command=self.edit_layer).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Visualization and Export
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Network Visualization
        viz_frame = ttk.LabelFrame(right_frame, text="Network Architecture", padding="5")
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.canvas = tk.Canvas(viz_frame, bg='white', height=300)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(viz_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configuration Preview
        preview_frame = ttk.LabelFrame(right_frame, text="Configuration Preview", padding="5")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.preview_text = tk.Text(preview_frame, height=15, wrap=tk.WORD)
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.preview_text.configure(yscrollcommand=preview_scroll.set)
        
        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(control_frame, text="Generate Fortran Config", 
                  command=self.generate_fortran_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Add Run button with prominent style
        run_button = ttk.Button(control_frame, text="▶ Run Fortran Model", 
                              command=self.run_fortran_model, style="Run.TButton")
        run_button.pack(side=tk.LEFT, padx=10)
        
        # Output console for Fortran execution
        output_frame = ttk.LabelFrame(right_frame, text="Fortran Execution Output", padding="5")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.output_console = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD,
                                                     background="black", foreground="green")
        self.output_console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.output_console.insert(tk.END, "Ready to run neural-fortran models...\n")
        self.output_console.config(state=tk.DISABLED)  # Read-only initially
        
        # Configure grid weights for resizing with the new output console
        right_frame.rowconfigure(3, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Create a special style for the run button
        style = ttk.Style()
        style.configure("Run.TButton", font=('Arial', 10, 'bold'))
        
        # Initialize with a sample network
        self.add_sample_network()
        
    def setup_parameter_widgets(self):
        # Clear existing widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        self.param_vars.clear()
        self.param_widgets.clear()
        
        layer_type = self.layer_type_var.get()
        params = self.layer_types.get(layer_type, {})
        
        row = 0
        
        # Common parameters based on layer type
        if 'neurons' in params:
            ttk.Label(self.param_frame, text="Neurons:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['neurons'] = tk.StringVar(value="64")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['neurons'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'filters' in params:
            ttk.Label(self.param_frame, text="Filters:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['filters'] = tk.StringVar(value="32")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['filters'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'kernel_size' in params:
            ttk.Label(self.param_frame, text="Kernel Size:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['kernel_size'] = tk.StringVar(value="3")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['kernel_size'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'stride' in params:
            ttk.Label(self.param_frame, text="Stride:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['stride'] = tk.StringVar(value="1")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['stride'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'padding' in params:
            ttk.Label(self.param_frame, text="Padding:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['padding'] = tk.StringVar(value="same")
            padding_combo = ttk.Combobox(self.param_frame, textvariable=self.param_vars['padding'], 
                                       values=["same", "valid"], width=8, state="readonly")
            padding_combo.grid(row=row, column=1, padx=5)
            row += 1
            
        if 'd_model' in params:
            ttk.Label(self.param_frame, text="Model Dim:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['d_model'] = tk.StringVar(value="512")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['d_model'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'num_heads' in params:
            ttk.Label(self.param_frame, text="Num Heads:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['num_heads'] = tk.StringVar(value="8")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['num_heads'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'dropout' in params or 'dropout_rate' in params:
            ttk.Label(self.param_frame, text="Dropout Rate:").grid(row=row, column=0, sticky=tk.W)
            param_name = 'dropout_rate' if 'dropout_rate' in params else 'dropout'
            self.param_vars[param_name] = tk.StringVar(value="0.1")
            ttk.Entry(self.param_frame, textvariable=self.param_vars[param_name], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'vocab_size' in params:
            ttk.Label(self.param_frame, text="Vocab Size:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['vocab_size'] = tk.StringVar(value="10000")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['vocab_size'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'embed_dim' in params:
            ttk.Label(self.param_frame, text="Embed Dim:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['embed_dim'] = tk.StringVar(value="128")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['embed_dim'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'pool_size' in params:
            ttk.Label(self.param_frame, text="Pool Size:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['pool_size'] = tk.StringVar(value="2")
            ttk.Entry(self.param_frame, textvariable=self.param_vars['pool_size'], width=10).grid(row=row, column=1, padx=5)
            row += 1
            
        if 'activation' in params:
            ttk.Label(self.param_frame, text="Activation:").grid(row=row, column=0, sticky=tk.W)
            self.param_vars['activation'] = tk.StringVar(value="relu")
            activation_combo = ttk.Combobox(self.param_frame, textvariable=self.param_vars['activation'], 
                                          values=["relu", "sigmoid", "tanh", "linear", "gelu", "swish"], width=8, state="readonly")
            activation_combo.grid(row=row, column=1, padx=5)
            row += 1
            
    def on_layer_type_change(self, event=None):
        self.setup_parameter_widgets()
        
    def add_layer(self):
        try:
            layer_type = self.layer_type_var.get()
            layer = {'type': layer_type}
            
            # Collect all parameters for this layer type
            for param_name, param_var in self.param_vars.items():
                value = param_var.get()
                if param_name in ['neurons', 'filters', 'kernel_size', 'stride', 'd_model', 'num_heads', 'vocab_size', 'embed_dim', 'pool_size']:
                    try:
                        layer[param_name] = int(value)
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid value for {param_name}: {value}")
                        return
                elif param_name in ['dropout', 'dropout_rate']:
                    try:
                        layer[param_name] = float(value)
                        if not (0 <= layer[param_name] <= 1):
                            messagebox.showerror("Error", f"Dropout rate must be between 0 and 1")
                            return
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid dropout rate: {value}")
                        return
                else:
                    layer[param_name] = value
            
            self.layers.append(layer)
            self.update_layer_list()
            self.update_visualization()
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add layer: {e}")
            
    def remove_layer(self):
        selection = self.layer_listbox.curselection()
        if selection:
            index = selection[0]
            del self.layers[index]
            self.update_layer_list()
            self.update_visualization()
            self.update_preview()
            
    def move_layer_up(self):
        selection = self.layer_listbox.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            self.layers[index], self.layers[index-1] = self.layers[index-1], self.layers[index]
            self.update_layer_list()
            self.layer_listbox.selection_set(index-1)
            self.update_visualization()
            self.update_preview()
            
    def move_layer_down(self):
        selection = self.layer_listbox.curselection()
        if selection and selection[0] < len(self.layers) - 1:
            index = selection[0]
            self.layers[index], self.layers[index+1] = self.layers[index+1], self.layers[index]
            self.update_layer_list()
            self.layer_listbox.selection_set(index+1)
            self.update_visualization()
            self.update_preview()
            
    def update_layer_list(self):
        self.layer_listbox.delete(0, tk.END)
        for i, layer in enumerate(self.layers):
            layer_desc = f"Layer {i+1}: {layer['type']}"
            
            # Add key parameters to description
            if layer['type'] == 'linear' and 'neurons' in layer:
                layer_desc += f" ({layer['neurons']} neurons)"
            elif layer['type'] in ['conv1d', 'conv2d'] and 'filters' in layer:
                layer_desc += f" ({layer['filters']} filters, k={layer.get('kernel_size', '?')})"
            elif 'attention' in layer['type'] and 'd_model' in layer:
                layer_desc += f" (d={layer['d_model']}, heads={layer.get('num_heads', '?')})"
            elif layer['type'] == 'dropout' and 'dropout_rate' in layer:
                layer_desc += f" (rate={layer['dropout_rate']})"
            elif layer['type'] == 'embedding' and 'vocab_size' in layer:
                layer_desc += f" (vocab={layer['vocab_size']}, dim={layer.get('embed_dim', '?')})"
            elif layer['type'] in ['maxpool1d', 'maxpool2d'] and 'pool_size' in layer:
                layer_desc += f" (pool={layer['pool_size']})"
                
            self.layer_listbox.insert(tk.END, layer_desc)
            
    def edit_layer(self):
        selection = self.layer_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a layer to edit")
            return
            
        index = selection[0]
        layer = self.layers[index]
        
        # Set the layer type and update parameters
        self.layer_type_var.set(layer['type'])
        self.setup_parameter_widgets()
        
        # Fill in the current values
        for param_name, param_var in self.param_vars.items():
            if param_name in layer:
                param_var.set(str(layer[param_name]))
                
        # Remove the old layer so user can add the edited version
        del self.layers[index]
        self.update_layer_list()
        self.update_visualization()
        self.update_preview()
        
    def update_visualization(self):
        self.canvas.delete("all")
        if not self.layers:
            return
            
        canvas_width = 500
        canvas_height = max(400, len(self.layers) * 80 + 100)
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        layer_height = 60
        layer_spacing = 80
        start_y = 50
        
        # Color scheme for different layer types - using valid Tkinter colors
        colors = {
            'linear': '#ADD8E6',        # light blue
            'conv1d': '#90EE90',        # light green
            'conv2d': '#90EE90',        # light green
            'self_attention': '#FFFFE0', # light yellow
            'cross_attention': '#FFFFE0', # light yellow
            'multihead_attention': '#FFFFE0', # light yellow
            'dropout': '#F08080',       # light coral
            'embedding': '#FFB6C1',     # light pink
            'flatten': '#D3D3D3',       # light gray
            'maxpool1d': '#E0FFFF',     # light cyan
            'maxpool2d': '#E0FFFF'      # light cyan
        }
        
        for i, layer in enumerate(self.layers):
            y = start_y + i * layer_spacing
            color = colors.get(layer['type'], '#lightsteelblue')
            
            # Draw layer box
            self.canvas.create_rectangle(50, y, 450, y + layer_height, 
                                       fill=color, outline='darkblue', width=2)
            
            # Layer label
            self.canvas.create_text(250, y + 15, 
                                  text=f"Layer {i+1}: {layer['type'].upper()}", 
                                  font=('Arial', 11, 'bold'))
            
            # Layer details
            details = []
            if 'neurons' in layer:
                details.append(f"Neurons: {layer['neurons']}")
            if 'filters' in layer:
                details.append(f"Filters: {layer['filters']}")
            if 'd_model' in layer:
                details.append(f"D-Model: {layer['d_model']}")
            if 'dropout_rate' in layer:
                details.append(f"Dropout: {layer['dropout_rate']}")
            if 'activation' in layer:
                details.append(f"Act: {layer['activation']}")
                
            detail_text = " | ".join(details[:3])  # Limit to 3 details for space
            self.canvas.create_text(250, y + 35, 
                                  text=detail_text, 
                                  font=('Arial', 9))
            
            # Additional details on next line
            if len(details) > 3:
                extra_details = " | ".join(details[3:])
                self.canvas.create_text(250, y + 50, 
                                      text=extra_details, 
                                      font=('Arial', 8))
            
            # Draw connections to next layer
            if i < len(self.layers) - 1:
                self.canvas.create_line(250, y + layer_height, 250, y + layer_spacing, 
                                      arrow=tk.LAST, width=3, fill='darkgreen')
                                      
    def update_preview(self):
        config = self.get_config()
        
        # Generate readable configuration text
        preview = "=== Neural Network Configuration ===\n\n"
        preview += f"Training Parameters:\n"
        preview += f"  Epochs: {config['epochs']}\n"
        preview += f"  Learning Rate: {config['learning_rate']}\n"
        preview += f"  Batch Size: {config['batch_size']}\n\n"
        
        preview += f"Network Architecture:\n"
        preview += f"  Total Layers: {len(config['layers'])}\n\n"
        
        for i, layer in enumerate(config['layers']):
            preview += f"  Layer {i+1} ({layer['type']}):\n"
            
            # Display all layer parameters dynamically
            for key, value in layer.items():
                if key != 'type':
                    preview += f"    {key.replace('_', ' ').title()}: {value}\n"
            preview += "\n"
            
        # Add some statistics
        linear_layers = [layer for layer in config['layers'] if layer['type'] == 'linear']
        if linear_layers:
            total_neurons = sum(layer.get('neurons', 0) for layer in linear_layers)
            preview += f"Statistics:\n"
            preview += f"  Linear Layer Neurons: {total_neurons}\n"
            
            if len(linear_layers) > 1:
                total_weights = sum(linear_layers[i]['neurons'] * linear_layers[i+1]['neurons'] 
                                  for i in range(len(linear_layers)-1))
                preview += f"  Approximate Linear Weights: {total_weights}\n"
        
        # Count layer types
        layer_counts = {}
        for layer in config['layers']:
            layer_type = layer['type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            
        preview += f"\nLayer Type Distribution:\n"
        for layer_type, count in layer_counts.items():
            preview += f"  {layer_type.replace('_', ' ').title()}: {count}\n"
        
        self.preview_text.delete('1.0', tk.END)
        self.preview_text.insert('1.0', preview)
        
    def get_config(self):
        return {
            'epochs': int(self.epochs_var.get()),
            'learning_rate': float(self.lr_var.get()),
            'batch_size': int(self.batch_var.get()),
            'layers': self.layers
        }
        
    def generate_fortran_config(self):
        config = self.get_config()
        
        # Show dialog to choose between config file or complete program
        choice = messagebox.askyesnocancel("Fortran Output", 
                                         "Choose output type:\n\n" +
                                         "Yes: Generate complete Fortran program\n" +
                                         "No: Generate configuration file only\n" +
                                         "Cancel: Cancel operation")
        
        if choice is None:  # Cancel
            return
        elif choice:  # Yes - Generate complete program
            self.generate_complete_fortran_program(config)
        else:  # No - Generate config only
            self.generate_fortran_config_only(config)
    
    def generate_complete_fortran_program(self, config):
        """Generate a complete runnable Fortran program"""
        generator = FortranNeuralGenerator()
        
        # Ask for program name
        program_name = tk.simpledialog.askstring("Program Name", 
                                                "Enter program name:", 
                                                initialvalue="my_neural_network")
        if not program_name:
            return
            
        # Generate the complete program
        fortran_code = generator.generate_complete_program(config, program_name)
        
        # Save to file
        filename = filedialog.asksaveasfilename(
            defaultextension=".f90",
            filetypes=[("Fortran files", "*.f90"), ("All files", "*.*")],
            title="Save Fortran Program",
            initialvalue=f"{program_name}.f90"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(fortran_code)
            messagebox.showinfo("Success", 
                              f"Complete Fortran program saved to {filename}\n\n" +
                              "To compile and run:\n" +
                              f"gfortran -o {program_name} {filename} -lneural-fortran\n" +
                              f"./{program_name}")
    
    def generate_fortran_config_only(self, config):
        """Generate configuration file only (original functionality)"""
        fortran_code = "! Generated Neural Network Configuration\n"
        fortran_code += "! Advanced Layer Types Supported\n\n"
        
        fortran_code += f"INTEGER, PARAMETER :: NUM_LAYERS = {len(config['layers'])}\n"
        fortran_code += f"INTEGER, PARAMETER :: EPOCHS = {config['epochs']}\n"
        fortran_code += f"REAL, PARAMETER :: LEARNING_RATE = {config['learning_rate']}\n"
        fortran_code += f"INTEGER, PARAMETER :: BATCH_SIZE = {config['batch_size']}\n\n"
        
        # Layer type enumeration
        fortran_code += "! Layer types: 1=Linear, 2=Conv1D, 3=Conv2D, 4=SelfAttn, 5=CrossAttn\n"
        fortran_code += "!             6=MultiAttn, 7=Dropout, 8=Embedding, 9=Flatten, 10=MaxPool1D, 11=MaxPool2D\n"
        
        type_map = {
            'linear': '1', 'conv1d': '2', 'conv2d': '3', 'self_attention': '4',
            'cross_attention': '5', 'multihead_attention': '6', 'dropout': '7',
            'embedding': '8', 'flatten': '9', 'maxpool1d': '10', 'maxpool2d': '11'
        }
        
        layer_types = [type_map.get(layer['type'], '1') for layer in config['layers']]
        fortran_code += f"INTEGER :: LAYER_TYPES(NUM_LAYERS) = (/ {', '.join(layer_types)} /)\n\n"
        
        # Generate parameter arrays for each layer type
        if any(layer['type'] == 'linear' for layer in config['layers']):
            neurons = [str(layer.get('neurons', 0)) if layer['type'] == 'linear' else '0' 
                      for layer in config['layers']]
            fortran_code += f"INTEGER :: LINEAR_NEURONS(NUM_LAYERS) = (/ {', '.join(neurons)} /)\n"
            
        if any('conv' in layer['type'] for layer in config['layers']):
            filters = [str(layer.get('filters', 0)) if 'conv' in layer['type'] else '0' 
                      for layer in config['layers']]
            kernels = [str(layer.get('kernel_size', 0)) if 'conv' in layer['type'] else '0' 
                      for layer in config['layers']]
            fortran_code += f"INTEGER :: CONV_FILTERS(NUM_LAYERS) = (/ {', '.join(filters)} /)\n"
            fortran_code += f"INTEGER :: CONV_KERNELS(NUM_LAYERS) = (/ {', '.join(kernels)} /)\n"
            
        if any('attention' in layer['type'] for layer in config['layers']):
            d_models = [str(layer.get('d_model', 0)) if 'attention' in layer['type'] else '0' 
                       for layer in config['layers']]
            num_heads = [str(layer.get('num_heads', 0)) if 'attention' in layer['type'] else '0' 
                        for layer in config['layers']]
            fortran_code += f"INTEGER :: ATTN_D_MODEL(NUM_LAYERS) = (/ {', '.join(d_models)} /)\n"
            fortran_code += f"INTEGER :: ATTN_HEADS(NUM_LAYERS) = (/ {', '.join(num_heads)} /)\n"
        
        # Save to file
        filename = filedialog.asksaveasfilename(
            defaultextension=".f90",
            filetypes=[("Fortran files", "*.f90"), ("All files", "*.*")],
            title="Save Fortran Configuration"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(fortran_code)
            messagebox.showinfo("Success", f"Fortran configuration saved to {filename}")
            
    def save_config(self):
        config = self.get_config()
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", f"Configuration saved to {filename}")
            
    def load_config(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Update UI with loaded config
                self.epochs_var.set(str(config['epochs']))
                self.lr_var.set(str(config['learning_rate']))
                self.batch_var.set(str(config['batch_size']))
                self.layers = config['layers']
                
                self.update_layer_list()
                self.update_visualization()
                self.update_preview()
                
                messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
                
    def clear_all(self):
        if messagebox.askyesno("Confirm", "Clear all layers and reset to defaults?"):
            self.layers.clear()
            self.epochs_var.set("100")
            self.lr_var.set("0.01")
            self.batch_var.set("32")
            
            self.update_layer_list()
            self.update_visualization()
            self.update_preview()
            
    def add_sample_network(self):
        # Add a more complex sample network with various layer types
        sample_layers = [
            {'type': 'embedding', 'vocab_size': 10000, 'embed_dim': 128},
            {'type': 'conv1d', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 'same', 'activation': 'relu'},
            {'type': 'maxpool1d', 'pool_size': 2, 'stride': 2},
            {'type': 'self_attention', 'd_model': 128, 'num_heads': 8, 'dropout': 0.1},
            {'type': 'dropout', 'dropout_rate': 0.2},
            {'type': 'flatten'},
            {'type': 'linear', 'neurons': 64, 'activation': 'relu'},
            {'type': 'linear', 'neurons': 10, 'activation': 'sigmoid'}
        ]
        
        self.layers = sample_layers
        self.update_layer_list()
        self.update_visualization()
        self.update_preview()
        
    def run_fortran_model(self):
        """Run the current neural network model using neural-fortran"""
        # Get current configuration
        config = self.get_config()
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        program_name = "nn_model"
        program_path = os.path.join(temp_dir, f"{program_name}.f90")
        executable_path = os.path.join(temp_dir, program_name)
        
        # Find neural-fortran path
        neural_fortran_path = self._get_neural_fortran_path()
        if not neural_fortran_path:
            return
        
        # Generate Fortran code
        try:
            generator = FortranNeuralGenerator()
            fortran_code = generator.generate_complete_program(config, program_name)
            
            # Write to temporary file
            with open(program_path, 'w') as f:
                f.write(fortran_code)
                
            self._log_output(f"Created Fortran program: {program_path}\n")
            self._log_output("Compiling with neural-fortran...\n")
            
            # Compile the code using local neural-fortran
            compile_command = self._get_compile_command(executable_path, program_path, neural_fortran_path)
            
            self._log_output(f"Using compile command: {compile_command}\n")
            
            # Run compilation in a separate thread to avoid freezing the GUI
            threading.Thread(target=self._compile_and_run, 
                           args=(compile_command, executable_path, temp_dir)).start()
            
        except Exception as e:
            self._log_output(f"Error: {e}\n", error=True)
            messagebox.showerror("Error", f"Failed to generate or compile Fortran code: {e}")
    
    def _get_neural_fortran_path(self):
        """Get the path to neural-fortran library"""
        # Check for cached path
        if hasattr(self, 'neural_fortran_path') and os.path.exists(self.neural_fortran_path):
            return self.neural_fortran_path
            
        # Try local directory first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, "neural-fortran")
        
        if os.path.exists(local_path):
            self.neural_fortran_path = local_path
            return local_path
            
        # Ask the user for the path
        messagebox.showinfo("Neural-Fortran Path", 
                          "Please select the neural-fortran directory")
        nf_dir = filedialog.askdirectory(
            title="Select neural-fortran directory",
            initialdir=current_dir
        )
        
        if nf_dir and os.path.exists(nf_dir):
            self.neural_fortran_path = nf_dir
            return nf_dir
        
        messagebox.showerror("Error", "Could not find neural-fortran. Please clone it first.")
        return None
    
    def _get_compile_command(self, executable_path, program_path, nf_path):
        """Generate the compilation command based on neural-fortran location"""
        # Check if we have a local build directory
        build_dir = os.path.join(nf_path, "build")
        mod_dir = os.path.join(build_dir, "mod")
        lib_dir = os.path.join(build_dir, "lib")
        
        if os.path.exists(build_dir):
            # Use local compilation if build directory exists
            includes = f"-I{mod_dir}"
            if os.path.exists(lib_dir):
                libs = f"-L{lib_dir} -lneural-fortran"
                return f"gfortran {includes} -o {executable_path} {program_path} {libs}"
            
        # Try simpler local compilation
        local_src = os.path.join(nf_path, "src")
        if os.path.exists(local_src):
            return (f"gfortran -I{local_src} -o {executable_path} {program_path} "
                   f"{os.path.join(local_src, '*.f90')}")
            
        # Fall back to system installation
        return f"gfortran -o {executable_path} {program_path} -lneural-fortran"
    
    def _check_neural_fortran(self):
        """Skip check - we'll handle paths manually"""
        return True

    def _compile_and_run(self, compile_command, executable_path, temp_dir):
        """Compile and run the Fortran code, streaming output to the console"""
        try:
            # Compile the code
            compile_process = subprocess.Popen(
                compile_command, shell=True, stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, universal_newlines=True
            )
            
            # Stream compilation output
            while True:
                output = compile_process.stdout.readline()
                if output == '' and compile_process.poll() is not None:
                    break
                if output:
                    self._log_output(output.strip() + "\n")
                    self.root.update_idletasks()  # Update GUI
            
            compile_process.wait()
            
            # Check if compilation was successful
            if compile_process.returncode != 0:
                self._log_output("Compilation failed!\n", error=True)
                return
                
            self._log_output("Compilation successful!\n")
            self._log_output("Running neural network model...\n")
            
            # Run the executable
            run_process = subprocess.Popen(
                executable_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, cwd=temp_dir
            )
            
            # Stream run output
            while True:
                output = run_process.stdout.readline()
                if output == '' and run_process.poll() is not None:
                    break
                if output:
                    self._log_output(output.strip() + "\n")
                    self.root.update_idletasks()  # Update GUI
            
            run_process.wait()
            
            if run_process.returncode == 0:
                self._log_output("Neural network execution completed successfully!\n")
            else:
                self._log_output("Neural network execution failed!\n", error=True)
                
        except Exception as e:
            self._log_output(f"Error: {e}\n", error=True)
    
    def _log_output(self, message, error=False):
        """Log message to the output console"""
        self.output_console.config(state=tk.NORMAL)  # Make writable
        
        # Set colors based on message type
        if error:
            self.output_console.tag_config("error", foreground="red")
            self.output_console.insert(tk.END, message, "error")
        else:
            self.output_console.insert(tk.END, message)
            
        self.output_console.see(tk.END)  # Scroll to end
        self.output_console.config(state=tk.DISABLED)  # Make read-only again

def main():
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    if error:
        self.output_console.tag_config("error", foreground="red")
        self.output_console.insert(tk.END, message, "error")
    else:
        self.output_console.insert(tk.END, message)
            
    self.output_console.see(tk.END)  # Scroll to end
    self.output_console.config(state=tk.DISABLED)  # Make read-only again

def main():
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
