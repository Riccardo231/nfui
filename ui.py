import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import simpledialog, filedialog
from fortran_mnist import execute_fortran_mnist
import threading
import time
import os
import subprocess
import json
from layer_row import LayerRow
from ui_init import UIInit

class NNBuilderApp(UIInit):
    def __init__(self, root):
        super().__init__(root)
        # do not reinitialize self.layer_rows here (UIInit already set up initial row)
        # only override config file + saved configs
        self.config_file = "nn_builder_configs.txt"
        self.saved_configs = {}
        self._load_saved_configs()

    def add_layer(self):
        row = LayerRow(self.layers_container, remove_callback=self.remove_layer)
        row.pack(fill="x", pady=2)
        # ensure the row reflects its current type selection UI
        row.update_fields()
        self.layer_rows.append(row)

    def remove_layer(self, row):
        row.destroy()
        self.layer_rows.remove(row)

    def clear_layers(self):
        for r in list(self.layer_rows):
            r.destroy()
        self.layer_rows.clear()
    
    def save_config(self):
        # validate current configuration first
        ok, err = self.validate_config()
        if not ok:
            messagebox.showerror("Invalid configuration", err)
            return

        # ask user for a name
        name = simpledialog.askstring("Save Config", "Enter a name for this configuration:")
        if not name:
            # user cancelled or empty name
            self.log("Save cancelled (no name provided).")
            return

        # build config dict (dataset stored as preset + optional path)
        config = {
            "epochs": int(self.epochs_var.get()),
            "learning_rate": float(self.lr_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "optimizer": self.opt_var.get(),
            "dataset": {"preset": self.dataset_preset_var.get(), "path": self.dataset_var.get()},
            "layers": [r.get_config() for r in self.layer_rows]
        }

        # store in-memory and persist to disk
        try:
            self.saved_configs[name] = config
            self._save_configs_to_file()
            self.log(f"Config '{name}' saved.")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save config: {e}")
            self.log(f"Error saving config '{name}': {e}")

    def load_config(self):
        if not self.saved_configs:
            messagebox.showinfo("Load Config", "No saved configurations available.")
            return

        # ask user to select a config
        name = simpledialog.askstring("Load Config", f"Available configs: {', '.join(self.saved_configs.keys())}\nEnter the name of the configuration to load:")
        if not name or name not in self.saved_configs:
            self.log("Load cancelled or config not found.")
            return

        # load and parse config
        try:
            stored = self.saved_configs[name]
            if isinstance(stored, str):
                config = json.loads(stored)
            else:
                config = stored

            # apply to UI
            self.epochs_var.set(config.get("epochs", self.epochs_var.get()))
            self.lr_var.set(str(config.get("learning_rate", self.lr_var.get())))
            self.opt_var.set(config.get("optimizer", self.opt_var.get()))
            # batch_size is an IntVar -> set with int
            self.batch_size_var.set(int(config.get("batch_size", self.batch_size_var.get())))

            # dataset (handle dict {preset,path} or legacy string)
            ds = config.get("dataset", "")
            if isinstance(ds, dict):
                preset = ds.get("preset", "None")
                path = ds.get("path", "")
            else:
                if ds in self.dataset_presets:
                    preset = ds
                    path = "" if ds != "Custom..." else ""
                else:
                    preset = "Custom..."
                    path = ds
            if preset not in self.dataset_presets:
                preset = "Custom..."
            self.dataset_preset_var.set(preset)
            self.dataset_var.set(path)
            self._on_dataset_preset_change(preset)

            # rebuild layers UI from saved config
            self.clear_layers()
            for layer_cfg in config.get("layers", []):
                row = LayerRow(self.layers_container, remove_callback=self.remove_layer)
                # restore layer type first
                row.type_var.set(layer_cfg.get("type", "Dense"))

                # restore common fields safely
                if "neurons" in layer_cfg:
                    try:
                        row.neurons_var.set(int(layer_cfg.get("neurons", row.neurons_var.get())))
                    except Exception:
                        pass
                if "filters" in layer_cfg:
                    try:
                        row.neurons_var.set(int(layer_cfg.get("filters", row.neurons_var.get())))
                    except Exception:
                        pass
                if "activation" in layer_cfg:
                    row.activation_var.set(layer_cfg.get("activation", row.activation_var.get()))

                # conv2d kernels
                if "kernel_width" in layer_cfg:
                    try:
                        row.kernel_width_var.set(int(layer_cfg.get("kernel_width", row.kernel_width_var.get())))
                    except Exception:
                        pass
                if "kernel_height" in layer_cfg:
                    try:
                        row.kernel_height_var.set(int(layer_cfg.get("kernel_height", row.kernel_height_var.get())))
                    except Exception:
                        pass
                # conv1d / locallyconnected kernel_size
                if "kernel_size" in layer_cfg:
                    try:
                        row.kernel_size_var.set(int(layer_cfg.get("kernel_size", row.kernel_size_var.get())))
                    except Exception:
                        pass
                # dropout
                if "rate" in layer_cfg:
                    try:
                        row.dropout_var.set(float(layer_cfg.get("rate", row.dropout_var.get())))
                    except Exception:
                        pass
                # reshape
                if "shape" in layer_cfg:
                    row.reshape_var.set(layer_cfg.get("shape", row.reshape_var.get()))
                # pooling
                if "pool_size" in layer_cfg:
                    try:
                        row.pool_size_var.set(int(layer_cfg.get("pool_size", row.pool_size_var.get())))
                    except Exception:
                        pass
                if "pool_width" in layer_cfg:
                    try:
                        row.pool_width_var.set(int(layer_cfg.get("pool_width", row.pool_width_var.get())))
                    except Exception:
                        pass
                if "pool_height" in layer_cfg:
                    try:
                        row.pool_height_var.set(int(layer_cfg.get("pool_height", row.pool_height_var.get())))
                    except Exception:
                        pass
                if "stride" in layer_cfg:
                    try:
                        row.stride_var.set(int(layer_cfg.get("stride", row.stride_var.get())))
                    except Exception:
                        pass

                # ensure UI shows the correct controls for the chosen type
                row.update_fields()
                row.pack(fill="x", pady=2)
                self.layer_rows.append(row)

            self.log(f"Config '{name}' loaded.")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load config: {e}")
            self.log(f"Error loading config '{name}': {e}")

    # new helpers for persistence
    def _load_saved_configs(self):
        try:
            if os.path.isfile(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = f.read().strip()
                    if data:
                        # expect a JSON object mapping names->config (each config is a dict or legacy JSON-string)
                        parsed = json.loads(data)
                        if isinstance(parsed, dict):
                            self.saved_configs = parsed
                            self.log(f"Loaded {len(parsed)} saved config(s) from disk.")
        except Exception as e:
            self.log(f"Warning: failed to load saved configs: {e}")

    def _save_configs_to_file(self):
        try:
            # write the mapping as JSON to the text file
            tmp_path = self.config_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.saved_configs, f, indent=2)
            os.replace(tmp_path, self.config_file)
        except Exception as e:
            raise

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _on_dataset_preset_change(self, value):
        """
        Called when the dataset preset changes. Enables entry/browse only for Custom...
        For built-in presets we set the dataset_var to the preset label and disable editing.
        """
        if value == "Custom...":
            try:
                self.dataset_entry.config(state="normal")
                self.dataset_browse_btn.config(state="normal")
            except Exception:
                pass
            # leave dataset_var unchanged for user to choose a path
        elif value == "None":
            try:
                self.dataset_entry.config(state="disabled")
                self.dataset_browse_btn.config(state="disabled")
            except Exception:
                pass
            self.dataset_var.set("")
        else:
            # built-in preset selected
            try:
                self.dataset_entry.config(state="disabled")
                self.dataset_browse_btn.config(state="disabled")
            except Exception:
                pass
            self.dataset_var.set(value)
        self.log(f"Dataset preset changed: {value}")

    def browse_dataset(self):
        path = filedialog.askopenfilename(title="Select dataset file")
        if path:
            self.dataset_var.set(path)
            # if user browsed for a custom file, ensure preset reflects that
            if self.dataset_preset_var.get() != "Custom...":
                self.dataset_preset_var.set("Custom...")
                self._on_dataset_preset_change("Custom...")
            self.log(f"Dataset selected: {path}")

    def validate_config(self):
        try:
            epochs = int(self.epochs_var.get())
            if epochs < 1:
                raise ValueError("Epochs must be >= 1")
            lr = float(self.lr_var.get())
            if lr <= 0:
                raise ValueError("Learning rate must be > 0")

            # dataset validation:
            # - if a built-in preset is selected (in dataset_presets, excluding "None"), it's valid
            # - require a file only when preset is "Custom..." or when a non-empty path is provided
            ds_preset = self.dataset_preset_var.get()
            ds_path = self.dataset_var.get().strip()

            if ds_preset == "None":
                # no dataset selected -> acceptable
                pass
            elif ds_preset == "Custom...":
                # custom preset requires a file path
                if not ds_path:
                    raise ValueError("Custom dataset selected but no file provided")
                if not os.path.exists(ds_path):
                    raise ValueError("Selected dataset file does not exist")
            elif ds_preset in getattr(self, "dataset_presets", []):
                # built-in preset selected -> valid without checking path
                pass
            else:
                # unknown preset: if a path is provided require it to exist, otherwise treat as invalid
                if ds_path:
                    if not os.path.exists(ds_path):
                        raise ValueError("Selected dataset file does not exist")
                else:
                    raise ValueError(f"Unknown dataset preset '{ds_preset}'. Choose a preset or provide a dataset path.")

            if not self.layer_rows:
                raise ValueError("Add at least one layer")

            # validate each layer according to its reported type and params
            for i, r in enumerate(self.layer_rows):
                cfg = r.get_config()
                t = cfg.get("type", "").lower()
                if not t or t == "layer":
                    raise ValueError(f"Layer {i+1}: please select a layer type")

                if t in ("dense", "input"):
                    neurons = cfg.get("neurons")
                    if neurons is None:
                        raise ValueError(f"Layer {i+1} ({t}): missing 'neurons'")
                    if int(neurons) < 1:
                        raise ValueError(f"Layer {i+1} ({t}): 'neurons' must be >= 1")

                elif t in ("conv1d", "conv2d", "locallyconnected"):
                    filters = cfg.get("filters") or cfg.get("neurons")
                    if filters is None:
                        raise ValueError(f"Layer {i+1} ({t}): missing 'filters'")
                    if int(filters) < 1:
                        raise ValueError(f"Layer {i+1} ({t}): 'filters' must be >= 1")
                    # kernel size checks
                    if t == "conv2d":
                        kw = cfg.get("kernel_width")
                        kh = cfg.get("kernel_height")
                        if kw is None or kh is None:
                            raise ValueError(f"Layer {i+1} (Conv2D): missing kernel dimensions")
                        if int(kw) < 1 or int(kh) < 1:
                            raise ValueError(f"Layer {i+1} (Conv2D): kernel dimensions must be >= 1")
                    else:
                        ks = cfg.get("kernel_size")
                        if ks is None:
                            raise ValueError(f"Layer {i+1} ({t}): missing 'kernel_size'")
                        if int(ks) < 1:
                            raise ValueError(f"Layer {i+1} ({t}): 'kernel_size' must be >= 1")

                elif t == "dropout":
                    rate = cfg.get("rate")
                    if rate is None:
                        raise ValueError(f"Layer {i+1} (Dropout): missing 'rate'")
                    rfloat = float(rate)
                    if not (0.0 < rfloat < 1.0):
                        raise ValueError(f"Layer {i+1} (Dropout): 'rate' must be between 0 and 1")

                elif t == "reshape":
                    shape = cfg.get("shape", "")
                    if not shape:
                        raise ValueError(f"Layer {i+1} (Reshape): missing 'shape'")
                    # try parsing comma-separated ints
                    parts = [p.strip() for p in str(shape).split(",") if p.strip()]
                    if not parts:
                        raise ValueError(f"Layer {i+1} (Reshape): invalid 'shape'")
                    for p in parts:
                        try:
                            if int(p) < 1:
                                raise ValueError
                        except Exception:
                            raise ValueError(f"Layer {i+1} (Reshape): all dimensions must be positive integers")

                elif t == "maxpool1d":
                    ps = cfg.get("pool_size")
                    st = cfg.get("stride")
                    if ps is None or st is None:
                        raise ValueError(f"Layer {i+1} (Maxpool1D): missing pool_size or stride")
                    if int(ps) < 1 or int(st) < 1:
                        raise ValueError(f"Layer {i+1} (Maxpool1D): pool_size and stride must be >= 1")

                elif t == "maxpool2d":
                    pw = cfg.get("pool_width")
                    ph = cfg.get("pool_height")
                    st = cfg.get("stride")
                    if pw is None or ph is None or st is None:
                        raise ValueError(f"Layer {i+1} (Maxpool2D): missing pool dimensions or stride")
                    if int(pw) < 1 or int(ph) < 1 or int(st) < 1:
                        raise ValueError(f"Layer {i+1} (Maxpool2D): pool dimensions and stride must be >= 1")

                # flatten and other types require no extra checks

            return True, None
        except Exception as e:
            return False, str(e)

    def on_play(self):
        # validate configuration first
        ok, err = self.validate_config()
        if not ok:
            messagebox.showerror("Invalid configuration", err)
            return

        # build config dict (same structure used in save_config)
        config = {
            "epochs": int(self.epochs_var.get()),
            "learning_rate": float(self.lr_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "optimizer": self.opt_var.get(),
            "dataset": {"preset": self.dataset_preset_var.get(), "path": self.dataset_var.get()},
            "layers": [r.get_config() for r in self.layer_rows]
        }

        # prepare repo/build info
        repo_url = "https://github.com/modern-fortran/neural-fortran"
        repo_dir = "neural-fortran"
        build_dir = os.path.join(repo_dir, "build")

        # disable UI and start indeterminate progress
        self.set_ui_state(disabled=True)
        self.progress.config(mode="indeterminate")
        self.progress.start(10)
        self.status_label.config(text="Starting build...")

        # run build (and then training) in background thread and stream logs
        t = threading.Thread(target=self._build_thread, args=(repo_url, repo_dir, build_dir, config), daemon=True)
        t.start()

    def _run_command_stream(self, cmd, cwd=None):
        """
        Run a command and stream stdout/stderr line-by-line to the GUI log.
        Returns the process returncode.
        """
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, text=True)
        except Exception as e:
            # can't start process - log and return non-zero
            self.root.after(0, self.log, f"Failed to start: {' '.join(cmd)} -> {e}")
            return 1

        # read lines as they arrive
        for line in proc.stdout:
            self.root.after(0, self.log, line.rstrip())
        proc.wait()
        return proc.returncode

    def _build_thread(self, repo_url, repo_dir, build_dir, config):
        try:
            # If MNIST is selected we don't need to clone/build the neural-fortran repo;
            # directly run training which will generate the Fortran file.
            if config.get("dataset", {}).get("preset") == "MNIST":
                self.root.after(0, self.log, "MNIST selected — skipping repository build and running Fortran generation...")
                # call training (runs execute_fortran_mnist inside)
                self._train_thread(config)
                return

            # Clone if needed
            if not os.path.isdir(repo_dir):
                self.root.after(0, self.status_label.config, {"text": "Cloning repository..."})
                rc = self._run_command_stream(["git", "clone", repo_url])
                if rc != 0:
                    raise RuntimeError("git clone failed")

            # Ensure build dir exists
            if not os.path.isdir(build_dir):
                os.makedirs(build_dir)

            # Run cmake
            self.root.after(0, self.status_label.config, {"text": "Running cmake..."})
            rc = self._run_command_stream(["cmake", ".."], cwd=build_dir)
            if rc != 0:
                raise RuntimeError("cmake failed")

            # Run make
            self.root.after(0, self.status_label.config, {"text": "Running make..."})
            rc = self._run_command_stream(["make"], cwd=build_dir)
            if rc != 0:
                raise RuntimeError("make failed")

            # success
            self.root.after(0, self.log, "Build completed successfully.")
            self.root.after(0, self.status_label.config, {"text": "Build succeeded"})

            # After a successful build run training with the same config
            self.root.after(0, self.log, "Starting training...")
            self._train_thread(config)

        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, self.status_label.config, {"text": "Build failed"})
            self.root.after(0, messagebox.showerror, "Build error", str(e))
        finally:
            # stop progress and re-enable UI
            def finish():
                try:
                    self.progress.stop()
                    self.progress.config(mode="determinate")
                except Exception:
                    pass
                self.set_ui_state(disabled=False)
            self.root.after(0, finish)

    def set_ui_state(self, disabled: bool):
        state = "disabled" if disabled else "normal"
        # disable top-level inputs
        for widget in (self.epochs_spin, self.lr_entry, self.opt_menu, self.play_btn):
            try:
                widget.config(state=state)
            except Exception:
                pass
        # disable dataset controls
        try:
            self.dataset_preset_menu.config(state=state)
            # the entry and browse button state depend on preset; set them explicitly when disabling
            if disabled:
                self.dataset_entry.config(state="disabled")
                self.dataset_browse_btn.config(state="disabled")
            else:
                # enable/disable according to current preset
                self._on_dataset_preset_change(self.dataset_preset_var.get())
        except Exception:
            pass
        # disable layer controls
        for r in self.layer_rows:
            try:
                r.neurons_spin.config(state=state)
                r.activation_menu.config(state=state)
                r.remove_btn.config(state=state)
            except Exception:
                pass

    def _train_thread(self, config):
        if config["dataset"]["preset"] == "MNIST":
            try:
                out_path, content = execute_fortran_mnist(config)
                # log path and file contents to GUI (thread-safe)
                self.root.after(0, self.log, f"Fortran source written to: {out_path}")
                for line in str(content).splitlines():
                    self.root.after(0, self.log, line)
            except Exception as e:
                self.root.after(0, self.log, f"Error generating Fortran: {e}")
            finally:
                # mark training finished / re-enable UI
                self.root.after(0, self._training_finished)
            return
        else:
         # Placeholder for a real build/train call. This simulates training progress.
         epochs = config["epochs"]
         for ep in range(1, epochs + 1):
             time.sleep(0.2)  # simulate training time per epoch
             # update UI via thread-safe call
             self.root.after(0, self._update_progress, ep, epochs)
             self.root.after(0, self.log, f"Epoch {ep}/{epochs} completed")
         # done
         self.root.after(0, self._training_finished)

    def _update_progress(self, epoch, total):
        self.progress["value"] = epoch
        self.status_label.config(text=f"Epoch {epoch}/{total}")

    def _training_finished(self):
        self.set_ui_state(disabled=False)
        self.status_label.config(text="Done")
        self.log("Training finished (simulated).")