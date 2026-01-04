import tkinter as tk
from ui import NNBuilderApp


def main():
    root = tk.Tk()
    app = NNBuilderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz