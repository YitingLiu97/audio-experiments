import sys
print(f"Using Python at: {sys.executable}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    import matplotlib.pyplot as plt
    
    # Create a simple plot
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.title("Test Plot")
    plt.show(block=False)
    print("✓ Matplotlib is working!")
    
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
    print("\nPlease check your Python environment:")
    print("1. Make sure you're using the Anaconda environment")
    print("2. Verify matplotlib is installed")
    print("3. Check the Python path in Cursor")