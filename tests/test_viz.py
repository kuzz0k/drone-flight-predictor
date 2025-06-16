#!/usr/bin/env python3
print("Starting test script...")

try:
    import matplotlib
    print("✓ Matplotlib imported")
    
    matplotlib.use('Agg')
    print("✓ Backend set to Agg")
    
    import matplotlib.pyplot as plt
    print("✓ Pyplot imported")
    
    import numpy as np
    print("✓ Numpy imported")
    
    # Test plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 4, 9, 16]
    ax.plot(x, y, 'bo-')
    ax.set_title('Test Plot')
    plt.savefig('test_plot.png')
    plt.close()
    print("✓ Test plot saved as test_plot.png")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test script finished.")
