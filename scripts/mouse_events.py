import threading

# get the thread id for the hook: threading.current_thread().ident 

from pynput.mouse import Button, Controller

mouse = Controller()

# Read pointer position
print('The current pointer position is {0}'.format(
    mouse.position))

# Set pointer position
#mouse.position = (10, 20)
#print('Now we have moved it to {0}'.format(
    #mouse.position))

# Press and release
mouse.press(Button.right)
mouse.release(Button.right)