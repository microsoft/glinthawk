try:
    from settings_local import *
except ImportError:
    print("No local settings found")
    pass
