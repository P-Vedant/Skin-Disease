import os
import platform

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def note(msg):
    print(msg)
    input("Press enter to continue.\n")
    clear_screen()

def err(msg):
    raise msg

def get_inp(msg, valid, err, dtype="str"):
    msg=msg+"\n>>> "
    while True:
        try:
            if dtype=="int":
                inp=int(input(msg))
            elif dtype=="float":
                inp=float(input(msg))
            else:
                inp=input(msg)
            
            if inp in valid:
                clear_screen()
                return inp
            else:
                clear_screen()
                note(err)
        except Exception as e:
            clear_screen()
            note(err)
