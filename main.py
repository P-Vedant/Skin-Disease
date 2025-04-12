import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import user_interface as ui
import model_builder as mb
import server as sv
import shutil
import os

def load_config():
    config_raw=None

    with open("config.txt","r") as f:
        config_raw=f.read()
    
    config_raw=config_raw.split("\n")
    
    config={}

    for line in config_raw:
        try:
            line=line.strip()
            line=line.split(":")
            config[line[0]]=line[1]
        except Exception as e:
            ui.note(f"Config error: {line}")
    
    return config

def init():
    if not os.path.exists("is_live.txt"):
        print("Writting program data...")
        with open("is_live.txt","w") as f:
            f.write("1")
    else:
        status=None
        with open("is_live.txt","r") as f:
            status=f.read()
        if status=="0":
            with open("is_live.txt","w") as f:
                f.write("1")
        else:
            return None
    
    print("Initializing tensorflow...")
    import tensorflow as tf
    
    config={}
    
    print("Loading configuration file to a python dictionary...")

    try:
        config=load_config()
    except Exception as e:
        ui.err(e)
    
    print("Checking environemnt...")

    if not os.path.isdir("Models"):
        print("Creating models directory...")
        os.makedirs("Models")
    if not os.path.exists("Models/model_version.txt"):
        print("Creating version file...")
        with open("Models/model_version.txt","w") as f:
            f.write("0")
    if not os.path.isdir("Database"):
        while True:
            data_path=input("Action needed!\n\nPlease provide the path to the dataset directory you would like to use:\n>>>")
            if os.path.isdir(data_path):
                break
            elif os.path.exists(data_path):
                ui.note("The specified path doesn't point to a readable directory.\nIf the dataset folder is zipped, please unzip it.")
            else:
                ui.note("The specified path does not exist.\nDouble check that you typed it in correctly, and use forwrd slashes when you do so.")
            shutil.copytree(data_path,"Database")
            
    last_model_version=None
    with open("Models/model_version.txt","r") as f:
        last_model_version=f.read()
    try:
        last_model_version=int(last_model_version)
    except Exception as e:
        ui.note("WARNING: the versions folder is corrupt.\nIf you had any saved models,\nthey will be overwritten by new ones unless\nthis issue is resolved.")
        with open("Models/model_version.txt","w") as f:
            f.write("0")
    ui.note("Your terminal environemnt is now prepared.")

    main(config, tf)

def select_model_UI(tf):
    print("Searching for models...")
    file_paths=[]
    selected_path=None
    versions=0
    with open("Models/model_version.txt","r") as f:
        versions=int(f.read())
    for i in range(0, versions+1):
        file_path=f"Models/model_{i}.keras"
        if os.path.exists(file_path):
            file_paths.append(file_path)
    while True:
        if len(file_paths)==0:
            ui.note("There exists no operational models which the search could uncover.\nYou will either have to train a model, or manualy download one.\nIf you know you saved a model, check the models folder.\nIf at any point you changed or deleted the `Models/model_version.txt` file,\nsome models may no longer be discoverable by this program.")
            return None, None
        elif len(file_paths)==1:
            if input(f"Search found exactly one trained model.\nUse `{file_paths[0]}`?\n[Y/n]\n>>> ")=="Y":
                selected_path=file_paths[0]
        else:
            msg="Please select one of the following paths:"
            for i in range(0, len(file_paths)):
                msg+=f"\n{i+1}. {file_paths[i]}"
            selected_path=file_paths[ui.get_inp(msg, range(1,len(file_paths)+1), "Please enter a number key.", "int")-1]
        try:
            return selected_path, tf.keras.models.load_model(selected_path)
        except Exception as e:
            ui.note(f"Model loading failed.\nThis may be due to corruption of the file. Please select a diffrent model or try again.\nError message: {e}")
    

def main(config, tf):
    model=None
    model_file=None

    while True:
        command=ui.get_inp("Please select an option:\n1. Train a model\n2. Build a model\n3. Save a model\n4. Select a model\n5. Open the server\n6. Quit", (1,2,3,4,5,6), "Please enter a number key between 1 and 3.", "int")
        if command==1:
            if model is None:
                ui.note("Please build or select a model first.")
            print("Collecting data...")
            train_dataset, test_dataset=mb.collect_data(
                config["pre.image_scale"], 
                config["pre.train_batch_size"], 
                config["pre.test_batch_size"], 
                config["pre.contrast_strength"],
                config["pre.shuffle_level"],
                tf
                )
            epochs=ui.get_inp("Please enter the epoch number (between 1 and 100 inclusive):", range(1,101), "Please enter a number key between 1 and 100 (inclusive).","int")
            print("Starting training sequence...")
            print(mb.train_model(model, train_dataset, test_dataset, epochs))
            ui.note("Model trained!\nMake sure to save if you want to keep the progress!")
        elif command==2:
            print("Building model...")
            model=model=mb.build_model(
                config["mod.conv_filters"].split(","),
                config["mod.conv_sizes"].split(","),
                config["mod.conv_stridesX"].split(","),
                config["mod.conv_stridesY"].split(","),
                config["mod.conv_LReLU_negative_slopes"].split(","),
                config["mod.conv_pool_sizes"].split(","),
                config["mod.dense_sizes"].split(","),
                config["mod.dense_activation_methods"].split(","),
                config["pre.image_scale"],
                config["mod.drop_out"],
                tf
            )
            print("Compiling model...")
            mb.compile_model(model)
            ui.note("Model is built and compiled!")
        elif command==3:
            print("Saving model...")
            try:
                if (not model_file is None) and input("Would you like to replace the previous save of this model with the new save?\n[Y/n]\n>>> ")=="Y":
                    model.save(model_file)
                last_version=0
                with open("Models/model_version.txt","r") as f:
                    last_version=int(f.read())
                with open("Models/model_version.txt","w") as f:
                    f.write(str(last_version+1))
                model.save(f"Models/model_{last_version}.keras")
            except Exception as e:
                if input(f"WARNING: normal saving method failed.\nError: {e}\n\nSave to temp.h5 folder instead?\n(caution: this will overwrite its contents if it exists)\n[Y/n]\n>>> ")=="Y":
                    model.save("temp.keras")
            ui.note("Model saved!")
        elif command==4:
            model_file, model=select_model_UI(tf)
            mb.compile_model(model)
            if not model is None:
                ui.note("Model selected and compiled!")
        elif command==5:
            sv.init_flask(model, config, tf)
        else:
            print("Quitting...")
            with open("is_live.txt","w") as f:
                f.write("0")
            quit()

if __name__=="__main__":
    init()
