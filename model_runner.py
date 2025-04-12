import model_builder as mb

def run_model(model, path, config, tf):
    img, _=mb.process_image(tf.keras.utils.load_img(path), None, float(config["pre.contrast_strength"]), int(config["pre.image_scale"]), tf)
    img=tf.expand_dims(img, axis=0)
    return model.predict(img)
