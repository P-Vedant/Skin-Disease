import functools
import os

def build_model(conv_filters, conv_sizes, conv_stridesX, conv_stridesY, conv_LReLU_negative_slopes, conv_pool_sizes, dense_sizes, dense_activation_methods, img_size, drop_out, tf):
    seq=[tf.keras.layers.Input(shape=(int(img_size), int(img_size), 3))]

    for i in range(0, len(conv_filters)):
        seq.append(
            tf.keras.layers.Conv2D(
                filters=int(conv_filters[i]),
                kernel_size=(int(conv_sizes[i]),int(conv_sizes[i])),
                strides=(int(conv_stridesX[i]),int(conv_stridesY[i])),
                activation=None,
                padding="same",
                use_bias=True
            )
        )
        seq.append(tf.keras.layers.LeakyReLU(negative_slope=float(conv_LReLU_negative_slopes[i])))

        seq.append(
            tf.keras.layers.MaxPooling2D((int(conv_pool_sizes[i]),int(conv_pool_sizes[i])))
        )
    
    seq.append(tf.keras.layers.GlobalAveragePooling2D())
    seq.append(tf.keras.layers.Dropout(float(drop_out)))

    for i in range(0, len(dense_sizes)):
        seq.append(
            tf.keras.layers.Dense(int(dense_sizes[i]), activation=dense_activation_methods[i])
        )
    
    return tf.keras.models.Sequential(seq)

def compile_model(model):
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

def train_model(model, train_dataset, test_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    return model.evaluate(test_dataset, verbose=0)

def crop_image(img, size, tf):
    shape=tf.shape(img)
    if shape[0]==size and shape[1]==size:
        return tf.cast(img, tf.float32)
    if shape[0]<size or shape[1]<size:
        return tf.cast(tf.image.resize(img, [size, size], method="bilinear"), tf.float32)
    offx=(shape[1]-size)//2
    offy=(shape[0]-size)//2
    return tf.cast(tf.image.crop_to_bounding_box(img, offy, offx, size, size), tf.float32)

def process_image(img, lbl, contrast_strength=1, size=1, tf=None):
    img=crop_image(img, size, tf)
    img=tf.cast(img, tf.float32)/255.0
    img=(tf.math.tanh((img-0.5)*contrast_strength)+1)/2 #use smooth contrast, as this is better for medical scans (varied lighting, critical mid-tones)
    img=tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def format_img(img_path, tf):
    img=tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
    img.set_shape([None, None, 3])
    return img, tf.strings.to_number(tf.strings.split(img_path, os.sep)[-2], tf.int32)

def collect_dir(dir, tf):
    return tf.data.Dataset.list_files(str(dir+"/*/*.jpg"), shuffle=False).map(functools.partial(format_img, tf=tf))

def collect_data(image_scale, training_batch_size, testing_batch_size, contrast_strength, shuffle_level, tf):
    image_scale=int(image_scale)

    print("Collecting testing data...")
    test_dataset = collect_dir("Database/Test", tf)

    print("Collecting training data...")
    train_dataset = collect_dir("Database/Train", tf)

    print("Processing data...")
    train_dataset=train_dataset.map(functools.partial(process_image, contrast_strength=float(contrast_strength), size=image_scale, tf=tf))
    test_dataset=test_dataset.map(functools.partial(process_image, contrast_strength=float(contrast_strength), size=image_scale, tf=tf))

    print("Configuring dataset objects pipeline modes...")
    train_dataset=train_dataset.shuffle(int(shuffle_level)).batch(int(training_batch_size)).prefetch(tf.data.AUTOTUNE)
    test_dataset=test_dataset.batch(int(testing_batch_size)).prefetch(tf.data.AUTOTUNE)

    return (train_dataset, test_dataset)
