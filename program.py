from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'genus/train',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'genus/validation',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'genus/test',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes = 25


model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
#model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')

model.save('model.keras')
