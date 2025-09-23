# model_fixer.py - CORRECTED FOR 5 CLASSES
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_fixed_sugarcane_model(num_classes=5):  # Changed to 5!
    """Creates a properly structured ResNet50 model for 5-class sugarcane disease classification"""
    print("Creating ResNet50 base model...")
    
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_tensor=Input(shape=(224, 224, 3)))
    
    print(f"Base model output shape: {base_model.output.shape}")
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(512, activation='relu', name='dense_512')(x)
    x = Dropout(0.5, name='dropout_0.5')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = Dropout(0.3, name='dropout_0.3')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)  # 5 classes!
    
    # Create the final model using Model() - NOT Sequential()!
    model = Model(inputs=base_model.input, outputs=predictions, name='sugarcane_resnet50_fixed')
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def main():
    print("🌾 CREATING FIXED SUGARCANE MODEL - 5 CLASSES")
    print("=" * 100)
    
    # Your actual 5 disease classes from the notebook
    DISEASE_CLASSES = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
    num_classes = len(DISEASE_CLASSES)
    
    print(f"Disease classes: {DISEASE_CLASSES}")
    print(f"Number of classes: {num_classes}")
    
    # Create the fixed model
    model = create_fixed_sugarcane_model(num_classes)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save the fixed model
    model.save('sugarcane_disease_FIXED.keras')
    print("✅ Fixed model saved as: sugarcane_disease_FIXED.keras")
    
    # Test loading
    test_model = tf.keras.models.load_model('sugarcane_disease_FIXED.keras')
    print("✅ Model loads successfully!")
    print("✅ Model is ready for 5-class predictions!")

if __name__ == "__main__":
    main()
