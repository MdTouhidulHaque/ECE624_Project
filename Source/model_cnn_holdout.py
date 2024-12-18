from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt 
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
)


def create_model():
    # Define the CNN model with L2 regularization
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flattening and Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(7, activation='softmax')  # 7 classes for classification
    ])

    # Compile the model
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return model


def load_images_and_labels(data_dir, img_size):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=img_size)
            image = img_to_array(image) / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

def train_model(epochs, img_size, num_classes, model_path):
        
    batch_size = 32
    
    # Load data
    X, y = load_images_and_labels(train_data_dir, img_size)
    y = to_categorical(y, num_classes=num_classes)          #one hot encoding
    model = create_model()
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)       #learning rate= .001 to .00001
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=624)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[lr_scheduler]
    )

    # Store training loss and accuracy, and validation loss and accuracy for each epoch
    train_loss.extend(history.history['loss'])
    train_accuracy.extend(history.history['accuracy'])
    val_loss.extend(history.history['val_loss'])
    val_accuracy.extend(history.history['val_accuracy'])

    # Evaluate on validation data
    final_val_loss, final_val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Hold-out Validation Accuracy: {final_val_accuracy:.4f}")

    # Save the model
    model.save(model_path)
    print(f"Model saved at {model_path}")

    #save them for further use
    np.save(f'../results/{training_on}/train_loss', train_loss)
    np.save(f'../results/{training_on}/train_accuracy', train_accuracy)
    np.save(f'../results/{training_on}/val_loss', val_loss)
    np.save(f'../results/{training_on}/val_accuracy', val_accuracy)
    return 

epochs= 140 
img_size = (96, 96)
num_classes = 7
#training_on= 'fuzzy_segmented'      #fuzzy_segmented or normal_unsegmented
training_on= 'normal_unsegmented'

if (training_on =='fuzzy_segmented'):   train_data_dir = '../data/segmented_train_balanced_upsampled_3000'
elif (training_on == 'normal_unsegmented'):     train_data_dir = '../data/train_balanced_upsampled_3000'
    

#uncomment below to train model and save values for validation loss, accu and train loss, accu
#model_path = f'../saved_models/saved_model_{training_on}.h5'
#train_model(epochs, img_size, num_classes, model_path)

#Evaluate model on unseen test data
if (training_on =='normal_unsegmented'):   test_data_dir = '../data/test'
elif (training_on == 'fuzzy_segmented'):     test_data_dir = '../data/segmented_images_test'

test_images, test_labels= load_images_and_labels (test_data_dir, img_size)
test_labels= to_categorical(test_labels, num_classes=num_classes) 
mpath= f'../saved_models/saved_model_{training_on}.h5'
model= load_model(mpath)
loss, accu= model.evaluate(test_images, test_labels)
print(f'Test accuracy {training_on}: {accu}')

#load presaved data
train_loss= np.load (f'../results/{training_on}/train_loss.npy')
train_accuracy= np.load (f'../results/{training_on}/train_accuracy.npy')
val_loss= np.load (f'../results/{training_on}/val_loss.npy')
val_accuracy= np.load (f'../results/{training_on}/val_accuracy.npy')



plt.figure(figsize=(4, 3))
plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', linewidth=2, color='red')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', linewidth=2, color='blue')
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Loss', fontsize=16, fontweight='bold')
plt.title('Loss vs. Epoch (Training and Validation)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
#plt.grid(True)
plt.tight_layout()
#plt.show()

# Subplot 2: Train vs. Validation Accuracy
plt.figure(figsize=(4, 3))
plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy',linewidth=2, color='green')
plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy', linewidth=2, color='orange')
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
plt.title('Accuracy vs. Epoch (Training and Validation)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()


def calculate_metrics(model, test_images, test_labels, class_names, normalize=True):
    y_pred = model.predict(test_images, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)
    
    # Classification report for individual class metrics
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
    
    # Print individual class metrics
    print("Individual Class Metrics:")
    for class_name in class_names:
        class_report = report[class_name]
        print(f"{class_name} - Precision: {class_report['precision']:.4f}, "
              f"Recall: {class_report['recall']:.4f}, "
              f"F1-Score: {class_report['f1-score']:.4f}, "
              f"Support: {class_report['support']}")
    
    # Weighted average metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )
    print("\nWeighted Average Metrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f" if normalize else "d")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    return precision, recall, f1_score, report
    
    

def plot_auc_roc_curve_test(model, X_test, y_test, num_classes, class_names):
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title('ROC Curves for Test Data', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18, prop={'weight': 'bold'})
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    #plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


# Load the actual class names from the test directory
class_names = sorted(os.listdir(test_data_dir))
#print(class_names)

# Plot AUC ROC Curve for Test Data
plot_auc_roc_curve_test(model, test_images, test_labels, num_classes, class_names)


calculate_metrics(model, test_images, test_labels, class_names)
